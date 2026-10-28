import logging, os, time, inspect
import math
import warnings
import numpy as np
from typing import Dict, Optional, Union, Any

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.linalg
import paddle.optimizer
import paddle.io
from paddle.base.libpaddle.eager import Tensor

from ..registry import register_symbolnet, detect_input_ops
from SemTrans.testing_and_repair.logging import PADDLE_LOG
from SemTrans.math_symbol_base.frontends.abstract.dtype import DType
from SemTrans.math_symbol_base.frontends.abstract.op import AbsOpBase
from SemTrans.math_symbol_base.frontends.abstract.gir import (
    InstExpr,
    Instruction,
    binary_associative_chain,
)
from SemTrans.math_symbol_base.frontends.abstract.tensor import AbsTensor
from SemTrans.math_symbol_base.frontends.abstract.error import ConstraintCheck, ConstraintError
from SemTrans.testing_and_repair.util import SourceLocator

from .paddleop import PaddleOpFactory, PaddleAdapter, PaddleFX
from .numeric import OpLossManager, numeric_valid

__INPUT_FOUND_NAN_MSG__ = "[NaN] in model inputs!"
__INPUT_FOUND_INF_MSG__ = "[Inf] in model inputs!"


# Debug decorator: Skip function execution in tracing mode
def debug_numeric(tensor_map):
    """Check for NaN or Inf values in the tensor mapping."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ConstraintCheck.true(
            not any([paddle.isinf(tensor).any().item() for tensor in tensor_map.values()]),
            __INPUT_FOUND_INF_MSG__,
        )
        ConstraintCheck.true(
            not any([paddle.isnan(tensor).any().item() for tensor in tensor_map.values()]),
            __INPUT_FOUND_NAN_MSG__,
        )


def check_type(op: AbsOpBase, tensors, is_input=True, msg=""):
    """Validate whether the concrete tensor matches the abstract type definition."""
    if op is not None:
        like = op.input_like if is_input else op.output_like
        ioro = "input" if is_input else "output"

        assert len(like) == len(
            tensors
        ), f"{op}'s {ioro} has {len(like)} abs. input, but got {len(tensors)} real inputs."

        for i, tensor in enumerate(tensors):
            ttype = AbsTensor(list(tensor.shape), DType.from_paddle(tensor.dtype))
            assert (
                    like[i] == ttype
            ), f"{msg} {ioro} abstract type != concrete {ttype} for {op} {op.input_like} -> {op.output_like}"


def debug_io(stmt_idx, itensors, otensors):
    if PADDLE_LOG.isEnabledFor(logging.DEBUG):
        PADDLE_LOG.debug(f">> statment {stmt_idx}")
        for inp_i, i in enumerate(itensors):
            PADDLE_LOG.debug(f"  (shape={i.shape} dtype={i.dtype})")
            PADDLE_LOG.debug(f"[inp]@{inp_i} :: {i.min().data:.5f} ~ {i.max().data:.5f}")
        for out_i, o in enumerate(otensors):
            PADDLE_LOG.debug(f"  (shape={o.shape} dtype={o.dtype})")
            PADDLE_LOG.debug(f"[out]@{out_i} :: {o.min().data:.5f} ~ {o.max().data:.5f}")


@register_symbolnet("paddle")
class SymbolNet(nn.Layer):
    def __init__(self, model: nn.Layer, framework_type, use_gradient=False, force_input_map=None):
        super(SymbolNet, self).__init__()
        self.check_intermediate_numeric = None
        self.differentiable = None
        self.optimizer = None
        self.loss = None
        self.to_train = None
        self.use_gradient = None
        self.framework_type = framework_type
        self.instructions = []
        self.model = model
        self.device = None
        self.proxy_enabled = False
        if force_input_map is not None:
            self.input_map = force_input_map
        else:
            self.input_map = detect_input_ops(self.model)
        self.gradient = use_gradient
        if use_gradient:
            self.enable_training()
        self.adapter = PaddleAdapter
        self.locator = SourceLocator(self.model)

    def get_layer_location(self, param_name):
        return self.locator.get_location(param_name)

    def to(self, device=None, dtype=None, blocking=True, **kwargs):
        if device is not None and isinstance(device, str):
            device = device.lower()
            self._device = device

        return super().to(device=device, dtype=dtype, **kwargs)

    def to_tensor(self, data) -> Union[dict[Any, Tensor], Any]:
        if isinstance(data, (list, tuple)):
            return type(data)(self.to_tensor(x) for x in data)
        if isinstance(data, dict):
            return {k: self.to_tensor(v) for k, v in data.items()}

        if isinstance(data, paddle.Tensor):
            tensor = data
        else:
            tensor = paddle.to_tensor(data)

        if self.device is not None:
            tensor = tensor.to(self.device)

        return tensor

    def to_numpy(self, output):
        results = {}

        def to_np(x):
            if isinstance(x, paddle.Tensor):
                return x.detach().cpu().numpy()
            return np.array(x)

        if isinstance(output, (list, tuple)):
            for i, val in enumerate(output):
                results[f"output_{i}"] = to_np(val)
        elif isinstance(output, dict):
            for k, v in output.items():
                results[k] = to_np(v)
        else:
            results["output_0"] = to_np(output)

        return results

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = value

    def get_op_key(self, inst):
        return inst.iexpr.op.op_type

    def get_runner(self, inst):
        return inst.runner

    @property
    def input_like(self):
        return self.input_map

    def set_seed(self, seed):
        paddle.seed(seed)

    def stop_grad(self):
        return paddle.no_grad()

    def _dtype_size(self, dtype):
        size_map = {
            'float32': 4,
            'float64': 8,
            'int32': 4,
            'int64': 8,
            'bool': 1
        }
        if isinstance(dtype, str):
            return size_map.get(dtype, 4)
        else:
            dtype_str = str(dtype).split('.')[-1]
            return size_map.get(dtype_str, 4)

    def empty_cache(self):
        paddle.device.cuda.empty_cache()

    def get_weights(self) -> Dict[str, np.ndarray]:
        weights = {}
        for name, param in self.model.state_dict().items():
            weights[name] = param.numpy()

        return weights

    def inject_value(self, target_tensor, source_numpy_data) -> bool:
        if target_tensor is None or source_numpy_data is None:
            return False

        try:

            if isinstance(source_numpy_data, paddle.Tensor):
                val = source_numpy_data
            else:
                val = paddle.to_tensor(source_numpy_data)

            if val.dtype != target_tensor.dtype:
                val = val.astype(target_tensor.dtype)

            if val.shape != target_tensor.shape:
                val = val.reshape(target_tensor.shape)

            paddle.assign(val, output=target_tensor)
            return True

        except Exception as e:
            print(f"[SymNet Error] Paddle inject failed: {e}")
            return False

    def get_layer_info(self) -> Dict[str, Dict[str, str]]:
        layer_info = {}
        for name, sublayer in self.model.named_sublayers(include_self=True):
            layer_type = sublayer.__class__.__name__
            for param_name, _ in sublayer.named_parameters(include_sublayers=False):
                full_param_name = f"{name}.{param_name}" if name else param_name

                layer_info[full_param_name] = {
                    'type': layer_type,
                    'parent': name
                }

        return layer_info

    def calculate_max_sample(self, MAX_MEM, MEM_FACTOR=0.4):
        nbytes = 0

        for iname, ii in self.input_like.items():
            if hasattr(ii, "numel"):
                elem_count = int(ii.numel())
            else:
                elem_count = int(np.prod(ii.shape))

            nbytes += elem_count * self._dtype_size(ii.dtype)

        for name, param in self.named_parameters():
            nbytes += int(param.numel()) * self._dtype_size(param.dtype)

        if nbytes == 0: nbytes = 1
        max_sample = int(MAX_MEM * MEM_FACTOR / nbytes)

        return max_sample, nbytes * max_sample / (1024 ** 3)

    def get_weight_tensor(self, shape, dtype, use_gpu=False):
        if isinstance(dtype, DType):
            dtype_str = dtype.name
        elif isinstance(dtype, str):
            dtype_str = dtype
        else:
            dtype_str = str(dtype).replace("paddle.", "")

        pd_dtype = getattr(paddle, dtype_str.replace("float", "float32"), paddle.float32)

        concrete_shape = [1 if (s == -1 or s == 'int') else int(s) for s in shape]

        tensor = paddle.randn(concrete_shape, dtype=pd_dtype)

        ndim = len(concrete_shape)
        if ndim >= 2:
            fan_in = 1.0

            if ndim == 2:
                fan_in = concrete_shape[0]
            elif ndim == 4:
                fan_in = concrete_shape[1] * concrete_shape[2] * concrete_shape[3]
            else:
                fan_in = concrete_shape[1]
                for s in concrete_shape[2:]:
                    fan_in *= s

            fan_in = max(fan_in, 1)
            std = 1.0 / math.sqrt(fan_in)
            tensor = tensor * std

        elif ndim == 1:
            tensor = paddle.uniform(concrete_shape, min=-1e-6, max=1e-6, dtype=pd_dtype)

        return tensor

    def random_tensor(self, shape, dtype, margin=2, base=0, use_gpu=False):
        if isinstance(dtype, DType):
            dt_obj = dtype
        elif isinstance(dtype, str):
            dt_obj = DType.from_str(dtype)
        elif isinstance(dtype, paddle.dtype):
            dt_obj = DType.from_paddle(dtype)
        else:
            raise ValueError(f"Unsupported dtype input: {type(dtype)} - {dtype}")

        paddle_dtype = dt_obj.paddle()

        concrete_shape = []
        for s in shape:
            if s == -1 or (isinstance(s, str) and not s.isdigit()):
                concrete_shape.append(1)
            else:
                concrete_shape.append(int(s))

        target_place_str = 'gpu:0' if use_gpu and paddle.is_compiled_with_cuda() else 'cpu'
        paddle.set_device(target_place_str)

        if base == "center":
            base = -margin / 2
        else:
            assert isinstance(base, (int, float))

        rand_data = paddle.rand(concrete_shape, dtype='float32')
        if dt_obj.is_float():
            fp_tensor = base + (rand_data - 0.5) * margin
            res = fp_tensor.astype(paddle_dtype)
        elif dt_obj == DType.bool:
            res = (rand_data > 0.5).astype(paddle_dtype)
        else:
            fp_tensor = base + (rand_data - 0.5) * margin
            res = paddle.round(fp_tensor).astype(paddle_dtype)

        return res

    def clear_cache(self):
        import gc
        gc.collect()

        if paddle.device.is_compiled_with_cuda():
            paddle.device.cuda.empty_cache()

    def stop_training(self):
        self.use_gradient = False
        self.loss = None

    def make_param_map(self) -> Dict[str, paddle.Tensor]:
        tensor_map: Dict[str, paddle.Tensor] = {}

        for k, v in self._parameters.items():
            if hasattr(self, k):
                attr = getattr(self, k)
                if isinstance(attr, paddle.Tensor):
                    tensor_map[k] = attr
                    continue

            tensor_map[k] = v

        if hasattr(self, 'static_param_map'):
            for static_name, tensor in self.static_param_map.items():
                tensor_map[static_name] = tensor
        return tensor_map

    def enable_training(self, extra_trainable: Dict[str, paddle.Tensor] = {}):
        self.use_gradient = True
        self.to_train = []

        for t in extra_trainable.values():
            self.to_train.append(t)

        for t in self.parameters():
            self.to_train.append(t)

        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)

        self.optimizer = paddle.optimizer.Adam(
            learning_rate=5e-1,
            parameters=self.to_train,
            grad_clip=clip
        )

        self.training_reset()

    def training_reset(self):
        if self.optimizer:
            self.optimizer.clear_grad()
        self.loss = None
        self.stop_updating_loss = False

    def reset_optimizer(self):
        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
        self.optimizer = paddle.optimizer.Adam(
            learning_rate=5e-1,
            parameters=self.to_train,
            grad_clip=clip
        )

    def get_random_inps(self, use_cuda=False):
        # center: -margin ~ 0 ~ +margin
        inputs = {}
        for key, abs_tensor in self.input_like.items():
            inputs[key] = self.random_tensor(
                abs_tensor.shape,
                abs_tensor.dtype,
                use_gpu=use_cuda
            )
        return inputs

    def grad_search(self, init_tensor_samples=None, init_weight_samples=None, use_cuda=False):
        from .input_gen import GradSearch

        return GradSearch(self, init_tensor_samples, init_weight_samples, use_cuda)

    def init_trace(self):
        from .init_trace import InitTracer
        ctx = getattr(self, 'diagnosis_ctx', {})
        entry_file = ctx.get('entry_file')

        user_dirs = []
        if entry_file:
            model_dir = os.path.dirname(os.path.abspath(entry_file))
            user_dirs.append(model_dir)

        return InitTracer(user_dirs=user_dirs)

    def grad_input_gen(
            self,
            init_tensors: Dict[str, paddle.Tensor],
            use_cuda=False,
            max_time=None,
    ) -> Optional[Dict[str, paddle.Tensor]]:
        if max_time is None:
            max_time = float(os.getenv("NNSMITH_GRAD_TIME", 0.5))

        # 1. Prepare dummy_inputs (extract values from init_tensors, preserving order)
        # Assumption: The order of init_tensors corresponds to the parameter order of the model's forward method
        dummy_inputs = list(init_tensors.values())

        # 2. Call the parsing function
        # This populates self.instructions and self.graph_input_names
        # It also fixes the computation graph structure for the current optimization process
        self.trace_model_to_instructions(self.model, dummy_inputs)

        inputs: Dict[str, paddle.Tensor] = {}
        for key, tensor in init_tensors.items():
            if tensor.dtype in [paddle.float16, paddle.float32, paddle.float64]:
                t = tensor.clone()
                t.stop_gradient = False
                inputs[key] = t
            else:
                inputs[key] = tensor.clone()

        self.enable_training(extra_trainable=inputs)

        last_check_intermediate_numeric = self.check_intermediate_numeric
        self.check_intermediate_numeric = True

        if use_cuda:
            if not paddle.device.is_compiled_with_cuda():
                print("[WARN] use_cuda=True but Paddle is not compiled with CUDA. Fallback to CPU.")
            else:
                for k in inputs:
                    inputs[k] = inputs[k].cuda()

                    if inputs[k].dtype in [paddle.float16, paddle.float32, paddle.float64]:
                        inputs[k].stop_gradient = False

        sat_inputs = None
        st = time.time()
        self.n_iter = 0
        self.cur_loss_name = None

        while time.time() - st < max_time:
            self.training_reset()
            self.n_iter += 1

            with paddle.no_grad():
                for inp in inputs.values():
                    if inp.dtype in [paddle.float16, paddle.float32, paddle.float64]:
                        clipped = paddle.clip(inp, -10.0, 10.0)
                        paddle.assign(clipped, output=inp)

            try:
                _ = self.forward_grad(*inputs.values())

                if self.invalid_found_last:
                    self.backward()
                else:
                    sat_inputs = {k: v.detach() for k, v in inputs.items()}
                    break
            except ConstraintError as e:
                if __INPUT_FOUND_INF_MSG__ in str(e) or __INPUT_FOUND_NAN_MSG__ in str(e):
                    with paddle.no_grad():
                        for inp in inputs.values():

                            is_nan_inf = paddle.logical_or(inp.isnan(), inp.isinf())

                            noise = self.random_tensor(
                                shape=inp.shape, dtype=inp.dtype
                            )
                            if inp.place.is_gpu_place():
                                noise = noise.cuda()

                            new_data = paddle.where(is_nan_inf, noise, inp)

                            paddle.assign(new_data, output=inp)
                    continue
                if 'PADDLE_LOG' in globals():
                    PADDLE_LOG.debug(e)
                else:
                    print(f"[DEBUG] {e}")
                break

        self.stop_training()
        if sat_inputs is None:
            msg = "[grad] no valid range found!!!"
            if 'PADDLE_LOG' in globals():
                PADDLE_LOG.debug(msg)
            else:
                print(msg)

        self.check_intermediate_numeric = last_check_intermediate_numeric
        return sat_inputs

    def forward(self, *args, **kwargs):
        self.differentiable = True
        return self.model(*args, **kwargs)

    def backward(self):
        if self.loss is not None:
            loss_val = self.loss[1]
            loss_val.backward()
            self.optimizer.step()
            self.optimizer.clear_grad()

    def _resolve_paddle_runner(self, op_type, raw_attrs):
        """
        [Paddle] Zero-Map Strategy: Reflection-based dynamic API lookup.
        Maps op.type (e.g., "elementwise_add") to paddle.add.

        """
        candidates_names = set()
        candidates_names.add(op_type)

        prefixes = ["elementwise_", "reduce_", "depthwise_", "bitwise_", "fused_"]
        suffixes = ["_v2", "_v1", "_grad"]

        clean_name = op_type
        if clean_name == "reshape2": clean_name = "reshape"
        for p in prefixes: clean_name = clean_name.replace(p, "")
        for s in suffixes: clean_name = clean_name.replace(s, "")

        candidates_names.add(clean_name)

        # 2. define Search Scope
        search_modules = [
            F,  # paddle.nn.functional (conv2d, relu, softmax)
            paddle,  # paddle (add, matmul, reshape)
            paddle.linalg,  # paddle.linalg (norm, cholesky)
            paddle.tensor,  # paddle.tensor
            paddle.optimizer,
            paddle.io
        ]

        possible_funcs = []

        for name in candidates_names:
            for module in search_modules:
                if hasattr(module, name):
                    func = getattr(module, name)
                    if callable(func):
                        possible_funcs.append(func)

        if not possible_funcs:
            # print(f"[Auto-Bind] Failed to find API for {op_type}, using identity.")
            return lambda *args: args[0] if args else None

        def runner(*inputs):
            last_error = None

            for func in possible_funcs:
                try:
                    sig = inspect.signature(func)
                    params = sig.parameters

                    filtered_kwargs = {}
                    for k, v in raw_attrs.items():
                        if k in params:
                            filtered_kwargs[k] = v

                    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                        filtered_kwargs = raw_attrs

                    return func(*inputs, **filtered_kwargs)

                except Exception as e:
                    last_error = e
                    continue

            try:
                return possible_funcs[0](*inputs)
            except:
                pass

            # return Identity
            # raise RuntimeError(f"Auto-binding failed for {op_type}. Last error: {last_error}")
            return inputs[0] if inputs else None

        return runner

    def _registered_leaf_layer_types(self):
        names = [
            'MultiHeadAttention',
            'TransformerEncoderLayer',
            'TransformerDecoderLayer',
        ]
        return tuple(
            cls for cls in (getattr(paddle.nn, name, None) for name in names)
            if cls is not None
        )

    def _transparent_container_types(self):
        names = ['TransformerEncoder', 'TransformerDecoder', 'Transformer']
        return tuple(
            cls for cls in (getattr(paddle.nn, name, None) for name in names)
            if cls is not None
        )

    def _activation_id(self, module):
        activation = getattr(module, 'activation', None)
        name = getattr(activation, '__name__', str(activation)).lower()
        return 1 if 'relu' in name else 2

    def _augment_transformer_attrs(self, attrs, module):
        if module is None:
            return attrs

        if isinstance(module, paddle.nn.MultiHeadAttention):
            attrs.update({
                'embed_dim': int(module.embed_dim),
                'num_heads': int(module.num_heads),
                'kdim': int(module.kdim),
                'vdim': int(module.vdim),
                'dropout': float(module.dropout),
                'need_weights': bool(getattr(module, 'need_weights', False)),
                'bias': module.q_proj.bias is not None,
                'training': bool(module.training),
            })
            return attrs

        if isinstance(module, (paddle.nn.TransformerEncoderLayer, paddle.nn.TransformerDecoderLayer)):
            config = getattr(module, '_config', {}) or {}
            attrs.update({
                'd_model': int(config.get('d_model', module.self_attn.embed_dim)),
                'nhead': int(config.get('nhead', module.self_attn.num_heads)),
                'dim_feedforward': int(config.get('dim_feedforward', module.linear1.bias.shape[0])),
                'dropout': float(config.get('dropout', getattr(module.dropout, 'p', 0.0))),
                'normalize_before': bool(module.normalize_before),
                'bias': module.self_attn.q_proj.bias is not None,
                'layer_norm_eps': float(getattr(module.norm1, '_epsilon', 1e-5)),
                'activation_id': self._activation_id(module),
                'training': bool(module.training),
            })
        return attrs

    def trace_model_to_instructions(self, model: paddle.nn.Layer, dummy_inputs):
        def _post_process_attrs(attrs):
            alias_sync_map = {
                'kernel_size': 'ksize', 'ksize': 'kernel_size',
                'stride': 'strides', 'strides': 'stride',
                'padding': 'paddings', 'paddings': 'padding',
                'dilation': 'dilations', 'dilations': 'dilation',
                'param_attr': 'weight_attr', 'weight_attr': 'param_attr',
                'bias_attr': 'bias_attr', 'divisor': 'divisor_override',
                'shape': 'p_shape'
            }

            for ek, tk in alias_sync_map.items():
                if attrs.get(ek) is not None:
                    attrs[tk] = attrs[ek]
                elif ek in attrs and tk not in attrs:
                    attrs[tk] = attrs[ek]
            for key in ['weight_attr', 'param_attr', 'bias_attr']:
                if attrs.get(key) is None:
                    attrs[key] = True

            if attrs.get('divisor_override') is not None:
                attrs['exclusive'] = False

            kernel_size = attrs.get('kernel_size')
            if isinstance(kernel_size, (int, float)):
                data_format = str(attrs.get('data_format', '')).upper()
                spatial_rank = 2 if data_format in {'NCHW', 'NHWC'} else 1
                kernel_size = [kernel_size] * spatial_rank
                attrs['kernel_size'] = list(kernel_size)
                attrs['ksize'] = list(kernel_size)

            if 'padding' in attrs:
                p = attrs['padding']
                ks = attrs.get('kernel_size', [1])
                if isinstance(ks, (int, float)): ks = [ks]

                if isinstance(p, str):
                    p_str = p.upper()
                    if p_str == 'VALID':
                        p = [0] * len(ks)
                    elif p_str == 'SAME':
                        dil = attrs.get('dilation', [1] * len(ks))
                        p = [((k_v - 1) * d_v) // 2 for k_v, d_v in zip(ks, dil)]
                elif isinstance(p, (int, float)):
                    p = [p] * len(ks)
                attrs['padding'] = p

            if 'stride' in attrs and isinstance(attrs['stride'], (int, float)):
                rank = len(attrs.get('kernel_size', [attrs['stride']]))
                attrs['stride'] = [attrs['stride']] * rank
                attrs['strides'] = list(attrs['stride'])

            if 'data_format' in attrs:
                fmt = attrs['data_format']
                if isinstance(fmt, str):
                    if fmt == 'NHWC' or fmt == 'NWC':
                        attrs['data_format'] = 'channels_last'
                    elif fmt == 'NCHW' or fmt == 'NCW':
                        attrs['data_format'] = 'channels_first'

            return attrs

        self.model = model

        self.static_param_map = {}
        for name, tensor in model.state_dict().items():
            self.static_param_map[name] = tensor

        tracer = PaddleFX(atomic_layer_types=self._registered_leaf_layer_types())
        raw_instructions = tracer.trace(model, dummy_inputs)
        self.static_param_map.update(tracer.constant_values)
        self.trace_value_map = dict(tracer.output_values)

        self.instructions = []
        self.graph_input_names = []
        self.unresolved_nodes = []

        inputs = dummy_inputs if isinstance(dummy_inputs, (list, tuple)) else [dummy_inputs]
        graph_input_values = {
            f"input_{i}": value for i, value in enumerate(inputs)
        }
        for i, _ in enumerate(inputs):
            self.graph_input_names.append(f"input_{i}")

        for item in raw_instructions:
            item['attrs'] = _post_process_attrs(item['attrs'])
            if item.get('module') is not None:
                item['attrs'] = self._augment_transformer_attrs(item['attrs'], item['module'])

            if not item['outputs']:
                self.unresolved_nodes.append({
                    'op_type': item['op_type'],
                    'module_path': item.get('module_path', ''),
                    'reason': 'trace call produced no tensor output',
                })
                continue

            normalized_op_type = item['op_type']
            if str(normalized_op_type).lower() in {'batchnorm', 'batchnorm2d'}:
                input_value = None
                if item.get('inputs'):
                    input_name = item['inputs'][0]
                    input_value = self.trace_value_map.get(input_name)
                    if input_value is None:
                        input_value = graph_input_values.get(input_name)
                input_rank = len(input_value.shape) if hasattr(input_value, 'shape') else None
                if input_rank == 5:
                    item['attrs']['artifact_original_op_type'] = item['op_type']
                    item['attrs']['semantic_rank_normalization'] = (
                        'BatchNorm2D_class_to_BatchNorm3D_semantics'
                    )
                    normalized_op_type = 'BatchNorm3D'

            if str(item['op_type']).lower() == 'concat':
                axis = int(item['attrs'].get('axis', 0))
                steps = binary_associative_chain(
                    item['inputs'],
                    item['outputs'][0],
                    f"{item['outputs'][0]}_semtrans_concat",
                )
                for step_inputs, step_output in steps:
                    if len(step_inputs) == 1:
                        op_type = 'Identity'
                        attrs = {}
                        runner = lambda value: value
                    else:
                        op_type = 'Concat'
                        attrs = {
                            'axis': axis,
                            'semantic_associative_binary': True,
                        }

                        def runner(left, right, _axis=axis):
                            return paddle.concat((left, right), axis=_axis)

                    paddle_op = PaddleOpFactory.get_op_instance(
                        op_type=op_type,
                        attrs=attrs,
                        input_names=list(step_inputs),
                        output_names=[step_output],
                        module=None,
                    )
                    self.instructions.append(Instruction(
                        InstExpr(op=paddle_op, args=tuple(step_inputs)),
                        (step_output,),
                        runner=runner,
                    ))
                continue

            paddle_op = PaddleOpFactory.get_op_instance(
                op_type=normalized_op_type,
                attrs=item['attrs'],
                input_names=item['inputs'],
                output_names=item['outputs'],
                module=item['module']
            )

            if item['module'] is not None:
                def runner(*args, _mod=item['module']):
                    return _mod(*args)
            else:
                runner = self._resolve_paddle_runner(item['op_type'], item['attrs'])

            iexpr = InstExpr(op=paddle_op, args=tuple(item['inputs']))
            self.instructions.append(Instruction(iexpr, tuple(item['outputs']), runner=runner))

    def forward_grad(self, *args):
        self.differentiable = True
        if len(args) != len(self.graph_input_names):
            raise ValueError(
                f"Input count mismatch! Model graph expects {len(self.graph_input_names)} inputs "
                f"({self.graph_input_names}), but received {len(args)}."
            )

        tensor_map: Dict[str, paddle.Tensor] = self.make_param_map()
        for i, internal_name in enumerate(self.graph_input_names):
            tensor_map[internal_name] = args[i]
        debug_numeric(tensor_map)

        self.invalid_found_last = False
        self.interm_grad = []

        if PADDLE_LOG.isEnabledFor(logging.DEBUG):
            for k, v in tensor_map.items():
                if not v.stop_gradient:
                    self.interm_grad.append((k, v))

            for i, p in enumerate(self.parameters()):
                if not p.stop_gradient:
                    self.interm_grad.append((f"p_{i}", p))

        for stmt_idx, instruction in enumerate(self.instructions):
            inps = instruction.iexpr.args
            inst = instruction.runner
            op = instruction.iexpr.op
            outs = instruction.retvals()
            try:
                input_tensors = [tensor_map[idx] for idx in inps]
            except KeyError as e:
                print("Sanitized text removed.")
                print("Sanitized text removed.")
                print("Sanitized text removed.")
                print("Sanitized text removed.")

                missing_key = str(e).strip("'")
                candidates = [k for k in tensor_map.keys() if missing_key.split('.')[-1] in k or k in missing_key]
                print("Sanitized text removed.")

                raise e

            if op.input_like[0] is None:
                op.bind_concrete_types(input_tensors, is_input=True)
            check_type(op, input_tensors, is_input=True, msg="input")

            output_tensors = inst(*input_tensors)
            if not isinstance(output_tensors, (list, tuple)):
                output_tensors = [output_tensors]

            if op.output_like[0] is None:
                op.bind_concrete_types(output_tensors, is_input=False)
            check_type(op, output_tensors, is_input=False, msg="output")

            if len(output_tensors) < len(outs):
                ignored_keys = outs[len(output_tensors):]

                if PADDLE_LOG.isEnabledFor(logging.DEBUG):
                    PADDLE_LOG.Warning(
                        f"[Output Mismatch] Op: {op.op_type} | "
                        f"Static expects {len(outs)} outs, Runner returned {len(output_tensors)}. "
                        f"Ignored unmatched static vars: {ignored_keys}"
                    )

            for out_key, tensor_val in zip(outs, output_tensors):
                tensor_map[out_key] = tensor_val
                if isinstance(tensor_val, paddle.Tensor):
                    self.differentiable &= (not tensor_val.stop_gradient)

            matched_count = min(len(outs), len(output_tensors))
            if op.output_like[0] is None:
                op.bind_concrete_types(output_tensors[:matched_count], is_input=False)

            debug_io(stmt_idx, input_tensors, output_tensors)

            if PADDLE_LOG.isEnabledFor(logging.DEBUG):
                if not output_tensors[0].stop_gradient:
                    for i in range(len(output_tensors)):
                        output_tensors[i].retain_grads()
                        self.interm_grad.append((f"{op}{i}", output_tensors[i]))

            if self.check_intermediate_numeric or (
                    self.use_gradient and not self.stop_updating_loss
            ):
                loss_generator = OpLossManager.dispatch(op)

                if loss_generator is not None:
                    loss_func = loss_generator(op)
                    loss = loss_func(*input_tensors)
                    if not isinstance(loss, tuple):
                        loss = ("", loss)
                    vul_op_loss = loss
                else:
                    vul_op_loss = None

                self.invalid_found_last |= not numeric_valid(output_tensors)

                is_valid = numeric_valid(output_tensors)
                self.invalid_found_last |= not is_valid

                if not is_valid:
                    print(f"\n[DEBUG] Exploded at stmt_idx={stmt_idx}, Op: {op}")
                    for idx, t in enumerate(output_tensors):
                        if isinstance(t, paddle.Tensor) and t.numel() > 0:
                            try:
                                max_val = float(paddle.max(paddle.abs(t)))
                                print(f"  -> Output[{idx}] Max Value: {max_val:.4e}")
                                print(f"  -> Stop Gradient: {t.stop_gradient}")
                            except Exception:
                                print(f"  -> Output[{idx}] (Cannot retrieve stats)")

                if self.invalid_found_last and (
                        self.use_gradient and not self.stop_updating_loss
                ):
                    if PADDLE_LOG.isEnabledFor(logging.DEBUG):
                        for inp_i, inp in enumerate(input_tensors):
                            PADDLE_LOG.info(
                                f"[inp]@{inp_i} :: {float(inp.min()):.5f} ~ {float(inp.max()):.5f}"
                            )

                    # [Added] Magnitude Overflow Check (Matches PyTorch logic)
                    # Prioritize penalizing large values before falling back to generic loss
                    if vul_op_loss is None:
                        NUMERIC_THRESHOLD = 1e6
                        for out_t in output_tensors:
                            if (isinstance(out_t, paddle.Tensor) and
                                    paddle.is_floating_point(out_t) and
                                    not out_t.stop_gradient and
                                    out_t.numel() > 0):

                                # Detach to check value without affecting graph
                                max_val = paddle.max(paddle.abs(out_t)).item()

                                if max_val > NUMERIC_THRESHOLD:
                                    # L1 Penalty for overflow
                                    loss_val = paddle.sum(
                                        paddle.nn.functional.relu(paddle.abs(out_t) - NUMERIC_THRESHOLD))

                                    if loss_val > 0:
                                        vul_op_loss = ("magnitude_overflow", loss_val)
                                        msg = f"Op {op} overflow {max_val:.2e} > 1e6. Enforcing L1 penalty."
                                        if 'PADDLE_LOG' in globals():
                                            PADDLE_LOG.debug(msg)
                                        else:
                                            print(f"[Grad Fix] {msg}")

                    if vul_op_loss is None:
                        if len(input_tensors) > 0:
                            fallback_loss = paddle.to_tensor(0.0, dtype=input_tensors[0].dtype,
                                                             place=input_tensors[0].place)
                        else:
                            fallback_loss = paddle.to_tensor(0.0)

                        for inp in input_tensors:
                            if isinstance(inp, paddle.Tensor) and paddle.isfinite(inp).all():
                                # square + mean
                                fallback_loss = fallback_loss + paddle.mean(paddle.square(inp)) * 0.01

                        for out in output_tensors:
                            if isinstance(out, paddle.Tensor) and paddle.isfinite(out).all():
                                fallback_loss = fallback_loss + paddle.mean(paddle.log1p(paddle.abs(out)))

                        if fallback_loss == 0.0:
                            fallback_loss = paddle.to_tensor(1.0, dtype=fallback_loss.dtype, place=fallback_loss.place)
                            fallback_loss.stop_gradient = False

                        vul_op_loss = ("fallback_robust", fallback_loss)

                        if PADDLE_LOG.isEnabledFor(logging.DEBUG):
                            PADDLE_LOG.debug(f"Op {op} produced NaN/Inf. Using Robust Log-L1 Fallback Loss (Paddle).")

                    ConstraintCheck.true(
                        vul_op_loss is not None,
                        f"{op = } has no `torch_loss` but produces NaN or INF!",
                    )
                    # Given its low chance of happening, ignore it for now.
                    loss_suf, l = vul_op_loss

                    masked_loss = paddle.sum(paddle.where(l > 0, l, paddle.zeros_like(l)))

                    l_min = l.min().item() if l.numel() > 0 else 0.0
                    l_max = l.max().item() if l.numel() > 0 else 0.0

                    msg = f"loss_{loss_suf}: {l_min:.3f} ~ {l_max:.3f} ~ {masked_loss.item()}"
                    PADDLE_LOG.debug(f"#{self.n_iter} [NaN/Inf] in output: {op} ~ {msg}")

                    ConstraintCheck.true(
                        paddle.all(l > 0),
                        f"`{op}` outputs NaN/INF found non-positive loss!",
                    )
                    loss_name = f"{op}_{loss_suf}"
                    ConstraintCheck.true(self.loss is None, "Multiple loss detected!")

                    self.loss = loss_name, masked_loss
                    if loss_name != self.cur_loss_name:
                        self.reset_optimizer()
                        self.cur_loss_name = loss_name

                    self.stop_updating_loss = True
                    return output_tensors

        self.first_run = False
        return tuple(tensor_map[key] for key in self.output_map) if hasattr(self, 'output_map') else tuple()
