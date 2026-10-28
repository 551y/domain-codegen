# Adapted from NNSmith and modified for SemTrans; see THIRD_PARTY_NOTICES.md.

import logging, os, time, math
import warnings
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.fx
from torch.fx import Tracer, GraphModule
from torch.fx.operator_schemas import normalize_function
import torch.nn as nn

from ..registry import register_symbolnet, detect_input_ops
from SemTrans.testing_and_repair.logging import TORCH_LOG
from SemTrans.math_symbol_base.frontends.abstract.dtype import DType, dtype_size
from SemTrans.math_symbol_base.frontends.abstract.op import AbsOpBase
from SemTrans.math_symbol_base.frontends.abstract.gir import (
    InstExpr,
    Instruction,
    binary_associative_chain,
)
from SemTrans.math_symbol_base.frontends.abstract.tensor import AbsTensor
from SemTrans.math_symbol_base.frontends.abstract.error import ConstraintCheck, ConstraintError
from SemTrans.testing_and_repair.util import SourceLocator

from .torchop import TorchOpFactory, TorchAdapter
from .numeric import OpLossManager, numeric_valid


__INPUT_FOUND_NAN_MSG__ = "[NaN] in model inputs!"
__INPUT_FOUND_INF_MSG__ = "[Inf] in model inputs!"


class _TorchLeafTracer(Tracer):
    def __init__(self, leaf_module_types, transparent_container_types=()):
        super().__init__()
        self.leaf_module_types = leaf_module_types
        self.transparent_container_types = transparent_container_types

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, self.transparent_container_types):
            return True
        if isinstance(m, self.leaf_module_types):
            return True
        return super().is_leaf_module(m, module_qualified_name)


@dataclass
class _ModuleInstructionSpec:
    module: nn.Module
    module_path: str
    input_vars: List[str]
    output_vars: List[str]


class _TorchModuleDecompositionContext:
    def __init__(self, registry):
        self.registry = registry
        self.specs = []

    def fresh_var(self, module_path, suffix):
        return f"{module_path.replace('.', '_')}_{suffix}"

    def emit_module(self, module, module_path, input_vars, output_vars):
        self.specs.append(_ModuleInstructionSpec(
            module=module,
            module_path=module_path,
            input_vars=list(input_vars),
            output_vars=list(output_vars),
        ))

    def decompose(self, module, module_path, input_vars, output_vars):
        return self.registry._expand_into(self, module, module_path, input_vars, output_vars)


class _TorchModuleDecomposerRegistry:
    def __init__(self):
        self._handlers = []

    def register(self, *module_types):
        def wrapper(func):
            self._handlers.append((module_types, func))
            return func

        return wrapper

    def module_types(self):
        types = []
        for registered_types, _ in self._handlers:
            types.extend(registered_types)
        return tuple(types)

    def expand(self, module, module_path, input_vars, output_vars):
        ctx = _TorchModuleDecompositionContext(self)
        if not ctx.decompose(module, module_path, input_vars, output_vars):
            return None
        return ctx.specs

    def _expand_into(self, ctx, module, module_path, input_vars, output_vars):
        for module_types, handler in self._handlers:
            if isinstance(module, module_types):
                return handler(ctx, module, module_path, list(input_vars), list(output_vars))
        return False


_TORCH_MODULE_DECOMPOSERS = _TorchModuleDecomposerRegistry()


@_TORCH_MODULE_DECOMPOSERS.register(nn.TransformerEncoder)
def _decompose_transformer_encoder(ctx, module, module_path, input_vars, output_vars):
    if not input_vars:
        return False

    current_vars = list(input_vars[:1])
    layer_count = len(module.layers)

    for idx, layer in enumerate(module.layers):
        layer_path = f"{module_path}.layers.{idx}"
        is_last_layer = idx == layer_count - 1 and module.norm is None
        layer_output = output_vars[0] if is_last_layer else ctx.fresh_var(module_path, f"layers_{idx}")
        ctx.emit_module(layer, layer_path, current_vars, [layer_output])
        current_vars = [layer_output]

    if module.norm is not None:
        ctx.emit_module(module.norm, f"{module_path}.norm", current_vars, output_vars)
        return True

    return layer_count > 0


@_TORCH_MODULE_DECOMPOSERS.register(nn.TransformerDecoder)
def _decompose_transformer_decoder(ctx, module, module_path, input_vars, output_vars):
    if len(input_vars) < 2:
        return False

    memory_var = input_vars[1]
    current_tgt = input_vars[0]
    layer_count = len(module.layers)

    for idx, layer in enumerate(module.layers):
        layer_path = f"{module_path}.layers.{idx}"
        is_last_layer = idx == layer_count - 1 and module.norm is None
        layer_output = output_vars[0] if is_last_layer else ctx.fresh_var(module_path, f"layers_{idx}")
        ctx.emit_module(layer, layer_path, [current_tgt, memory_var], [layer_output])
        current_tgt = layer_output

    if module.norm is not None:
        ctx.emit_module(module.norm, f"{module_path}.norm", [current_tgt], output_vars)
        return True

    return layer_count > 0


@_TORCH_MODULE_DECOMPOSERS.register(nn.Transformer)
def _decompose_transformer(ctx, module, module_path, input_vars, output_vars):
    if len(input_vars) < 2:
        return False

    src_var, tgt_var = input_vars[0], input_vars[1]
    memory_var = ctx.fresh_var(module_path, "encoder")

    if not ctx.decompose(module.encoder, f"{module_path}.encoder", [src_var], [memory_var]):
        return False

    return ctx.decompose(module.decoder, f"{module_path}.decoder", [tgt_var, memory_var], output_vars)


# Debug decorator: Skip function execution in tracing mode
def debug_numeric(tensor_map):
    """Check for NaN or Inf values in the tensor mapping."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # PyTorch check: isinf/isnan
        ConstraintCheck.true(
            not any([torch.isinf(tensor).any().item() for tensor in tensor_map.values()]),
            __INPUT_FOUND_INF_MSG__,
        )
        ConstraintCheck.true(
            not any([torch.isnan(tensor).any().item() for tensor in tensor_map.values()]),
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
            # DType.from_torch need to be implemented/available
            ttype = AbsTensor(list(tensor.shape), DType.from_torch(tensor.dtype))
            assert (
                    like[i] == ttype
            ), f"{msg} {ioro} abstract type != concrete {ttype} for {op} {op.input_like} -> {op.output_like}"


def debug_io(stmt_idx, itensors, otensors):
    if TORCH_LOG.isEnabledFor(logging.DEBUG):
        TORCH_LOG.debug(f">> statment {stmt_idx}")
        for inp_i, i in enumerate(itensors):
            TORCH_LOG.debug(f"  (shape={i.shape} dtype={i.dtype})")
            # .data in PyTorch is similar, but .detach() is preferred for logging values
            TORCH_LOG.debug(f"[inp]@{inp_i} :: {i.min().item():.5f} ~ {i.max().item():.5f}")
        for out_i, o in enumerate(otensors):
            TORCH_LOG.debug(f"  (shape={o.shape} dtype={o.dtype})")
            TORCH_LOG.debug(f"[out]@{out_i} :: {o.min().item():.5f} ~ {o.max().item():.5f}")


@register_symbolnet("pytorch")
class SymbolNet(nn.Module):
    def __init__(self, model: nn.Module, framework_type, use_gradient=False, force_input_map=None):
        super(SymbolNet, self).__init__()
        self.check_intermediate_numeric = None
        self.differentiable = None
        self.optimizer = None
        self.loss = None
        self.to_train = None
        self.use_gradient = None
        self.instructions = []
        self.model = model
        self.framework_type = framework_type
        self.device = None
        self.proxy_enabled = False
        if force_input_map is not None:
            self.input_map = force_input_map
        else:
            self.input_map = detect_input_ops(self.model)
        self.gradient = use_gradient
        if use_gradient:
            self.enable_training()
        self._parameters_dict = dict(self.model.named_parameters())  # Cache for param mapping
        self.adapter = TorchAdapter
        self.locator = SourceLocator(self.model)

    def get_layer_location(self, param_name):
        return self.locator.get_location(param_name)

    def to(self, *args, **kwargs):
        device = None

        if 'device' in kwargs:
            device = kwargs['device']

        elif len(args) > 0:
            arg0 = args[0]
            if isinstance(arg0, (torch.device, str, int)):
                device = arg0
            elif isinstance(arg0, torch.Tensor):
                device = arg0.device

        if isinstance(device, str):
            device = device.lower().replace('gpu', 'cuda')

        if device is not None:
            self._device = device

        new_args = list(args)
        if 'device' in kwargs:
            kwargs['device'] = device
        elif len(args) > 0 and isinstance(args[0], str):
            new_args[0] = device

        return super().to(*new_args, **kwargs)

    def to_tensor(self, data):
        if isinstance(data, (list, tuple)):
            return type(data)(self.to_tensor(x) for x in data)

        
        if isinstance(data, dict):
            return {k: self.to_tensor(v) for k, v in data.items()}

        
        if isinstance(data, torch.Tensor):
            tensor = data
        else:
            # torch.as_tensor preserves data if already a tensor, otherwise converts
            tensor = torch.as_tensor(data)

        if self.device is not None:
            tensor = tensor.to(self.device)

        return tensor

    def to_numpy(self, output):
        results = {}

        def to_np(x):
            if isinstance(x, torch.Tensor):
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

    def get_op_key(self, inst):
        return inst.iexpr.op.op_type

    def get_runner(self, inst):
        return inst.runner

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = value

    @property
    def input_like(self):
        return self.input_map

    def set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def stop_grad(self):
        return torch.no_grad()

    def empty_cache(self):
        torch.cuda.empty_cache()

    def get_weights(self) -> Dict[str, np.ndarray]:
        weights = {}
        for name, param in self.model.state_dict().items():
            weights[name] = param.detach().cpu().numpy()

        return weights

    def inject_value(self, target_tensor, source_numpy_data) -> bool:
        if target_tensor is None or source_numpy_data is None:
            return False

        try:
            if isinstance(source_numpy_data, torch.Tensor):
                val = source_numpy_data
            else:
                val = torch.tensor(source_numpy_data)

            val = val.to(
                device=target_tensor.device,
                dtype=target_tensor.dtype,
                non_blocking=False
            )

            with torch.no_grad():
                target_tensor.copy_(val)
            return True

        except Exception as e:
            print(f"[SymNet Error] PyTorch inject failed: {e}")
            return False

    def get_layer_info(self) -> Dict[str, Dict[str, str]]:
        layer_info = {}
        for name, module in self.model.named_modules():
            layer_type = module.__class__.__name__
            for param_name, _ in module.named_parameters(recurse=False):
                full_param_name = f"{name}.{param_name}" if name else param_name
                layer_info[full_param_name] = {
                    'type': layer_type,
                    'parent': name
                }
        return layer_info

    def calculate_max_sample(self, MAX_MEM, MEM_FACTOR=0.4):
        nbytes = 0

        for iname, ii in self.input_like.items():
            nbytes += (len(ii.shape)
                       * dtype_size(ii.dtype)
                       )
        for name, param in self.named_parameters():
            nbytes += param.numel() * dtype_size(param.dtype)

        max_sample = int(MAX_MEM * MEM_FACTOR / nbytes)

        return max_sample, nbytes * max_sample / (1024 ** 3)

    def random_tensor(self, shape, dtype, margin=2, base=0, use_gpu=False):
        """
        [PyTorch] Optimized random tensor generation.
        """

        # 1. Unified type handling
        if isinstance(dtype, DType):
            dt_enum = dtype
        elif isinstance(dtype, str):
            dt_enum = DType.from_str(dtype)
        elif isinstance(dtype, torch.dtype):
            dt_enum = DType.from_torch(dtype)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        torch_dtype = dt_enum.torch()

        # 2. Device handling (Prioritize using the saved device in the class to reduce overhead)
        # If self.device is not set, fallback based on use_gpu
        if self.device:
            target_device = self.device
        else:
            target_device = torch.device("cuda") if use_gpu and torch.cuda.is_available() else torch.device("cpu")

        # 3. Shape processing
        concrete_shape = [1 if (s == -1 or s == 'int') else int(s) for s in shape]

        # 4. Generation (Generate directly on the target device to avoid CPU->GPU copy overhead)
        if base == "center":
            base = -margin / 2

        # Generate base float data
        # Note: torch.rand generates [0, 1)
        rand_data = torch.rand(concrete_shape, dtype=torch.float32, device=target_device)
        fp_tensor = base + (rand_data - 0.5) * margin

        # 5. Type conversion
        if dt_enum.is_float():
            return fp_tensor.to(torch_dtype)
        elif dt_enum == DType.bool:
            return (rand_data > 0.5).to(torch_dtype)
        else:
            return torch.round(fp_tensor).to(torch_dtype)

    def get_weight_tensor(self, shape, dtype, use_gpu=False):
        """
        Generate weight tensor using Variance Scaling (Kaiming Init equivalent).
        Designed for numerical stability in deep networks.
        """
        if isinstance(dtype, DType):
            dt_enum = dtype
        elif isinstance(dtype, str):
            dt_enum = DType.from_str(dtype)
        elif isinstance(dtype, torch.dtype):
            dt_enum = DType.from_torch(dtype)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        torch_dtype = dt_enum.torch()

        if self.device:
            target_device = self.device
        else:
            target_device = torch.device("cuda") if use_gpu and torch.cuda.is_available() else torch.device("cpu")

        concrete_shape = [1 if (s == -1 or s == 'int') else int(s) for s in shape]

        tensor = torch.randn(concrete_shape, dtype=torch.float32, device=target_device)

        ndim = len(concrete_shape)
        if ndim >= 2:
            fan_in = concrete_shape[1]
            for s in concrete_shape[2:]:
                fan_in *= s
            fan_in = max(fan_in, 1)

            std = 1.0 / math.sqrt(fan_in)
            tensor.mul_(std)

        elif ndim == 1:
            tensor.uniform_(-1e-6, 1e-6)

        if dt_enum.is_float():
            return tensor.to(torch_dtype)
        else:
            return torch.round(tensor).to(torch_dtype)

    def clear_cache(self):
        import gc
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def stop_training(self):
        self.use_gradient = False
        self.loss = None

    def make_param_map(self) -> Dict[str, torch.Tensor]:
        tensor_map: Dict[str, torch.Tensor] = {}

        # In PyTorch, named_parameters() is the standard way
        for k, v in self.model.named_parameters():
            tensor_map[k] = v

        # Also include buffers (like BN running_mean) if needed, though they aren't trained usually
        for k, v in self.model.named_buffers():
            tensor_map[k] = v

        return tensor_map

    def enable_training(self, extra_trainable: Dict[str, torch.Tensor]= {}):
        self.use_gradient = True
        self.to_train = []

        for t in extra_trainable.values():
            self.to_train.append(t)

        for t in self.parameters():
            self.to_train.append(t)

        # PyTorch Adam syntax: params is the first arg
        self.optimizer = torch.optim.Adam(self.to_train, lr=5e-1)

        self.training_reset()

    def training_reset(self):
        if self.optimizer:
            self.optimizer.zero_grad()
        self.loss = None
        self.stop_updating_loss = False

    def reset_optimizer(self):
        self.optimizer = torch.optim.Adam(self.to_train, lr=5e-2)

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
            init_tensors: Dict[str, torch.Tensor],
            use_cuda=False,
            max_time=None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if max_time is None:
            max_time = float(os.getenv("NNSMITH_GRAD_TIME", 0.5))

        dummy_inputs = list(init_tensors.values())

        # 2. Call the parsing function (JIT Trace for PyTorch)
        self.trace_model_to_instructions(self.model, dummy_inputs)

        inputs: Dict[str, torch.Tensor] = {}
        for key, tensor in init_tensors.items():
            if tensor.dtype in [torch.float16, torch.float32, torch.float64]:
                t = tensor.clone().detach()  # Detach first
                t.requires_grad = True  # Enable gradient
                inputs[key] = t
            else:
                inputs[key] = tensor.clone().detach()

        self.enable_training(extra_trainable=inputs)

        last_check_intermediate_numeric = self.check_intermediate_numeric
        self.check_intermediate_numeric = True

        if use_cuda:
            for k in inputs:
                # Need to recreate variable on GPU and retain grad requirement
                if inputs[k].requires_grad:
                    inputs[k] = inputs[k].cuda().detach().requires_grad_(True)
                else:
                    inputs[k] = inputs[k].cuda()

        sat_inputs = None
        st = time.time()
        self.n_iter = 0
        self.cur_loss_name = None

        while time.time() - st < max_time:
            self.training_reset()
            self.n_iter += 1

            with torch.no_grad():
                for inp in inputs.values():
                    if isinstance(inp, torch.Tensor) and inp.is_floating_point():
                        inp.clamp_(-10.0, 10.0)

            try:
                _ = self.forward_grad(*inputs.values())
                if self.invalid_found_last:
                    self.backward()
                else:
                    # Detach to stop tracking
                    sat_inputs = {k: v.detach() for k, v in inputs.items()}
                    break
            except ConstraintError as e:
                if __INPUT_FOUND_INF_MSG__ in str(e) or __INPUT_FOUND_NAN_MSG__ in str(e):
                    with torch.no_grad():
                        for inp in inputs.values():
                            is_nan_inf = torch.logical_or(torch.isnan(inp), torch.isinf(inp))

                            noise = self.random_tensor(
                                shape=inp.shape, dtype=inp.dtype
                            )
                            if inp.is_cuda:
                                noise = noise.cuda()

                            new_data = torch.where(is_nan_inf, noise, inp)

                            # In-place update in PyTorch
                            inp.data.copy_(new_data)
                    continue
                if 'TORCH_LOG' in globals():
                    TORCH_LOG.debug(e)
                else:
                    print(f"[DEBUG] {e}")
                break

        self.stop_training()
        if sat_inputs is None:
            msg = "[grad] no valid range found!!!"
            if 'TORCH_LOG' in globals():
                TORCH_LOG.debug(msg)
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
            if self.to_train:
                torch.nn.utils.clip_grad_norm_(self.to_train, max_norm=1.0)
            self.optimizer.step()

    def _make_function_runner(self, func, args_schema, kwargs_schema):
        def runner(*runtime_inputs):
            real_args = []
            tensor_idx = 0

            for arg in args_schema:
                if isinstance(arg, torch.fx.Node):
                    real_args.append(runtime_inputs[tensor_idx])
                    tensor_idx += 1
                elif isinstance(arg, (list, tuple)):
                    container = []
                    for item in arg:
                        if isinstance(item, torch.fx.Node):
                            container.append(runtime_inputs[tensor_idx])
                            tensor_idx += 1
                        else:
                            container.append(item)
                    real_args.append(type(arg)(container))
                else:
                    real_args.append(arg)

            return func(*real_args, **kwargs_schema)

        return runner

    def _make_method_runner(self, method_name, args_schema, kwargs_schema):
        def runner(*runtime_inputs):
            real_args = []
            tensor_idx = 0
            for arg in args_schema:
                if isinstance(arg, torch.fx.Node):
                    real_args.append(runtime_inputs[tensor_idx])
                    tensor_idx += 1
                else:
                    real_args.append(arg)

            obj = real_args[0]
            method = getattr(obj, method_name)
            return method(*real_args[1:], **kwargs_schema)

        return runner

    def _recursive_getattr(self, obj, target):
        attrs = target.split(".")
        curr = obj
        for attr in attrs:
            curr = getattr(curr, attr)
        return curr

    def _extract_torch_attrs(self, module):
        attrs = {}

        for k, v in module.__dict__.items():
            if k.startswith('_'): continue

            if isinstance(v, (torch.Tensor, torch.nn.Module, torch.nn.Parameter)):
                continue

            if isinstance(v, (int, float, bool, str, tuple, list)):
                attrs[k] = v

        if hasattr(module, 'padding_mode') and 'padding_mode' not in attrs:
            attrs['padding_mode'] = module.padding_mode

        if 'data_format' in attrs:
            val = attrs['data_format']
            if val == 'channels_last' or val == torch.channels_last:
                attrs['data_format'] = 'channels_last'

        return attrs

    def _activation_id(self, module):
        act_id = getattr(module, 'activation_relu_or_gelu', None)
        if act_id in (1, 2):
            return int(act_id)

        act = getattr(module, 'activation', None)
        act_name = getattr(act, '__name__', str(act)).lower()
        if 'gelu' in act_name:
            return 2
        return 1

    def _extract_transformer_layer_attrs(self, module):
        attrs = self._extract_torch_attrs(module)
        attrs.update({
            'd_model': int(module.self_attn.embed_dim),
            'nhead': int(module.self_attn.num_heads),
            'dim_feedforward': int(module.linear1.out_features),
            'dropout': float(module.dropout.p),
            'batch_first': bool(module.self_attn.batch_first),
            'norm_first': bool(module.norm_first),
            'bias': module.self_attn.in_proj_bias is not None,
            'layer_norm_eps': float(module.norm1.eps),
            'activation_id': self._activation_id(module),
            'training': bool(module.training),
        })
        return attrs

    def _extract_multihead_attention_attrs(self, module):
        attrs = self._extract_torch_attrs(module)
        attrs.update({
            'bias': module.in_proj_bias is not None,
            'add_bias_kv': module.bias_k is not None and module.bias_v is not None,
        })
        return attrs

    def _registered_leaf_module_types(self):
        return (
            nn.MultiheadAttention,
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
        )

    def _transparent_container_types(self):
        return _TORCH_MODULE_DECOMPOSERS.module_types()

    def _extract_module_attrs(self, module):
        if isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
            return self._extract_transformer_layer_attrs(module)
        if isinstance(module, nn.MultiheadAttention):
            return self._extract_multihead_attention_attrs(module)
        return self._extract_torch_attrs(module)

    def _append_torch_instruction(self, op_type, attrs, input_vars, output_vars, runner, real_module):
        torch_op = TorchOpFactory.get_op_instance(
            op_type=op_type,
            attrs=attrs,
            input_names=input_vars,
            output_names=output_vars,
            module=real_module
        )
        iexpr = InstExpr(op=torch_op, args=tuple(input_vars))
        instruction = Instruction(iexpr, tuple(output_vars), runner=runner)
        self.instructions.append(instruction)
        return instruction

    @staticmethod
    def _is_torch_concat_target(target):
        candidates = {
            candidate
            for candidate in (
                getattr(torch, "cat", None),
                getattr(torch, "concat", None),
                getattr(torch, "concatenate", None),
            )
            if callable(candidate)
        }
        return target in candidates

    def _append_torch_concat_chain(self, input_vars, output_var, axis):
        steps = binary_associative_chain(
            input_vars,
            output_var,
            f"{output_var}_semtrans_concat",
        )
        for step_inputs, step_output in steps:
            if len(step_inputs) == 1:
                self._append_torch_instruction(
                    op_type="Identity",
                    attrs={},
                    input_vars=list(step_inputs),
                    output_vars=[step_output],
                    runner=lambda value: value,
                    real_module=None,
                )
                continue

            def concat_runner(left, right, _axis=int(axis)):
                return torch.cat((left, right), dim=_axis)

            self._append_torch_instruction(
                op_type="cat",
                attrs={"dim": int(axis), "semantic_associative_binary": True},
                input_vars=list(step_inputs),
                output_vars=[step_output],
                runner=concat_runner,
                real_module=None,
            )

    def _normalize_function_literal_attrs(self, target, args, kwargs, trace_target=None):
        """Recover schema names and graph-bound constants for FX functions.

        FX keeps positional literals as ``arg_N`` when tracing a call
        function.  MathIR bindings, however, are expressed using the
        framework API names (for example ``start_dim`` and ``end_dim`` for
        ``torch.flatten``). PyTorch's schema normalizer knows overload
        signatures without requiring an operator-specific table. Parameters
        lifted into ``get_attr`` nodes are also constants in the exported
        graph, so materialize those values under their schema names (for
        example ``weight`` and ``bias``). Runtime data dependencies remain in
        the instruction input list and are not copied into attributes.
        """
        try:
            arg_types = tuple(
                torch.Tensor if isinstance(value, torch.fx.Node) else type(value)
                for value in args
            )
            normalized = normalize_function(
                target,
                args,
                kwargs,
                arg_types=arg_types,
                normalize_to_only_use_kwargs=True,
            )
            if normalized is None:
                return {}
            values = dict(normalized.kwargs)

            unresolved = object()

            def materialize_constant(value):
                if isinstance(value, torch.fx.Node):
                    if value.op != "get_attr" or trace_target is None:
                        return unresolved
                    try:
                        return self._recursive_getattr(trace_target, value.target)
                    except (AttributeError, KeyError):
                        return unresolved
                if isinstance(value, (tuple, list)):
                    items = [materialize_constant(item) for item in value]
                    if any(item is unresolved for item in items):
                        return unresolved
                    return type(value)(items)
                if isinstance(value, dict):
                    items = {
                        key: materialize_constant(item)
                        for key, item in value.items()
                    }
                    if any(item is unresolved for item in items.values()):
                        return unresolved
                    return items
                return value

            attrs = {}
            for key, value in values.items():
                materialized = materialize_constant(value)
                if materialized is not unresolved:
                    attrs[key] = materialized
            return attrs
        except Exception:
            # Some Python/operator targets have no FX schema or have
            # ambiguous overloads.  The caller retains the existing arg_N
            # attributes in that case; no operator-specific guess is needed.
            return {}

    @staticmethod
    def _function_op_type(target):
        """Return the real framework operation name for an FX callable.

        ``torch.export`` uses ``OpOverload`` targets whose Python ``__name__``
        is only the overload label (for example ``default`` or
        ``using_ints``).  The dispatcher schema retains the actual ATen name
        and is the authoritative key for MathIR lookup.
        """
        schema = getattr(target, "_schema", None)
        schema_name = str(getattr(schema, "name", "") or "")
        if schema_name:
            return schema_name.rsplit("::", 1)[-1]
        return target.__name__ if hasattr(target, "__name__") else str(target)

    def _decompose_module_node(self, module, module_path, input_vars, output_vars):
        specs = _TORCH_MODULE_DECOMPOSERS.expand(module, module_path, input_vars, output_vars)
        if not specs:
            return False

        for spec in specs:
            attrs = self._extract_module_attrs(spec.module)
            self._append_torch_instruction(
                op_type=spec.module.__class__.__name__,
                attrs=attrs,
                input_vars=spec.input_vars,
                output_vars=spec.output_vars,
                runner=spec.module,
                real_module=spec.module,
            )
        return True

    def trace_model_to_instructions(self, model: nn.Module, dummy_inputs):
        self.model = model
        self.instructions = []

        # =========================================================
        # 1. Robust Symbolic Tracing
        # =========================================================
        try:
            trace_target = model
            tracer = _TorchLeafTracer(
                self._registered_leaf_module_types(),
                self._transparent_container_types()
            )
            graph = tracer.trace(trace_target)
            gm = GraphModule(trace_target, graph)
        except Exception as e:
            # FX cannot evaluate shape-dependent Python control flow (for
            # example ``if x.shape[-1] < k``).  The export tracer receives the
            # concrete input shapes and can legally specialize those guards;
            # its GraphModule uses the same node contract consumed below.
            try:
                export_api = getattr(torch, "export", None)
                export_fn = getattr(export_api, "export", None)
                if export_fn is None:
                    raise RuntimeError("torch.export.export is unavailable")
                export_result = export_fn(
                    model,
                    tuple(dummy_inputs) if isinstance(dummy_inputs, (list, tuple)) else (dummy_inputs,),
                    strict=False,
                )
                # ``graph_module`` exposes lifted parameters and buffers as
                # placeholders.  ``module()`` restores them as GetAttr nodes,
                # leaving only real model inputs in the public graph contract.
                # SymbolNet can then bind parameters through its normal
                # instruction dataflow instead of misclassifying them as
                # external test inputs.
                gm = export_result.module()
                trace_target = gm
            except Exception as export_error:
                # Fallback only for actual parallel wrappers.
                parallel_wrappers = (nn.DataParallel,)
                ddp = getattr(nn.parallel, 'DistributedDataParallel', None)
                if ddp is not None:
                    parallel_wrappers = parallel_wrappers + (ddp,)
                if isinstance(model, parallel_wrappers):
                    return self.trace_model_to_instructions(model.module, dummy_inputs)
                raise RuntimeError(
                    f"FX Trace failed on {model.__class__.__name__}: {e}; "
                    f"torch.export fallback failed: {export_error}"
                ) from export_error

        self.graph_input_names = []

        # =========================================================
        # 2. Iterate FX Graph Nodes
        # =========================================================
        for node in gm.graph.nodes:

            # --- Case A: Input Placeholder ---
            if node.op == 'placeholder':
                self.graph_input_names.append(node.name)
                continue

            # --- Case B: Output ---
            if node.op == 'output':
                continue

            # --- Case C: Get Attribute ---
            if node.op == 'get_attr':
                val = self._recursive_getattr(trace_target, node.target)

                def make_getattr_runner(v):
                    return lambda *args: v

                runner = make_getattr_runner(val)

                op_type = "GetAttr"
                attrs = {'name': node.target}
                torch_op = TorchOpFactory.get_op_instance(
                    op_type=op_type, attrs=attrs, input_names=[],
                    output_names=[node.name], module=None
                )

                iexpr = InstExpr(op=torch_op, args=())
                self.instructions.append(Instruction(iexpr, tuple([node.name]), runner=runner))
                continue

            # --- Case D: Computation Nodes ---
            input_vars = []

            # Recursively extract dependency names
            def extract_node_names(args_obj):
                if isinstance(args_obj, torch.fx.Node):
                    input_vars.append(args_obj.name)
                elif isinstance(args_obj, (list, tuple)):
                    for item in args_obj:
                        extract_node_names(item)

            extract_node_names(node.args)
            extract_node_names(node.kwargs)

            output_vars = [node.name]
            runner = None
            op_type = "unknown"
            attrs = node.kwargs.copy()

            def contains_fx_node(value):
                if isinstance(value, torch.fx.Node):
                    return True
                if isinstance(value, (list, tuple)):
                    return any(contains_fx_node(item) for item in value)
                if isinstance(value, dict):
                    return any(contains_fx_node(item) for item in value.values())
                return False

            for arg_index, arg_value in enumerate(node.args):
                if not contains_fx_node(arg_value):
                    attrs.setdefault(f"arg_{arg_index}", arg_value)

            real_module = None

            if node.op == 'call_function':
                runner = self._make_function_runner(node.target, node.args, node.kwargs)
                op_type = self._function_op_type(node.target)
                real_module = model
                attrs.update(
                    self._normalize_function_literal_attrs(
                        node.target, node.args, node.kwargs, trace_target
                    )
                )
                if self._is_torch_concat_target(node.target):
                    axis = attrs.get("dim", attrs.get("axis", attrs.get("arg_1", 0)))
                    self._append_torch_concat_chain(input_vars, output_vars[0], axis)
                    continue
            elif node.op == 'call_module':
                sub_mod = gm.get_submodule(node.target)
                runner = sub_mod
                op_type = sub_mod.__class__.__name__
                real_module = sub_mod

                if self._decompose_module_node(sub_mod, node.target, input_vars, output_vars):
                    continue

                # [CRITICAL FIX]: Extract attributes from the module instance
                # This ensures src_op.attrs is populated (e.g., kernel_size, stride)
                module_attrs = self._extract_module_attrs(sub_mod)
                attrs.update(module_attrs)

            elif node.op == 'call_method':
                runner = self._make_method_runner(node.target, node.args, node.kwargs)
                op_type = node.target

            # Construct Instruction
            torch_op = TorchOpFactory.get_op_instance(
                op_type=op_type,
                attrs=attrs,  # Now contains extracted hyperparameters
                input_names=input_vars,
                output_names=output_vars,
                module=real_module
            )

            iexpr = InstExpr(op=torch_op, args=tuple(input_vars))
            self.instructions.append(Instruction(iexpr, tuple(output_vars), runner=runner))

    def forward_grad(self, *args):
        self.differentiable = True
        # Validate inputs

        tensor_map: Dict[str, torch.Tensor] = self.make_param_map()

        # Bind external inputs to internal graph names
        for i, internal_name in enumerate(self.graph_input_names):
            tensor_map[internal_name] = args[i]

        debug_numeric(tensor_map)

        self.invalid_found_last = False
        self.interm_grad = []

        if TORCH_LOG.isEnabledFor(logging.DEBUG):
            for k, v in tensor_map.items():
                if v.requires_grad:
                    self.interm_grad.append((k, v))

            for i, p in enumerate(self.parameters()):
                if p.requires_grad:
                    self.interm_grad.append((f"p_{i}", p))

        for stmt_idx, instruction in enumerate(self.instructions):
            inps = instruction.iexpr.args
            inst = instruction.runner
            op = instruction.iexpr.op
            outs = instruction.retvals()

            input_tensors = [tensor_map[idx] for idx in inps]

            # bind input dtype
            if op.input_like[0] is None:
                op.bind_concrete_types(input_tensors, is_input=True)
            check_type(op, input_tensors, is_input=True, msg="input")

            # Execute op
            # Note: op here is a TorchParsedOp callable wrapping torch functional/module
            output_tensors = inst(*input_tensors)

            if not isinstance(output_tensors, list) and not isinstance(output_tensors, tuple):
                output_tensors = [output_tensors]

            # bind output dtype
            if op.output_like[0] is None:
                op.bind_concrete_types(output_tensors, is_input=False)
            check_type(op, output_tensors, is_input=False, msg="output")

            for i, out_key in enumerate(outs):
                tensor_map[out_key] = output_tensors[i]
                # PyTorch check: if requires_grad is True, it tracks gradients
                self.differentiable &= output_tensors[i].requires_grad

            debug_io(stmt_idx, input_tensors, output_tensors)

            if TORCH_LOG.isEnabledFor(logging.DEBUG):
                if output_tensors[0].requires_grad:
                    for i in range(len(output_tensors)):
                        output_tensors[i].retain_grad()
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
                is_valid = numeric_valid(output_tensors)
                self.invalid_found_last |= not is_valid
                if not is_valid:
                    print(f"\n[DEBUG] Exploded at stmt_idx={stmt_idx}, Op: {op}")
                    for idx, t in enumerate(output_tensors):
                        if t.numel() > 0:
                            max_val = t.abs().max().item()
                            print(f"  -> Output[{idx}] Max Value: {max_val:.4e}")
                            print(f"  -> Requires Grad: {t.requires_grad}, Grad_Fn: {t.grad_fn}")

                if self.invalid_found_last and (
                        self.use_gradient and not self.stop_updating_loss
                ):
                    if vul_op_loss is None:
                        NUMERIC_THRESHOLD = 1e6

                        for out_t in output_tensors:
                            if (out_t.is_floating_point() and
                                    out_t.requires_grad and
                                    out_t.numel() > 0):

                                max_val = out_t.detach().abs().max()

                                if max_val > NUMERIC_THRESHOLD:
                                    loss_val = (out_t.abs() - NUMERIC_THRESHOLD).relu().sum()

                                    if loss_val > 0:
                                        vul_op_loss = ("magnitude_overflow", loss_val)
                                        msg = f"Op {op} overflow {max_val:.2e} > 1e6. Enforcing L1 penalty."
                                        if 'TORCH_LOG' in globals():
                                            TORCH_LOG.debug(msg)
                                        else:
                                            print(f"[Grad Fix] {msg}")

                    if vul_op_loss is None:
                        fallback_loss = torch.tensor(0.0, device=input_tensors[0].device)

                        for inp in input_tensors:
                            if torch.isfinite(inp).all():
                                fallback_loss += inp.pow(2).mean() * 0.01

                        for out in output_tensors:
                            if isinstance(out, torch.Tensor) and torch.isfinite(out).all():
                                fallback_loss += torch.log1p(out.abs()).mean()

                        if fallback_loss == 0.0:
                            fallback_loss = torch.tensor(1.0, device=input_tensors[0].device, requires_grad=True)

                        vul_op_loss = ("fallback_robust", fallback_loss)

                        if TORCH_LOG.isEnabledFor(logging.DEBUG):
                            TORCH_LOG.debug(f"Op {op} produced NaN/Inf. Using Robust L1 Fallback Loss.")

                    ConstraintCheck.true(
                        vul_op_loss is not None,
                        f"{op = } has no `torch_loss` but produces NaN or INF!",
                    )

                    loss_suf, l = vul_op_loss

                    masked_loss = torch.sum(torch.where(l > 0, l, torch.zeros_like(l)))

                    l_min = l.min().item() if l.numel() > 0 else 0.0
                    l_max = l.max().item() if l.numel() > 0 else 0.0

                    msg = f"loss_{loss_suf}: {l_min:.3f} ~ {l_max:.3f} ~ {masked_loss.item()}"
                    TORCH_LOG.debug(f"#{self.n_iter} [NaN/Inf] in output: {op} ~ {msg}")

                    ConstraintCheck.true(
                        torch.all(l > 0),
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
