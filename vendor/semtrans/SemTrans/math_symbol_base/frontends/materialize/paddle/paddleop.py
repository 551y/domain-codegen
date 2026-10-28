import paddle
from typing import Dict, Any, List, Type
import inspect
from functools import wraps

from SemTrans.math_symbol_base.frontends.abstract.op import ParsedOp, OpFactory, BaseAdapter
from SemTrans.math_symbol_base.frontends.abstract.dtype import DType
from SemTrans.math_symbol_base.frontends.abstract.tensor import AbsTensor


class PaddleParsedOp(ParsedOp):
    def __init__(self, op_type: str, attrs: Dict[str, Any], input_names: List[str], output_names: List[str],
                 module: Any = None):
        super().__init__(op_type, attrs, input_names, output_names)
        self.dialect = 'paddle'
        self.module = module

    def get_dialect(self):
        return self.dialect

    def _convert_dtype(self, paddle_dtype):
        return DType.from_paddle(paddle_dtype)

    def __str__(self):
        attrs_str = ""
        if self.attrs:
            attrs_str = f", attrs={list(self.attrs.keys())}"
        return f"{self.dialect}.{self.op_type}(inputs={self.input_names}{attrs_str})"

    def __repr__(self):
        return self.__str__()


class PaddleOpFactory(OpFactory):
    _class_cache: Dict[str, Type[PaddleParsedOp]] = {}

    @staticmethod
    def get_op_instance(op_type: str, attrs: Dict, input_names: List, output_names: List,
                        module: Any = None) -> PaddleParsedOp:
        if op_type not in PaddleOpFactory._class_cache:
            class_name = f"Paddle{op_type.capitalize()}"
            new_class = type(class_name, (PaddleParsedOp,), {})
            PaddleOpFactory._class_cache[op_type] = new_class

        target_class = PaddleOpFactory._class_cache[op_type]
        return target_class(op_type, attrs, input_names, output_names, module=module)


class PaddleAdapter(BaseAdapter):
    _WEIGHT_UNPACKERS = {}

    @classmethod
    def register_unpacker(cls, op_class):
        def wrapper(func):
            cls._WEIGHT_UNPACKERS[op_class] = func
            return func

        return wrapper

    @staticmethod
    def to_abs(obj):
        stride = list(obj.strides) if hasattr(obj, 'strides') else None
        return AbsTensor(
            shape=list(obj.shape),
            dtype=DType.from_paddle(obj.dtype),
            stride=stride
        )

    @staticmethod
    def is_tensor(obj):
        return isinstance(obj, paddle.Tensor)

    @staticmethod
    def parse_op_key(op_obj):
        raw_name = getattr(op_obj, 'op_type', str(op_obj))
        name = raw_name.replace("elementwise_", "").replace("_v2", "").replace("depthwise_", "")
        aliases = {
            'avgpool1d': 'AvgPool1D',
            'avgpool2d': 'AvgPool2D',
            'avgpool3d': 'AvgPool3D',
            'batchnorm': 'BatchNorm2D',
            'batchnorm1d': 'BatchNorm1D',
            'batchnorm2d': 'BatchNorm2D',
            'batchnorm3d': 'BatchNorm3D',
            'pad2d': 'Pad2D',
        }
        return aliases.get(name.lower(), name.capitalize())

    @staticmethod
    def parse_attrs(op_obj):
        return getattr(op_obj, 'attrs', {})


    @staticmethod
    def get_named_tensor(op, name):
        if hasattr(op, 'module') and op.module is not None:
            mod = op.module
            unpacker = PaddleAdapter._WEIGHT_UNPACKERS.get(type(mod))
            if unpacker is not None:
                val = unpacker(mod, name)
                if val is not None:
                    return val

            if hasattr(mod, '_parameters') and name in mod._parameters:
                return mod._parameters[name]
            if hasattr(mod, '_buffers') and name in mod._buffers:
                return mod._buffers[name]

            val = getattr(mod, name, None)
            if isinstance(val, paddle.Tensor):
                return val

            # Paddle names the affine multiplier of normalization layers
            # ``scale`` while the shared MathIR parameter is ``weight``.
            # Resolve that framework-level alias only after a real weight
            # lookup has failed, so convolution and linear parameters retain
            # their native binding.
            if name == 'weight':
                if hasattr(mod, '_parameters') and 'scale' in mod._parameters:
                    return mod._parameters['scale']
                val = getattr(mod, 'scale', None)
                if isinstance(val, paddle.Tensor):
                    return val

        if hasattr(op, 'attrs') and name in op.attrs:
            val = op.attrs[name]
            if isinstance(val, paddle.Tensor):
                return val
        return None

    @staticmethod
    def assign_param(op, name, numpy_val):
        if not hasattr(op, 'module') or op.module is None:
            return False

        tensor = None
        mod = op.module

        if hasattr(mod, '_parameters') and name in mod._parameters:
            tensor = mod._parameters[name]
        elif name == 'weight' and hasattr(mod, '_parameters') and 'scale' in mod._parameters:
            tensor = mod._parameters['scale']
        elif hasattr(mod, '_buffers') and name in mod._buffers:
            tensor = mod._buffers[name]
        else:
            tensor = getattr(mod, name, None)

        if tensor is not None and isinstance(tensor, paddle.Tensor):
            try:
                with paddle.no_grad():
                    new_tensor = paddle.to_tensor(numpy_val).astype(tensor.dtype)
                    if tensor.place.is_gpu_place():
                        new_tensor = new_tensor.cuda()

                    paddle.assign(new_tensor, output=tensor)
                return True
            except Exception as e:
                print(f"[PaddleAdapter] Inject failed for {name}: {e}")
                return False
        return False

    inject_value = assign_param

    @classmethod
    def recover_and_run(cls, op, inputs, symnet):
        if symnet and hasattr(symnet, 'model'):
            if hasattr(symnet.model, 'sublayers'):
                subs = symnet.model.sublayers()
                if len(subs) > 0:
                    return subs[0](*inputs)

        raise RuntimeError(f"PaddleAdapter: op {op} missing module, and recovery failed.")


def _transpose_weight(weight):
    if weight is None:
        return None
    return paddle.transpose(weight, [1, 0])


@PaddleAdapter.register_unpacker(paddle.nn.MultiHeadAttention)
def _unpack_multihead_attention(mod, name):
    tensors = {
        'W_q': _transpose_weight(mod.q_proj.weight),
        'B_q': mod.q_proj.bias if mod.q_proj.bias is not None else 0,
        'W_k': _transpose_weight(mod.k_proj.weight),
        'B_k': mod.k_proj.bias if mod.k_proj.bias is not None else 0,
        'W_v': _transpose_weight(mod.v_proj.weight),
        'B_v': mod.v_proj.bias if mod.v_proj.bias is not None else 0,
        'W_out': _transpose_weight(mod.out_proj.weight),
        'B_out': mod.out_proj.bias if mod.out_proj.bias is not None else 0,
    }
    return tensors.get(name)


def _unpack_attention_projection(attn, prefix, name):
    tensors = {
        f'W_{prefix}_q': _transpose_weight(attn.q_proj.weight),
        f'B_{prefix}_q': attn.q_proj.bias if attn.q_proj.bias is not None else 0,
        f'W_{prefix}_k': _transpose_weight(attn.k_proj.weight),
        f'B_{prefix}_k': attn.k_proj.bias if attn.k_proj.bias is not None else 0,
        f'W_{prefix}_v': _transpose_weight(attn.v_proj.weight),
        f'B_{prefix}_v': attn.v_proj.bias if attn.v_proj.bias is not None else 0,
        f'W_{prefix}_out': _transpose_weight(attn.out_proj.weight),
        f'B_{prefix}_out': attn.out_proj.bias if attn.out_proj.bias is not None else 0,
    }
    return tensors.get(name)


@PaddleAdapter.register_unpacker(paddle.nn.TransformerEncoderLayer)
def _unpack_transformer_encoder_layer(mod, name):
    val = _unpack_attention_projection(mod.self_attn, 'self', name)
    if val is not None:
        return val

    tensors = {
        'W_linear1': _transpose_weight(mod.linear1.weight),
        'B_linear1': mod.linear1.bias if mod.linear1.bias is not None else 0,
        'W_linear2': _transpose_weight(mod.linear2.weight),
        'B_linear2': mod.linear2.bias if mod.linear2.bias is not None else 0,
        'W_norm1': mod.norm1.weight if mod.norm1.weight is not None else 1,
        'B_norm1': mod.norm1.bias if mod.norm1.bias is not None else 0,
        'W_norm2': mod.norm2.weight if mod.norm2.weight is not None else 1,
        'B_norm2': mod.norm2.bias if mod.norm2.bias is not None else 0,
    }
    return tensors.get(name)


@PaddleAdapter.register_unpacker(paddle.nn.TransformerDecoderLayer)
def _unpack_transformer_decoder_layer(mod, name):
    val = _unpack_attention_projection(mod.self_attn, 'self', name)
    if val is not None:
        return val
    val = _unpack_attention_projection(mod.cross_attn, 'cross', name)
    if val is not None:
        return val

    tensors = {
        'W_linear1': _transpose_weight(mod.linear1.weight),
        'B_linear1': mod.linear1.bias if mod.linear1.bias is not None else 0,
        'W_linear2': _transpose_weight(mod.linear2.weight),
        'B_linear2': mod.linear2.bias if mod.linear2.bias is not None else 0,
        'W_norm1': mod.norm1.weight if mod.norm1.weight is not None else 1,
        'B_norm1': mod.norm1.bias if mod.norm1.bias is not None else 0,
        'W_norm2': mod.norm2.weight if mod.norm2.weight is not None else 1,
        'B_norm2': mod.norm2.bias if mod.norm2.bias is not None else 0,
        'W_norm3': mod.norm3.weight if mod.norm3.weight is not None else 1,
        'B_norm3': mod.norm3.bias if mod.norm3.bias is not None else 0,
    }
    return tensors.get(name)


class PaddleFX:
    def __init__(self, atomic_layer_types=()):
        self.instructions = []
        self.tensor_map = {}
        self.constant_values = {}
        self.tensor_refs = {}
        self.output_values = {}
        self.in_atomic_layer = False
        self.atomic_depth = 0
        self.original_attributes = {}
        self.atomic_layer_types = tuple(atomic_layer_types)
        self.layer_paths = {}
        self.output_counter = 0

        self.target_modules = [
            paddle,
            paddle.nn.functional,
            paddle.linalg,
            paddle.fft,
            paddle.signal,
            paddle.tensor
        ]

        self.ignore_names = {
            'backward', 'clear_grad', 'set_value', 'clone',
            'detach', 'stop_gradient', 'name', 'place', 'shape', 'dtype',
            '__init__', '__new__', '__repr__', '__str__',
            '__class__', '__getattr__', '__setattr__', '__del__',
            '__dir__', '__hash__', '__eq__', '__ne__'
        }

    def _flatten(self, item):
        if isinstance(item, (list, tuple)):
            out = []
            for x in item:
                out.extend(self._flatten(x))
            return out
        elif isinstance(item, dict):
            out = []
            for x in item.values():
                out.extend(self._flatten(x))
            return out
        return [item]

    def _get_tensor_name(self, t):
        if not isinstance(t, paddle.Tensor): return str(t)
        tid = id(t)
        self.tensor_refs[tid] = t
        if tid not in self.tensor_map:
            name = f"const_{tid}"
            self.tensor_map[tid] = name
            self.constant_values[name] = t
        return self.tensor_map[tid]

    def _register_output(self, t, op_name):
        if not isinstance(t, paddle.Tensor): return
        tid = id(t)
        self.tensor_refs[tid] = t
        existing = self.tensor_map.get(tid)
        if existing is not None and not existing.startswith(('input_', 'const_')):
            self.output_values[existing] = t
            return existing
        name = f"{op_name}_{len(self.instructions)}_{self.output_counter}_out"
        self.output_counter += 1
        self.tensor_map[tid] = name
        self.output_values[name] = t
        return name

    def _extract_layer_attrs(self, layer):
        """Extract public attributes and simple private aliases from a Paddle layer."""
        attrs = {}

        internal_keys = {
            '_full_name', '_helper', '_built', '_parameters', '_buffers',
            '_sub_layers', '_forward_pre_hooks', '_forward_post_hooks',
            '_custom_parameter_names', '_loaddict_holder'
        }

        for k, v in vars(layer).items():
            if k in internal_keys: continue

            if isinstance(v, (int, float, bool, str, list, tuple, paddle.dtype)):
                if not k.startswith('_'):
                    attrs[k] = v
                elif k.startswith('_') and not k.startswith('__'):
                    public_name = k[1:]
                    if public_name not in attrs:
                        attrs[public_name] = v
        if layer.__class__.__name__.lower().startswith('batchnorm'):
            weight = getattr(layer, 'weight', None)
            if attrs.get('num_features') is None and weight is not None:
                shape = getattr(weight, 'shape', ())
                if len(shape) == 1:
                    attrs['num_features'] = int(shape[0])
        return attrs

    # ==========================================
    # 1. Function wrapper
    # ==========================================
    def _create_wrapper(self, name, original_func, is_method=False):
        @wraps(original_func)
        def wrapper(*args, **kwargs):
            out = original_func(*args, **kwargs)

            if self.in_atomic_layer:
                return out

            all_input_values = list(args) + list(kwargs.values())
            flat_args = self._flatten(all_input_values)
            has_tensor = any(isinstance(x, paddle.Tensor) for x in flat_args)

            if not has_tensor and not is_method:
                return out

            flat_inputs = self._flatten(all_input_values)
            input_names = [self._get_tensor_name(arg) for arg in flat_inputs if isinstance(arg, paddle.Tensor)]

            out_names = []
            flat_outs = self._flatten(out)
            for o in flat_outs:
                if isinstance(o, paddle.Tensor):
                    out_names.append(self._register_output(o, name))

            control_tensor_attrs = {
                'axis', 'axes', 'shape', 'starts', 'ends', 'strides', 'perm'
            }
            attrs = {
                key: value
                for key, value in kwargs.items()
                if key in control_tensor_attrs
                or not any(
                    isinstance(item, paddle.Tensor)
                    for item in self._flatten(value)
                )
            }
            try:
                bound = inspect.signature(original_func).bind_partial(*args, **kwargs)
                for param_name, value in bound.arguments.items():
                    if param_name in {'self', 'args', 'kwargs'}:
                        continue
                    if (
                        param_name in control_tensor_attrs
                        or (
                            not any(
                            isinstance(item, paddle.Tensor)
                            for item in self._flatten(value)
                            )
                            and not inspect.ismodule(value)
                        )
                    ):
                        attrs.setdefault(param_name, value)
            except (TypeError, ValueError):
                pass

            for i, arg in enumerate(args):
                if is_method and i == 0: continue

                if (
                    not any(
                        isinstance(item, paddle.Tensor)
                        for item in self._flatten(arg)
                    )
                    and not inspect.ismodule(arg)
                ):
                    attrs[f"arg_{i}"] = arg

            self.instructions.append({
                'op_type': name,
                'inputs': input_names,
                'outputs': out_names,
                'attrs': attrs,
                'module': None
            })
            return out

        return wrapper

    # ==========================================
    # 2. Patch callable Paddle APIs
    # ==========================================
    def _patch_all(self):
        for module in self.target_modules:
            for name in dir(module):
                if name.startswith('_') or name in self.ignore_names: continue

                try:
                    attr = getattr(module, name)
                except:
                    continue

                if callable(attr) and not isinstance(attr, type):
                    if (module, name) not in self.original_attributes:
                        self.original_attributes[(module, name)] = attr
                        setattr(module, name, self._create_wrapper(name, attr))

        for name in dir(paddle.Tensor):
            if name.startswith('_') or name in self.ignore_names: continue

            try:
                attr = getattr(paddle.Tensor, name)
            except:
                continue

            if callable(attr):
                if (paddle.Tensor, name) not in self.original_attributes:
                    self.original_attributes[(paddle.Tensor, name)] = attr
                    setattr(paddle.Tensor, name, self._create_wrapper(name, attr, is_method=True))

    def _restore_all(self):
        for (obj, name), orig in self.original_attributes.items():
            setattr(obj, name, orig)

    # ==========================================
    # 3. Layer hooks
    # ==========================================
    def _is_atomic_layer(self, layer):
        return len(layer.sublayers()) == 0 or isinstance(layer, self.atomic_layer_types)

    def _pre_hook(self, layer, inputs):
        if self.in_atomic_layer:
            self.atomic_depth += 1
            return

        if self._is_atomic_layer(layer):
            self.in_atomic_layer = True
            self.atomic_depth = 1

    def _post_hook(self, layer, inputs, outputs):
        if self.in_atomic_layer:
            self.atomic_depth -= 1
            if self.atomic_depth > 0:
                return
            self.in_atomic_layer = False

            op_type = layer.__class__.__name__

            attrs = self._extract_layer_attrs(layer)

            inp = inputs if isinstance(inputs, tuple) else (inputs,)
            flat_inp = self._flatten(inp)
            in_names = [self._get_tensor_name(x) for x in flat_inp if isinstance(x, paddle.Tensor)]

            out = outputs if isinstance(outputs, (list, tuple)) else (outputs,)
            flat_out = self._flatten(out)
            out_names = [self._register_output(x, op_type) for x in flat_out if isinstance(x, paddle.Tensor)]

            self.instructions.append({
                'op_type': op_type,
                'inputs': in_names,
                'outputs': out_names,
                'attrs': attrs,
                'module': layer
            })
            self.instructions[-1]['module_path'] = self.layer_paths.get(id(layer), '')

    # ==========================================
    # 4. Main tracing flow
    # ==========================================
    def trace(self, model, dummy_inputs):
        self.instructions = []
        self.tensor_map = {}
        self.constant_values = {}
        self.tensor_refs = {}
        self.output_values = {}
        self.original_attributes = {}
        self.in_atomic_layer = False
        self.atomic_depth = 0
        self.output_counter = 0

        for name, tensor in model.state_dict().items():
            self.tensor_map[id(tensor)] = name
            self.tensor_refs[id(tensor)] = tensor

        raw_inputs = dummy_inputs if isinstance(dummy_inputs, (list, tuple)) else [dummy_inputs]
        real_inputs = []
        for i, x in enumerate(raw_inputs):
            t = paddle.to_tensor(x)
            real_inputs.append(t)
            self.tensor_map[id(t)] = f"input_{i}"
            self.tensor_refs[id(t)] = t

        hooks = []
        sublayers = [('', model)] + list(model.named_sublayers())
        self.layer_paths = {id(layer): name for name, layer in sublayers}

        for _, layer in sublayers:
            hooks.append(layer.register_forward_pre_hook(self._pre_hook))
            hooks.append(layer.register_forward_post_hook(self._post_hook))

        try:
            self._patch_all()

            with paddle.no_grad():
                model(*real_inputs)

        finally:
            for h in hooks: h.remove()
            self._restore_all()

        return self.instructions
