import torch
from SemTrans.math_symbol_base.frontends.abstract.op import ParsedOp, OpFactory, BaseAdapter
from SemTrans.math_symbol_base.frontends.abstract.dtype import DType
from SemTrans.math_symbol_base.frontends.abstract.tensor import AbsTensor
from typing import Dict, Any, List, Type


class TorchParsedOp(ParsedOp):
    def __init__(self, op_type: str, attrs: Dict[str, Any], input_names: List[str], output_names: List[str], module: Any = None):
        super().__init__(op_type, attrs, input_names, output_names)
        self.dialect = 'torch'
        self.module = module  

    def get_dialect(self):
        return self.dialect

    def _convert_dtype(self, torch_dtype):
        return DType.from_torch(torch_dtype)

    def __str__(self):
        attrs_str = ""
        if self.attrs:
            attrs_str = f", attrs={list(self.attrs.keys())}"
        return f"{self.dialect}.{self.op_type}(inputs={self.input_names}{attrs_str})"

    def __repr__(self):
        return self.__str__()

class TorchOpFactory(OpFactory):
    _class_cache: Dict[str, Type[TorchParsedOp]] = {}

    @staticmethod
    def get_op_instance(op_type: str, attrs: Dict, input_names: List, output_names: List, module: Any = None) -> TorchParsedOp:
        if op_type not in TorchOpFactory._class_cache:
            safe_name = op_type.replace("::", "_").replace(".", "_")
            class_name = f"Torch{safe_name.capitalize()}"
            new_class = type(class_name, (TorchParsedOp,), {})

            TorchOpFactory._class_cache[op_type] = new_class

        target_class = TorchOpFactory._class_cache[op_type]
        
        return target_class(op_type, attrs, input_names, output_names, module=module)


class TorchAdapter(BaseAdapter):
    _WEIGHT_UNPACKERS = {}

    @classmethod
    def register_unpacker(cls, op_class):
        def wrapper(func):
            cls._WEIGHT_UNPACKERS[op_class] = func
            return func

        return wrapper

    @staticmethod
    def to_abs(obj):
        return AbsTensor(
            shape=list(obj.shape),
            dtype=DType.from_torch(obj.dtype),
            stride=list(obj.stride())
        )

    @staticmethod
    def is_tensor(obj):
        return isinstance(obj, torch.Tensor)

    @staticmethod
    def parse_op_key(op_obj):
        raw_name = getattr(op_obj, 'op_type', str(op_obj))
        return raw_name.split('.')[-1]

    @staticmethod
    def parse_attrs(op_obj):
        attrs = getattr(op_obj, 'attrs', {}).copy()
        if hasattr(op_obj, 'module') and op_obj.module is not None:
            mod = op_obj.module

            try:
                instance_attrs = vars(mod)
            except TypeError:
                instance_attrs = {}

            for key, val in instance_attrs.items():
                if key.startswith('_'):
                    continue
                if isinstance(val, (torch.Tensor, torch.nn.Parameter, torch.nn.Module)):
                    continue

                if key not in attrs:
                    attrs[key] = val

        return attrs

    @staticmethod
    def get_named_tensor(op, name):
        if hasattr(op, 'module') and op.module is not None:
            mod = op.module

            if type(mod) in TorchAdapter._WEIGHT_UNPACKERS:
                val = TorchAdapter._WEIGHT_UNPACKERS[type(mod)](mod, name)
                if val is not None:
                    return val

            val = getattr(mod, name, None)
            if isinstance(val, torch.Tensor):
                return val
            if val is not None:
                return val

            for n, p in mod.named_parameters():
                if n == name: return p
            for n, b in mod.named_buffers():
                if n == name: return b

            for n, p in mod.named_parameters():
                if n == name: return p
            for n, b in mod.named_buffers():
                if n == name: return b

        if hasattr(op, 'attrs') and name in op.attrs:
            val = op.attrs[name]
            if isinstance(val, torch.Tensor):
                return val

        return None

    @staticmethod
    def assign_param(op, name, numpy_val):
        if not hasattr(op, 'module') or op.module is None:
            return

        tensor = getattr(op.module, name, None)
        if tensor is not None and isinstance(tensor, torch.Tensor):
            with torch.no_grad():
                new_tensor = torch.from_numpy(numpy_val).to(tensor.device, tensor.dtype)
                tensor.copy_(new_tensor)

    @classmethod
    def recover_and_run(cls, op, inputs, symnet):
        if symnet and hasattr(symnet, 'model'):
            children = list(symnet.model.children())
            if len(children) > 0:
                return children[0](*inputs)

        raise RuntimeError(f"TorchAdapter: op {op} missing module, and recovery failed.")

    @classmethod
    def run_reference_inference(cls, op, inputs, symnet=None):
        executable = op

        if hasattr(op, 'module') and not isinstance(op, torch.nn.Module):
            if op.module is not None:
                executable = op.module
        cpu_inputs = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                cpu_inputs.append(torch.zeros_like(inp, device='cpu'))
            else:
                cpu_inputs.append(inp)

        try:
            if isinstance(executable, torch.nn.Module):
                try:
                    executable.to('cpu')
                except:
                    pass
            with torch.no_grad():
                return executable(*cpu_inputs)

        except Exception as e:
            print(f"\n[Dry Run Error] Op: {type(op)}")
            print(f"  > Executable: {type(executable)}")
            print(f"  > Error: {e}")
            return cls.recover_and_run(op, inputs, symnet)


@TorchAdapter.register_unpacker(torch.nn.MultiheadAttention)
def _unpack_mha_weights(mod, name):
    embed_dim = mod.embed_dim

    if name == 'W_q':
        if mod.in_proj_weight is not None:
            return mod.in_proj_weight[0:embed_dim, :]
        else:
            return getattr(mod, 'q_proj_weight', None)
    elif name == 'W_k':
        if mod.in_proj_weight is not None:
            return mod.in_proj_weight[embed_dim:2 * embed_dim, :]
        else:
            return getattr(mod, 'k_proj_weight', None)
    elif name == 'W_v':
        if mod.in_proj_weight is not None:
            return mod.in_proj_weight[2 * embed_dim:3 * embed_dim, :]
        else:
            return getattr(mod, 'v_proj_weight', None)

    elif name == 'B_q':
        if mod.in_proj_bias is not None: return mod.in_proj_bias[0:embed_dim]
    elif name == 'B_k':
        if mod.in_proj_bias is not None: return mod.in_proj_bias[embed_dim:2 * embed_dim]
    elif name == 'B_v':
        if mod.in_proj_bias is not None: return mod.in_proj_bias[2 * embed_dim:3 * embed_dim]

    elif name == 'W_out':
        return mod.out_proj.weight if hasattr(mod.out_proj, 'weight') else None
    elif name == 'B_out':
        return mod.out_proj.bias if hasattr(mod.out_proj, 'bias') else None
    elif name == 'Bias_k_seq':
        return getattr(mod, 'bias_k', None)
    elif name == 'Bias_v_seq':
        return getattr(mod, 'bias_v', None)

    return None


def _unpack_qkv_projection(attn, prefix, name):
    embed_dim = attn.embed_dim
    if name == f'W_{prefix}_q':
        if attn.in_proj_weight is not None:
            return attn.in_proj_weight[0:embed_dim, :]
        return getattr(attn, 'q_proj_weight', None)
    if name == f'W_{prefix}_k':
        if attn.in_proj_weight is not None:
            return attn.in_proj_weight[embed_dim:2 * embed_dim, :]
        return getattr(attn, 'k_proj_weight', None)
    if name == f'W_{prefix}_v':
        if attn.in_proj_weight is not None:
            return attn.in_proj_weight[2 * embed_dim:3 * embed_dim, :]
        return getattr(attn, 'v_proj_weight', None)
    if name == f'B_{prefix}_q':
        return attn.in_proj_bias[0:embed_dim] if attn.in_proj_bias is not None else 0
    if name == f'B_{prefix}_k':
        return attn.in_proj_bias[embed_dim:2 * embed_dim] if attn.in_proj_bias is not None else 0
    if name == f'B_{prefix}_v':
        return attn.in_proj_bias[2 * embed_dim:3 * embed_dim] if attn.in_proj_bias is not None else 0
    if name == f'W_{prefix}_out':
        return attn.out_proj.weight
    if name == f'B_{prefix}_out':
        return attn.out_proj.bias if attn.out_proj.bias is not None else 0
    return None


@TorchAdapter.register_unpacker(torch.nn.TransformerEncoderLayer)
def _unpack_transformer_encoder_layer_weights(mod, name):
    val = _unpack_qkv_projection(mod.self_attn, 'self', name)
    if val is not None:
        return val

    tensor_map = {
        'W_linear1': mod.linear1.weight,
        'B_linear1': mod.linear1.bias if mod.linear1.bias is not None else 0,
        'W_linear2': mod.linear2.weight,
        'B_linear2': mod.linear2.bias if mod.linear2.bias is not None else 0,
        'W_norm1': mod.norm1.weight if mod.norm1.weight is not None else 1,
        'B_norm1': mod.norm1.bias if mod.norm1.bias is not None else 0,
        'W_norm2': mod.norm2.weight if mod.norm2.weight is not None else 1,
        'B_norm2': mod.norm2.bias if mod.norm2.bias is not None else 0,
    }
    return tensor_map.get(name)


@TorchAdapter.register_unpacker(torch.nn.TransformerDecoderLayer)
def _unpack_transformer_decoder_layer_weights(mod, name):
    val = _unpack_qkv_projection(mod.self_attn, 'self', name)
    if val is not None:
        return val
    val = _unpack_qkv_projection(mod.multihead_attn, 'cross', name)
    if val is not None:
        return val

    tensor_map = {
        'W_linear1': mod.linear1.weight,
        'B_linear1': mod.linear1.bias if mod.linear1.bias is not None else 0,
        'W_linear2': mod.linear2.weight,
        'B_linear2': mod.linear2.bias if mod.linear2.bias is not None else 0,
        'W_norm1': mod.norm1.weight if mod.norm1.weight is not None else 1,
        'B_norm1': mod.norm1.bias if mod.norm1.bias is not None else 0,
        'W_norm2': mod.norm2.weight if mod.norm2.weight is not None else 1,
        'B_norm2': mod.norm2.bias if mod.norm2.bias is not None else 0,
        'W_norm3': mod.norm3.weight if mod.norm3.weight is not None else 1,
        'B_norm3': mod.norm3.bias if mod.norm3.bias is not None else 0,
    }
    return tensor_map.get(name)
