import tokenize
from sympy import symbols, oo
from tokenize import STRING, NAME, OP
from .custom_symbolic_ops import *


def string_token_transform(tokens, local_dict, global_dict):
    result = []
    for token in tokens:
        tok_num, tok_val = token

        if tok_num == STRING:
            result.extend([
                (NAME, 'SymStr'),
                (OP, '('),
                (STRING, tok_val),
                (OP, ')')
            ])
        else:
            result.append(token)
    return result


def keyword_none_transform(tokens, local_dict, global_dict):
    result = []
    for toknum, tokval in tokens:
        if toknum == tokenize.NAME and tokval == 'None':
            result.append((tokenize.NAME, 'S_None'))
        else:
            result.append((toknum, tokval))
    return result


class TensorMixin:
    def cast(self, dtype): return Function('Cast')(self, dtype)
    def __invert__(self): return Function('BitwiseNot')(self)
    def __and__(self, other): return Function('BitwiseAnd')(self, other)
    def __or__(self, other): return Function('BitwiseOr')(self, other)
    def __xor__(self, other): return Function('BitwiseXor')(self, other)
    def __lshift__(self, other): return Function('BitwiseLeftShift')(self, other)
    def __rshift__(self, other): return Function('BitwiseRightShift')(self, other)


def SmartTensor(name):
    return _SmartTensorImpl(name)


def SmartTensors(names):
    if isinstance(names, str):
        name_list = names.replace(',', ' ').split()
    else:
        name_list = names

    res = [_SmartTensorImpl(n) for n in name_list]

    return res[0] if len(res) == 1 else tuple(res)


class _SmartTensorImpl(IndexedBase, TensorMixin):
    pass


class CallMechanism(Function, Boolean):
    is_Boolean = True


class DynamicProxyMeta(type):
    def __getattr__(cls, name):
        if hasattr(sympy, name): return getattr(sympy, name)

        if hasattr(sympy, name.title()): return getattr(sympy, name.title())
        return symbols(name)


class SymbolSpace(metaclass=DynamicProxyMeta):
    # --- Standard Tensors ---
    X = SmartTensor('X')
    Indices = SmartTensor('Indices')
    X1, X2, X3, X4, X5, X6, X7, X8, X9 = SmartTensors('X1 X2 X3 X4 X5 X6 X7 X8 X9')
    W = SmartTensor('W')
    B = SmartTensor('B')
    Y = SmartTensor('Y')
    Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8, Y9 = SmartTensors('Y1 Y2 Y3 Y4 Y5 Y6 Y7 Y8 Y9')
    A = SmartTensor('A')

    # --- RNN Weight and Bias ---
    W_ih = SmartTensor('W_ih')
    W_hh = SmartTensor('W_hh')

    B_ih = SmartTensor('B_ih')
    B_hh = SmartTensor('B_hh')

    # --- Transformer Weight and Bias ---
    W_q = SmartTensor('W_q')
    W_k = SmartTensor('W_k')
    W_v = SmartTensor('W_v')
    W_out = SmartTensor('W_out')

    B_q = SmartTensor('B_q')
    B_k = SmartTensor('B_k')
    B_v = SmartTensor('B_v')
    B_out = SmartTensor('B_out')
    Bias_k_seq = SmartTensor('Bias_k_seq')
    Bias_v_seq = SmartTensor('Bias_v_seq')

    # Buffers
    cutoffs = SmartTensor('cutoffs')
    Query = SmartTensor('Query')
    Key = SmartTensor('Key')
    Value = SmartTensor('Value')
    Memory = SmartTensor('Memory')
    running_mean = SmartTensor('running_mean')
    running_var = SmartTensor('running_var')

    # --- Scalar Symbols ---
    n_vocab = symbols('n_vocab', integer=True)
    pad_idx = symbols('pad_idx', integer=True)
    i, j, k, l, m, n, idx = symbols('i j k l m n idx', integer=True)
    h, w, kh, kw = symbols('h w kh kw', integer=True)
    d, kd = symbols('d kd', integer=True)
    il, id, ih, iw = symbols('il id ih iw', integer=True)
    c, cin, c1, c2, cout, out, f= symbols('c cin c1 c2 cout out f', integer=True)

    Strd, P, P_out, Dil, K, G = symbols('Strd P P_out Dil K G', integer=True)
    R = symbols('R', real=True)
    S_h, S_w, P_h, P_w, P_out_h, OP_h, P_out_w, OP_w, D_h, D_w, K_h, K_w = symbols(
        'S_h S_w P_h P_w P_out_h OP_h P_out_w OP_w D_h D_w K_h K_w', integer=True)
    H_out, L_w = symbols('H_out L_w', integer=True)
    R_h, R_w = symbols('R_h R_w', real=True)
    S_d, P_d, P_out_d, OP_d, D_d, K_d, D_out = symbols('S_d P_d P_out_d OP_d D_d K_d D_out', integer=True)
    R_d = symbols('R_d', real=True)

    epsilon = symbols('epsilon')
    p_norm = symbols('p_norm', real=True)
    p_attn = symbols('p_attn', real=True)
    p_act = symbols('p_act', real=True)
    alpha = symbols('alpha', real=True)
    beta = symbols('beta', real=True)
    norm_bound = symbols('norm_bound', real=True)
    norm_type = symbols('norm_type', real=True)
    min_val = symbols('min_val', real=True)
    max_val = symbols('max_val', real=True)

    dim = symbols('dim', integer=True)
    dims = symbols('dims', integer=True)
    keepdim = symbols('keepdim')

    approx_mode = symbols('approx_mode', integer=True)
    pad_l, pad_r, pad_t, pad_b, pad_f, pad_bk = symbols('pad_l pad_r pad_t pad_b pad_f pad_bk', integer=True)
    pad_val = symbols('pad_val', real=True)
    H, d_model, d_k, d_v, L_q, L_kv = symbols('H d_model d_k d_v L_q L_kv', integer=True)
    batch_first, norm_first, bias, nonlinearity = symbols('batch_first norm_first bias nonlinearity')
    padding_mode = symbols('padding_mode')
    bias_flag = symbols('bias')
    infinity = +oo
    neg_infinity = -oo

    # --- Native Structural Operators ---
    Sum = TensorSum
    Prod = TensorProd
    Piecewise = sympy.Piecewise
    GetItem = GetItem
    Reshape = Reshape

    Mean = Mean
    Var = Var

    Abs = sympy.Abs
    Mod = sympy.Mod
    floor = sympy.floor
    ceiling = sympy.ceiling
    Eq = sympy.Eq
    Ne = sympy.Ne
    Gt = sympy.Gt
    Lt = sympy.Lt
    And = sympy.And
    Or = sympy.Or
    Not = sympy.Not
    Pi = sympy.pi
    Sin = sympy.sin
    Cos = sympy.cos
    Acos = sympy.acos
    Asin = sympy.asin
    Atan = sympy.atan
    Atan2 = sympy.atan2
    Cosh = sympy.cosh
    Acosh = sympy.acosh
    Asinh = sympy.asinh

    # --- Abstract Computational Operators ---
    Min = Function('Min')
    Max = Function('Max')
    ArgMax = Function('ArgMax')
    ArgMin = Function('ArgMin')
    Exp = Function('Exp')
    Log = Function('Log')
    Sqrt = Function('Sqrt')
    Round = Function('Round')

    # --- Linear Algebra & Tensor ---
    GetAttr = Function('GetAttr')
    MatMul = Function('MatMul')
    EmbeddingLookup = Function('EmbeddingLookup')
    Softmax = Function('Softmax')
    Transpose = Function('Transpose')
    AxisBroadcastShape = Function('AxisBroadcastShape')
    Squeeze = Function('Squeeze')
    ReduceMean = Function('ReduceMean')
    ReduceMax = Function('ReduceMax')
    ReduceSum = Function('ReduceSum')
    Split = Function('Split')
    Tile = Function('Tile')
    SliceFill = Function('SliceFill')
    Pad = Function('Pad')
    PadShape = Function('PadShape')
    StridedSlice = Function('StridedSlice')
    ExpandDims = Function('ExpandDims')
    Gather = Function('Gather')
    SplitV = Function('SplitV')
    ResizeNearestNeighbor = Function('ResizeNearestNeighbor')
    Resize2D = Function('Resize2D')
    Cast = Function('Cast')
    Shape = Function('Shape')
    Rank = Function('Rank')
    Dim = Function('Dim')
    GetItem = GetItem

    ATOMS = {
        'Min', 'Max', 'Sum', 'Mean', 'Var', 'Prod', 'ArgMax', 'ArgMin',
        'Exp', 'Log', 'Sqrt', 'Round',
        'MatMul', 'EmbeddingLookup', 'Softmax', 'Transpose', 'AxisBroadcastShape', 'Squeeze', 'ReduceMean', 'ReduceMax', 'ReduceSum', 'Reshape', 'Split', 'SplitV', 'Tile', 'SliceFill', 'Pad', 'PadShape', 'StridedSlice', 'ExpandDims', 'Gather', 'ResizeNearestNeighbor', 'Resize2D', 'Cast', 'Concat',
        'Shape', 'Rank', 'Dim', 'GetItem', 'GetAttr', 'Join',
        'Relu', 'Gelu', 'Dropout', 'ScaledDotProductAttention',
        'CallMechanism', 'Cast', 'Piecewise', 'is_list',
        'BitwiseNot', 'BitwiseAnd', 'BitwiseOr', 'BitwiseXor', 'BitwiseLeftShift', 'BitwiseRightShift'
    }

    @classmethod
    def is_atomic(cls, name):
        return name in cls.ATOMS

    @classmethod
    def get_env(cls):
        def _symbolic_len(obj):
            if isinstance(obj, (Expr, Symbol, IndexedBase)):
                return cls.Rank(obj)
            if hasattr(obj, 'size') and not isinstance(obj, (list, tuple, dict, set, str)):
                return obj.size
            try:
                return len(obj)
            except:
                return cls.Rank(obj)

        env = {k: v for k, v in cls.__dict__.items() if not k.startswith('_')}
        env['BitwiseAnd'] = BitwiseAnd
        env['BitwiseNot'] = BitwiseNot
        env['BitwiseOr'] = BitwiseOr
        env['BitwiseXor'] = BitwiseXor

        env['S_None'] = S_None
        if 'None' in env: del env['None']
        env['SymStr'] = SymStr
        env['is_list'] = is_list
        env['join'] = join
        env['concat'] = Concat
        env['True'] = S.true
        env['False'] = S.false
        env['GetItem'] = cls.GetItem
        env['len'] = _symbolic_len
        return env


def resolve_split_to_indexed(expr):
    def _expand_logic(node):
        if isinstance(node, Indexed) and hasattr(node.base.label, 'func'):
            if node.base.label.func.__name__ == 'Split':
                split_args = node.base.label.args
                X_tensor = split_args[0]
                dim_axis = int(split_args[1])
                k_idx = split_args[2]
                T_total = split_args[3]
                current_indices = list(node.indices)

                offset = k_idx * (SymbolSpace.Dim(X_tensor, dim_axis) / T_total)
                current_indices[dim_axis] = current_indices[dim_axis] + offset
                return Indexed(X_tensor, *current_indices)
        return node
    return expr.replace(Indexed, _expand_logic)


def apply_symbolic_lowering(expr):
    def _expand_logic(node):
        if isinstance(node, Function) and node.func.__name__ == 'Split':
            return _lowering_split_core(node)

        if isinstance(node, Indexed) and hasattr(node.base.label, 'func'):
            if node.base.label.func.__name__ == 'Split':
                return _lowering_split_core(node.base.label, node.indices)

        return node

    def _lowering_split_core(split_func, indices=None):
        X_tensor, d_axis, k_idx, T_total = split_func.args
        dim_axis = int(d_axis)
        if indices is None:
            all_axes = [SymbolSpace.n, SymbolSpace.c, SymbolSpace.h, SymbolSpace.w,
                        SymbolSpace.l, SymbolSpace.d]
            indices = all_axes
        offset = k_idx * (SymbolSpace.Dim(X_tensor, dim_axis) / T_total)
        current_indices = list(indices)
        if dim_axis < len(current_indices):
            current_indices[dim_axis] = current_indices[dim_axis] + offset
        return Indexed(X_tensor, *current_indices)

    return expr.replace(lambda x: isinstance(x, (Function, Indexed)), _expand_logic)


OPERATOR_LOWERING_MAPS = {
    'Split': resolve_split_to_indexed
}
