import paddle
import paddle.nn.functional as F

from SemTrans.math_symbol_base.frontends.abstract import OpLossManager


@paddle.jit.not_to_static
def numeric_valid(tensors, threshold=1e6):
    for t in tensors:
        if not paddle.isfinite(t).all():
            return False
        if t.abs().max() > threshold:
            return False

    return True


def loss_ge(x, val): return F.relu(val - x)  # x >= val


def loss_le(x, val): return F.relu(x - val)  # x <= val


def loss_gt_zero(x): return F.relu(1e-6 - x)  # x > 0


def loss_neq_zero(x): return F.relu(1e-4 - paddle.abs(x))  # x != 0


@OpLossManager.register("acos", framework="paddle")
def loss_acos(op):
    return lambda x: loss_le(paddle.abs(x), 1.0)


@OpLossManager.register("asin", framework="paddle")
def loss_asin(op):
    
    return lambda x: loss_le(paddle.abs(x), 1.0)


@OpLossManager.register("sqrt", framework="paddle")
def loss_sqrt(op):
    
    return lambda x: loss_ge(x, 0.0)


@OpLossManager.register("log", framework="paddle")
def loss_log(op):
    
    return lambda x: loss_gt_zero(x)


@OpLossManager.register("log2", framework="paddle")
def loss_log2(op):
    return lambda x: loss_gt_zero(x)


@OpLossManager.register("log10", framework="paddle")
def loss_log10(op):
    return lambda x: loss_gt_zero(x)


@OpLossManager.register("reciprocal", framework="paddle")
def loss_reciprocal(op):
    return lambda x: loss_neq_zero(x)
