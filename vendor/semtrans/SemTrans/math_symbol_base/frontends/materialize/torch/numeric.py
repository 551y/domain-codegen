# Adapted from NNSmith and modified for SemTrans; see THIRD_PARTY_NOTICES.md.

import torch
import torch.nn.functional as F

from SemTrans.math_symbol_base.frontends.abstract import OpLossManager


@torch.jit.ignore
def numeric_valid(tensors, threshold=1e6):
    for t in tensors:
        if not torch.isfinite(t).all():
            return False
        if t.abs().max() > threshold:
            return False

    return True


def loss_ge(x, val): return F.relu(val - x)  # x >= val


def loss_le(x, val): return F.relu(x - val)  # x <= val


def loss_gt_zero(x): return F.relu(1e-6 - x)  # x > 0


def loss_neq_zero(x): return F.relu(1e-4 - torch.abs(x))  # x != 0


@OpLossManager.register("acos", framework="torch")
def loss_acos(op):
    return lambda x: loss_le(torch.abs(x), 1.0)


@OpLossManager.register("asin", framework="torch")
def loss_asin(op):
    
    return lambda x: loss_le(torch.abs(x), 1.0)


@OpLossManager.register("sqrt", framework="torch")
def loss_sqrt(op):
    
    return lambda x: loss_ge(x, 0.0)


@OpLossManager.register("log", framework="torch")
def loss_log(op):
    
    return lambda x: loss_gt_zero(x)


@OpLossManager.register("log2", framework="torch")
def loss_log2(op):
    return lambda x: loss_gt_zero(x)


@OpLossManager.register("log10", framework="torch")
def loss_log10(op):
    return lambda x: loss_gt_zero(x)


@OpLossManager.register("reciprocal", framework="torch")
def loss_reciprocal(op):
    return lambda x: loss_neq_zero(x)
