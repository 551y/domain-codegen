# Adapted from NNSmith and modified for SemTrans; see THIRD_PARTY_NOTICES.md.

from functools import reduce
from typing import List

from SemTrans.math_symbol_base.frontends.abstract.arith import *
from SemTrans.math_symbol_base.frontends.abstract.dtype import DType
from SemTrans.math_symbol_base.frontends.abstract.error import ConstraintCheck, SanityCheck


class AbsTensor:
    def __init__(self, shape: List[Union[None, int, z3.ExprRef]], dtype: DType, stride: List[int] = None):
        assert isinstance(
            shape, (list, tuple)
        ), f"Shape must be a list/tuple, but got {shape}"
        self.shape = list(shape)
        self.dtype = DType(dtype)

        if stride is not None:
            self.stride = list(stride)
        elif self.is_concrete():
            self.stride = self._calc_default_stride(self.shape)
        else:
            self.stride = None

    def _calc_default_stride(self, shape):
        if not shape: return []
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return strides

    @property
    def is_contiguous(self) -> bool:
        if self.stride is None:
            return True
        if not self.shape: return True
        expected = 1
        for i in range(len(self.shape) - 1, -1, -1):
            if self.shape[i] == 1: continue
            if self.stride[i] != expected:
                return False
            expected *= self.shape[i]
        return True

    @property
    def is_floating_point(self):
        return self.dtype.is_float()

    @property
    def is_complex(self):
        return "complex" in self.dtype.short()

    def downcast_rank(self):
        return AbsTensor(shape=[None] * self.ndims, dtype=self.dtype)

    def __hash__(self) -> int:
        return hash((tuple(self.shape), self.dtype))

    def __repr__(self) -> str:
        return f"AbsTensor<{self.dtype.short()}>{str(self.shape)}"

    def pretty(self) -> str:
        return f"{self.dtype.short()}{self.shape}"

    def weak_compare(self, other: "AbsTensor") -> bool:
        if self.dtype != other.dtype or self.ndims != other.ndims:
            return False
        for l, r in zip(self.shape, other.shape):
            if isinstance(l, z3.ExprRef) or isinstance(r, z3.ExprRef):
                continue
            if l != r:
                return False
        return True

    def strong_compare(self, other: "AbsTensor") -> bool:
        return self.shape == other.shape and self.dtype == other.dtype

    def __eq__(self, other: "AbsTensor") -> bool:
        return self.strong_compare(other)

    def ge_zero(self):
        ret = []
        for s in self.shape:
            if isinstance(s, z3.ExprRef):
                ret.append(tensor_ge(s, 0))
            else:
                ConstraintCheck.ge(s, 0)
        return ret

    def sym_gt_conc_ge_zero(self):
        ret = []
        for s in self.shape:
            if isinstance(s, z3.ExprRef):
                ret.append(tensor_gt(s, 0))
            else:
                ConstraintCheck.ge(s, 0)
        return ret

    def gt_zero(self):
        ret = []
        for s in self.shape:
            if isinstance(s, z3.ExprRef):
                ret.append(tensor_gt(s, 0))
            else:
                ConstraintCheck.gt(s, 0)
        return ret

    def eq(self, other):
        SanityCheck.eq(self.ndims, other.ndims)
        ret = []
        for i in range(self.ndims):
            if isinstance(self.shape[i], z3.ExprRef) or isinstance(
                other.shape[i], z3.ExprRef
            ):
                ret.append(tensor_eq(self.shape[i], other.shape[i]))
            else:
                ConstraintCheck.eq(self.shape[i], other.shape[i])
        return ret

    def constains_symbol(self) -> bool:
        return any(isinstance(s, z3.ExprRef) for s in self.shape)

    def nelement(self):
        if len(self.shape) == 0:  # Scalar
            return 1
        return reduce(lambda x, y: tensor_mul(x, y), self.shape, 1)

    def nbytes(self) -> int:
        return self.nelement() * self.dtype.sizeof()

    def deepcopy(self):
        return AbsTensor(shape=list(self.shape), dtype=self.dtype, stride=list(self.stride) if self.stride else None)

    @property
    def ndims(self):
        return len(self.shape)

    def is_concrete(self) -> bool:
        return all(isinstance(s, int) for s in self.shape)

    def htype(self):
        return (self.dtype, self.ndims)
