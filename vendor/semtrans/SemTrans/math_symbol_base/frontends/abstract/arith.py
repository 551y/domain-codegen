# Adapted from NNSmith and modified for SemTrans; see THIRD_PARTY_NOTICES.md.

from typing import Union

import z3

from SemTrans.math_symbol_base.frontends.abstract.error import ConstraintCheck, SanityCheck

ARITH_MAX_WIDTH: int = 64


def align_bvs(
        left: Union[float, int, z3.ExprRef],
        right: Union[float, int, z3.ExprRef],
        carry=False,
        mult=False,
):
    """
    Aligns bitvectors to the same width using SIGNED extension.
    CRITICAL FIX: Changed from Zero-Extension to Sign-Extension to support negative coordinates (Padding/Offsets).
    """
    left_is_arith = isinstance(left, (int, float, z3.ArithRef))
    right_is_arith = isinstance(right, (int, float, z3.ArithRef))

    if left_is_arith and right_is_arith:
        return left, right

    left_size = ARITH_MAX_WIDTH
    if isinstance(left, int):
        left_size = min(ARITH_MAX_WIDTH, left.bit_length() + 1)  # +1 for sign bit
    elif isinstance(left, z3.BitVecRef):
        left_size = left.size()

    right_size = ARITH_MAX_WIDTH
    if isinstance(right, int):
        right_size = min(ARITH_MAX_WIDTH, right.bit_length() + 1)
    elif isinstance(right, z3.BitVecRef):
        right_size = right.size()

    # Calculate max width needed
    target_width = max(left_size, right_size)
    if carry: target_width += 1
    if mult: target_width = left_size + right_size

    # Cap at MAX_WIDTH
    target_width = min(target_width, ARITH_MAX_WIDTH)

    # --- Helper to extend values ---
    def extend_to(val, curr_size, target_size):
        if curr_size >= target_size:
            return val

        diff = target_size - curr_size

        if isinstance(val, int):
            # Python ints handle their own extension, just wrap in BitVec
            return z3.BitVecVal(val, target_size)
        elif isinstance(val, z3.BitVecRef):
            # [CRITICAL] Use SignExt instead of Concat(0, ...)
            return z3.SignExt(diff, val)
        else:
            raise RuntimeError(f"Unsupported type for extension: {type(val)}")

    # Convert/Extend Left
    if isinstance(left, (int, float)) and not isinstance(left, z3.ExprRef):
        # If python number, convert directly to target width BitVec
        left = z3.BitVecVal(int(left), target_width)
    elif isinstance(left, z3.BitVecRef):
        left = extend_to(left, left.size(), target_width)

    # Convert/Extend Right
    if isinstance(right, (int, float)) and not isinstance(right, z3.ExprRef):
        right = z3.BitVecVal(int(right), target_width)
    elif isinstance(right, z3.BitVecRef):
        right = extend_to(right, right.size(), target_width)

    return left, right


def tensor_mul(
        left: Union[float, int, z3.ExprRef], right: Union[float, int, z3.ExprRef]
):
    left, right = align_bvs(left, right, mult=True)
    return left * right


def tensor_add(
        left: Union[float, int, z3.ExprRef], right: Union[float, int, z3.ExprRef]
):
    left, right = align_bvs(left, right, carry=True)
    return left + right


def tensor_sub(
        left: Union[float, int, z3.ExprRef], right: Union[float, int, z3.ExprRef]
):
    left, right = align_bvs(left, right, carry=True)  # subtraction can overflow too
    return left - right


def tensor_eq(
        left: Union[float, int, z3.ExprRef], right: Union[float, int, z3.ExprRef]
):
    left, right = align_bvs(left, right)
    return left == right


def tensor_neq(
        left: Union[float, int, z3.ExprRef], right: Union[float, int, z3.ExprRef]
):
    left, right = align_bvs(left, right)
    return left != right


def tensor_ge(
        left: Union[float, int, z3.ExprRef], right: Union[float, int, z3.ExprRef]
):
    left, right = align_bvs(left, right)
    if isinstance(left, z3.BitVecRef) or isinstance(right, z3.BitVecRef):
        return z3.SGE(left, right)
    return left >= right


def tensor_gt(
        left: Union[float, int, z3.ExprRef], right: Union[float, int, z3.ExprRef]
):
    left, right = align_bvs(left, right)
    if isinstance(left, z3.BitVecRef) or isinstance(right, z3.BitVecRef):
        return z3.SGT(left, right)
    return left > right


def tensor_le(
        left: Union[float, int, z3.ExprRef], right: Union[float, int, z3.ExprRef]
):
    left, right = align_bvs(left, right)
    if isinstance(left, z3.BitVecRef) or isinstance(right, z3.BitVecRef):
        return z3.SLE(left, right)
    return left <= right


def tensor_lt(
        left: Union[float, int, z3.ExprRef], right: Union[float, int, z3.ExprRef]
):
    left, right = align_bvs(left, right)
    if isinstance(left, z3.BitVecRef) or isinstance(right, z3.BitVecRef):
        return z3.SLT(left, right)
    return left < right


def tensor_div(
        left: Union[float, int, z3.ExprRef], right: Union[float, int, z3.ExprRef]
):
    left, right = align_bvs(left, right)
    if isinstance(left, z3.BitVecRef) or isinstance(right, z3.BitVecRef):
        return z3.SDiv(left, right)
    if isinstance(left, int) and isinstance(right, int):
        ConstraintCheck.true(right != 0, "Div by zero")
        return left // right
    return left / right


def tensor_mod(
        left: Union[float, int, z3.ExprRef], right: Union[float, int, z3.ExprRef]
):
    left, right = align_bvs(left, right)
    if isinstance(left, z3.BitVecRef) or isinstance(right, z3.BitVecRef):
        return z3.SRem(left, right)  # [FIX] Signed Remainder
    if isinstance(right, int):
        ConstraintCheck.true(right != 0, "Mod by zero")
    return left % right


def tensor_min(left, right):
    # Min needs to be signed-aware too
    if isinstance(left, int) and isinstance(right, int):
        return min(left, right)
    left, right = align_bvs(left, right)
    return z3.If(tensor_le(left, right), left, right)


def tensor_max(left, right):
    # Max needs to be signed-aware too
    if isinstance(left, int) and isinstance(right, int):
        return max(left, right)
    left, right = align_bvs(left, right)
    return z3.If(tensor_ge(left, right), left, right)


def tensor_and(left, right):
    if isinstance(left, bool) and isinstance(right, bool):
        return left and right
    return z3.And(left, right)


def tensor_or(left, right):
    if isinstance(left, bool) and isinstance(right, bool):
        return left or right
    return z3.Or(left, right)


def tensor_not(expr):
    if isinstance(expr, bool):
        return not expr
    return z3.Not(expr)


def tensor_abs(val):
    """
    Returns the absolute value. Crucial for verifying L1 distance or bounds.
    """
    if isinstance(val, (int, float)):
        return abs(val)

    # Assume signed logic for BitVec
    # If val < 0, return -val, else val
    zero = z3.BitVecVal(0, val.size()) if isinstance(val, z3.BitVecRef) else 0
    return z3.If(tensor_lt(val, zero), -val, val)


def tensor_is_in_range(val, low, high):
    """
    Verifies if val is within [low, high).
    Useful for checking if optimized index maps still point to valid memory.
    """
    # Using Signed Comparisons
    return z3.And(tensor_ge(val, low), tensor_lt(val, high))


def tensor_select(cond, true_val, false_val):
    """
    Generic selection (like C++ ternary operator or np.where).
    Useful for Piecewise function verification.
    """
    # Ensure values are aligned before selecting
    true_val, false_val = align_bvs(true_val, false_val)
    return z3.If(cond, true_val, false_val)
