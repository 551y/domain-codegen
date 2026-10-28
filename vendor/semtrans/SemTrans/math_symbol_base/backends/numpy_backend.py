import numpy as np
import functools
import scipy.special as sp
from typing import Union, Any


# =========================================================
# Sparse Broadcast Masker
# =========================================================
class _SparseReducer:
    def __init__(self, numpy_func, default_val):
        self.np_func = numpy_func
        self.default = default_val

    def reduce(self, args, *oargs, **kwargs):
        if isinstance(args, (list, tuple)):
            if not args: return np.array(self.default)
            return functools.reduce(self.np_func, args)

        return self.np_func.reduce(args, *oargs, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.np_func(*args, **kwargs)


class SmartLogicalOp:
    def __init__(self, op):
        self.op = op

    def __call__(self, *args):
        return functools.reduce(self.op, args)

    def reduce(self, args, axis=None, keepdims=False):
        if isinstance(args, (tuple, list)):
            return functools.reduce(self.op, args)
        return self.op.reduce(args, axis=axis, keepdims=keepdims)

_smart_and = SmartLogicalOp(np.logical_and)
_smart_or = SmartLogicalOp(np.logical_or)


# =========================================================
# Impl Helper Functions
# =========================================================
def _impl_hybrid_getitem(base, *indices):
    if base is None or isinstance(base, (bool, int, float, np.bool_, np.number)):
        return base

    if isinstance(base, (list, tuple)):
        idx = int(indices[0])
        return base[idx]

    if hasattr(base, 'shape'):
        safe_indices = []
        has_array_index = False
        for i, idx in enumerate(indices):
            if i >= len(base.shape):
                break

            if isinstance(idx, np.ndarray):
                idx = idx.astype(np.intp)
                has_array_index = True
            else:
                idx = int(idx)

            safe_indices.append(np.clip(idx, 0, base.shape[i] - 1))

        if has_array_index:
            index_arrays = [
                idx if isinstance(idx, np.ndarray) else np.asarray(idx, dtype=np.intp)
                for idx in safe_indices
            ]
            broadcasted_indices = np.broadcast_arrays(*index_arrays)
            flat_indices = np.ravel_multi_index(
                tuple(arr.reshape(-1) for arr in broadcasted_indices),
                base.shape,
                mode='clip'
            )
            return base.reshape(-1)[flat_indices].reshape(broadcasted_indices[0].shape)

        return base[tuple(safe_indices)]

    # Fallback
    raise ValueError(f"GetItem received unsupported base type: {type(base)}")


def _impl_dim(obj, axis):
    axis = int(axis)
    if hasattr(obj, 'shape'):
        return obj.shape[axis]
    if isinstance(obj, (list, tuple)):
        return obj[axis]
    if hasattr(obj, 'data') and hasattr(obj.data, 'shape'):
        return obj.data.shape[axis]
    return 0


def _impl_rank(obj):
    if hasattr(obj, 'ndim'):
        return obj.ndim
    if hasattr(obj, 'shape'):
        return len(obj.shape)
    if isinstance(obj, (list, tuple)):
        return len(obj)
    if hasattr(obj, 'data') and hasattr(obj.data, 'shape'):
        return len(obj.data.shape)
    return 0


def _impl_shape(obj):
    if hasattr(obj, 'shape'):
        return tuple(obj.shape)
    if isinstance(obj, (list, tuple)):
        return tuple(obj)
    if hasattr(obj, 'data') and hasattr(obj.data, 'shape'):
        return tuple(obj.data.shape)
    return ()


def _impl_and(*args):
    if not args: return np.array(True)
    return functools.reduce(np.logical_and, args)


def _impl_or(*args):
    if not args: return np.array(False)
    return functools.reduce(np.logical_or, args)


def _impl_piecewise(*args):
    if not args:
        return np.array(0.0, dtype=np.float32)

    value_arrays = [np.asarray(value) for value, _ in args]
    value_dtypes = [value.dtype for value in value_arrays if value.dtype.kind != 'O']
    try:
        result_dtype = np.result_type(*value_dtypes) if value_dtypes else np.dtype(np.float32)
    except TypeError:
        result_dtype = np.dtype(np.float32)

    last_val, last_cond = args[-1]
    last_cond_arr = np.asarray(last_cond, dtype=bool)
    result = np.asarray(last_val, dtype=result_dtype)

    for val, cond in reversed(args[:-1]):
        cond_arr = np.asarray(cond, dtype=bool)
        val_arr = np.asarray(val, dtype=result_dtype)

        try:
            cond_b, result_b = np.broadcast_arrays(cond_arr, result)
        except ValueError:
            # A tensor predicate often covers all logical output positions
            # except the final feature dimension (e.g. [B, L] for [B, L, D]).
            # Preserve ordinary NumPy trailing-dimension broadcasting first,
            # then support this explicit prefix-predicate convention.
            if cond_arr.ndim < result.ndim and tuple(cond_arr.shape) == tuple(result.shape[:cond_arr.ndim]):
                cond_arr = cond_arr.reshape(cond_arr.shape + (1,) * (result.ndim - cond_arr.ndim))
                cond_b, result_b = np.broadcast_arrays(cond_arr, result)
            else:
                raise
        val_b, result_b = np.broadcast_arrays(val_arr, result_b)
        result = np.where(cond_b, val_b, result_b)

    return result


def _parse_axis(args):
    if not args: return None
    ax = args[0]
    if isinstance(ax, (int, np.integer)):
        return int(ax)
    if isinstance(ax, np.ndarray):
        if ax.ndim == 0 or ax.size == 1:
            return int(np.asarray(ax).reshape(-1)[0])
        return None
    if isinstance(ax, (list, tuple)):
        parsed = []
        for x in ax:
            if isinstance(x, (int, np.integer)):
                parsed.append(int(x))
            elif isinstance(x, np.ndarray):
                if x.ndim == 0 or x.size == 1:
                    parsed.append(int(np.asarray(x).reshape(-1)[0]))
                else:
                    return None
            else:
                try:
                    parsed.append(int(x))
                except Exception:
                    return None
        return tuple(parsed)
    try:
        return int(ax)
    except:
        return None


def _impl_max(*args):
    if not args: return np.array(-np.inf)
    valid_args = [a for a in args if not isinstance(a, tuple)]
    if not valid_args: return np.array(-np.inf)
    return functools.reduce(np.maximum, valid_args)


def _impl_min(*args):
    if not args: return np.array(np.inf)
    valid_args = [a for a in args if not isinstance(a, tuple)]
    if not valid_args: return np.array(np.inf)
    return functools.reduce(np.minimum, valid_args)


def _impl_argmax(*args):
    if not args:
        return np.array(0)
    x = np.asarray(args[0])
    axes = []
    for arg in args[1:]:
        try:
            axis = int(np.asarray(arg).reshape(-1)[0])
            axes.append(axis)
        except Exception:
            pass
    if not axes:
        return np.asarray(np.argmax(x), dtype=np.float32)
    axes = tuple(sorted({a if a >= 0 else x.ndim + a for a in axes}))
    if len(axes) == 1:
        axis = axes[0]
        return np.expand_dims(np.argmax(x, axis=axis), axis=axis).astype(np.float32)

    remaining_axes = [i for i in range(x.ndim) if i not in axes]
    moved = np.moveaxis(x, axes, range(len(remaining_axes), x.ndim))
    flat_shape = moved.shape[:len(remaining_axes)] + (int(np.prod([x.shape[a] for a in axes])),)
    flat = moved.reshape(flat_shape)
    result = np.argmax(flat, axis=-1)
    for axis in axes:
        result = np.expand_dims(result, axis=axis)
    return result.astype(np.float32)


def _impl_argmin(*args):
    if not args:
        return np.array(0)
    x = np.asarray(args[0])
    axes = []
    for arg in args[1:]:
        try:
            axis = int(np.asarray(arg).reshape(-1)[0])
            axes.append(axis)
        except Exception:
            pass
    if not axes:
        return np.asarray(np.argmin(x), dtype=np.float32)
    axes = tuple(sorted({a if a >= 0 else x.ndim + a for a in axes}))
    if len(axes) == 1:
        axis = axes[0]
        return np.expand_dims(np.argmin(x, axis=axis), axis=axis).astype(np.float32)

    remaining_axes = [i for i in range(x.ndim) if i not in axes]
    moved = np.moveaxis(x, axes, range(len(remaining_axes), x.ndim))
    flat_shape = moved.shape[:len(remaining_axes)] + (int(np.prod([x.shape[a] for a in axes])),)
    flat = moved.reshape(flat_shape)
    result = np.argmin(flat, axis=-1)
    for axis in axes:
        result = np.expand_dims(result, axis=axis)
    return result.astype(np.float32)


def _impl_transpose(x, *args):
    if len(args) == 0:
        return x.T

    if len(args) == 1 and isinstance(args[0], (tuple, list, np.ndarray)):
        return np.transpose(x, args[0])

    if len(args) == 2 and not isinstance(args[0], (tuple, list, np.ndarray)):
        return np.swapaxes(x, args[0], args[1])

    return np.transpose(x, args)


def _impl_make_swap_perm(rank, dim_a, dim_b):
    rank = int(rank)
    dim_a = int(dim_a)
    dim_b = int(dim_b)
    if dim_a < 0:
        dim_a += rank
    if dim_b < 0:
        dim_b += rank
    perm = list(range(rank))
    perm[dim_a], perm[dim_b] = perm[dim_b], perm[dim_a]
    return tuple(perm)


def _impl_axis_broadcast_shape(rank, axis, size):
    rank = int(rank)
    axis = int(axis)
    size = int(size)
    if axis < 0:
        axis += rank
    if rank < 1 or axis < 0 or axis >= rank:
        raise IndexError(f"Broadcast axis {axis} is outside tensor rank {rank}")
    shape = [1] * rank
    shape[axis] = size
    return tuple(shape)


def _impl_squeeze(x, dim=None):
    value = np.asarray(x)
    if (
        dim is None
        or type(dim).__name__ == 'NoneSymbol'
        or (isinstance(dim, str) and dim.strip("'\"").lower() == 'all')
        or (callable(dim) and getattr(dim, '__name__', '').lower() == 'all')
    ):
        return np.squeeze(value)

    axes = []
    for raw_axis in np.asarray(dim).reshape(-1):
        axis = int(raw_axis)
        if axis < 0:
            axis += value.ndim
        if axis < 0 or axis >= value.ndim:
            raise IndexError(
                f"Squeeze dim {axis} is outside tensor rank {value.ndim}"
            )
        axes.append(axis)

    # PyTorch and Paddle both preserve selected non-singleton dimensions.
    removable_axes = tuple(sorted({axis for axis in axes if value.shape[axis] == 1}))
    if not removable_axes:
        return value
    return np.squeeze(value, axis=removable_axes)


def _impl_reduce_mean(x, axis=None, keepdim=False):
    if (
        axis is None
        or type(axis).__name__ == 'NoneSymbol'
        or (isinstance(axis, str) and axis.strip("'\"").lower() == 'all')
        or (callable(axis) and getattr(axis, '__name__', '').lower() == 'all')
    ):
        parsed_axis = None
    else:
        parsed_axis = _parse_axis((axis,))
        if parsed_axis is None:
            raise ValueError(f"Unsupported ReduceMean axis: {axis!r}")
    return np.mean(np.asarray(x), axis=parsed_axis, keepdims=bool(keepdim))


def _impl_reduce_max(x, axis=None, keepdim=False):
    if axis is None:
        parsed_axis = None
    else:
        parsed_axis = _parse_axis((axis,))
        if parsed_axis is None:
            raise ValueError(f"Unsupported ReduceMax axis: {axis!r}")
    return np.max(np.asarray(x), axis=parsed_axis, keepdims=bool(keepdim))


def _impl_gelu(x, approximate=False):
    value = np.asarray(x)
    if isinstance(approximate, (bool, np.bool_)):
        use_tanh = bool(approximate)
    else:
        use_tanh = str(approximate).strip("'\"").lower() in {'tanh', 'true', '1'}
    if use_tanh:
        inner = np.sqrt(2.0 / np.pi) * (value + 0.044715 * np.power(value, 3))
        return 0.5 * value * (1.0 + np.tanh(inner))
    return 0.5 * value * (1.0 + sp.erf(value / np.sqrt(2.0)))


def _impl_softmax(x, dim=-1):
    e_x = np.exp(x - np.max(x, axis=dim, keepdims=True))
    return e_x / np.sum(e_x, axis=dim, keepdims=True)


def _uses_float_dtype(*values):
    for value in values:
        try:
            if np.asarray(value).dtype.kind == 'f':
                return True
        except Exception:
            pass
    return False


def _impl_bitwise_and(x, y):
    if _uses_float_dtype(x, y):
        return np.logical_and(x, y)
    return np.bitwise_and(x, y)


def _impl_bitwise_or(x, y):
    if _uses_float_dtype(x, y):
        return np.logical_or(x, y)
    return np.bitwise_or(x, y)


def _impl_bitwise_xor(x, y):
    if _uses_float_dtype(x, y):
        return np.logical_xor(x, y)
    return np.bitwise_xor(x, y)


def _impl_layer_norm(x, gamma=None, beta=None, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    out = (x - mean) / np.sqrt(var + eps)
    if gamma is not None: out = out * gamma
    if beta is not None: out = out + beta
    return out


def _impl_tensor_sum(x, *args):
    dims_to_reduce = []
    for arg in args:
        if isinstance(arg, (int, np.integer)):
            dims_to_reduce.append(int(arg))
            continue

        if isinstance(arg, np.ndarray):
            for dim_idx, size in enumerate(arg.shape):
                if size > 1:
                    dims_to_reduce.append(dim_idx)
                    break
            continue

        if isinstance(arg, str):
            try:
                dims_to_reduce.append(int(arg.strip("'").strip('"')))
                continue
            except:
                pass

        if isinstance(arg, (tuple, list)):
            continue

    if dims_to_reduce:
        dims_to_reduce = tuple(sorted(set(dims_to_reduce)))
        if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.floating):
            return np.sum(x, axis=dims_to_reduce, keepdims=True, dtype=np.float64)
        return np.sum(x, axis=dims_to_reduce, keepdims=True)

    if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.floating):
        return np.sum(x, keepdims=True, dtype=np.float64)
    return np.sum(x, keepdims=True)


def _impl_tensor_max(x, *args):
    dims = []
    for arg in args:
        if isinstance(arg, (int, np.integer)):
            dims.append(int(arg))
            continue
        if isinstance(arg, np.ndarray):
            if arg.ndim == 0 or arg.size == 1:
                dims.append(int(np.asarray(arg).reshape(-1)[0]))
            continue
        if isinstance(arg, str):
            try:
                dims.append(int(arg.strip("'").strip('"')))
                continue
            except:
                pass
    if dims and isinstance(x, np.ndarray):
        normalized = []
        for dim in dims:
            if x.ndim == 0:
                continue
            if dim >= x.ndim:
                dim = x.ndim - 1
            elif x.ndim > 1 and 0 <= dim < x.ndim - 1 and x.shape[dim] == 1:
                trailing_non_singleton = [idx for idx in range(dim + 1, x.ndim) if x.shape[idx] > 1]
                if trailing_non_singleton:
                    dim = trailing_non_singleton[-1]
            normalized.append(dim)
        dims = tuple(sorted(set(normalized)))
    return np.max(x, axis=dims, keepdims=True) if dims else np.max(x, keepdims=True)


def _impl_tensor_min(x, *args):
    dims = []
    for arg in args:
        if isinstance(arg, (int, np.integer)):
            dims.append(int(arg))
            continue
        if isinstance(arg, np.ndarray):
            if arg.ndim == 0 or arg.size == 1:
                dims.append(int(np.asarray(arg).reshape(-1)[0]))
            continue
        if isinstance(arg, str):
            try:
                dims.append(int(arg.strip("'").strip('"')))
                continue
            except:
                pass
    if dims and isinstance(x, np.ndarray):
        normalized = []
        for dim in dims:
            if x.ndim == 0:
                continue
            if dim >= x.ndim:
                dim = x.ndim - 1
            elif x.ndim > 1 and 0 <= dim < x.ndim - 1 and x.shape[dim] == 1:
                trailing_non_singleton = [idx for idx in range(dim + 1, x.ndim) if x.shape[idx] > 1]
                if trailing_non_singleton:
                    dim = trailing_non_singleton[-1]
            normalized.append(dim)
        dims = tuple(sorted(set(normalized)))
    return np.min(x, axis=dims, keepdims=True) if dims else np.min(x, keepdims=True)


def _impl_tensor_prod(x, *args):
    dim = None
    for arg in args:
        if isinstance(arg, (int, np.integer)):
            dim = int(arg)
            break
        if isinstance(arg, str):
            try:
                dim = int(arg.strip("'").strip('"'))
                break
            except:
                pass

    return np.prod(x, axis=dim, keepdims=True)


def _impl_attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    scores = np.matmul(q, k.swapaxes(-2, -1)) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    attn = _impl_softmax(scores, dim=-1)
    return np.matmul(attn, v)


def _impl_reshape(x, *args):
    if len(args) == 1 and isinstance(args[0], (tuple, list)):
        new_shape = args[0]
    else:
        new_shape = args

    clean_shape = []
    for s in new_shape:
        if hasattr(s, 'ndim'):
            if s.ndim == 0:
                clean_shape.append(int(s))
            elif s.size == 1:
                clean_shape.append(int(s.item()))
            else:
                try:
                    val = int(s.flat[0])
                    clean_shape.append(val)
                except:
                    raise ValueError(f"Reshape dimension must be scalar, got array with shape {s.shape}")
        else:
            try:
                clean_shape.append(int(s))
            except:
                clean_shape.append(s)

    return np.reshape(x, tuple(clean_shape))


def _impl_slice_fill(x, start, size, stride, fill_value):
    """TensorRT ISliceLayer FILL semantics for static start/size/stride."""

    value = np.asarray(x)

    def dimensions(raw, name):
        if isinstance(raw, np.ndarray):
            raw = raw.tolist()
        if not isinstance(raw, (list, tuple)):
            raw = (raw,)
        try:
            return tuple(int(item) for item in raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"SliceFill {name} must be an integer sequence") from exc

    starts = dimensions(start, "start")
    sizes = dimensions(size, "size")
    strides = dimensions(stride, "stride")
    if not (len(starts) == len(sizes) == len(strides) == value.ndim):
        raise ValueError(
            "SliceFill start/size/stride ranks must equal the input rank"
        )
    if any(item < 0 for item in sizes):
        raise ValueError("SliceFill sizes must be non-negative")
    if any(item == 0 for item in strides):
        raise ValueError("SliceFill strides must be non-zero")

    fill = np.asarray(fill_value).reshape(-1)
    if fill.size != 1:
        raise ValueError("SliceFill fill_value must be scalar")
    output = np.full(sizes, fill[0], dtype=value.dtype)
    output_indices = []
    input_indices = []
    for axis, (axis_start, axis_size, axis_stride) in enumerate(
        zip(starts, sizes, strides)
    ):
        mapped = axis_start + np.arange(axis_size, dtype=np.int64) * axis_stride
        valid = (mapped >= 0) & (mapped < value.shape[axis])
        output_indices.append(np.flatnonzero(valid))
        input_indices.append(mapped[valid])
    if all(indices.size for indices in output_indices):
        output[np.ix_(*output_indices)] = value[np.ix_(*input_indices)]
    return output


def _impl_join(*args):
    res = ()
    for arg in args:
        if isinstance(arg, (np.ndarray, np.generic)):
            if arg.ndim == 0:
                val = arg.item() if hasattr(arg, 'item') else arg
                res = res + (val,)
            else:
                try:
                    res = res + tuple(arg.tolist())
                except:
                    res = res + (arg,)

        elif isinstance(arg, (list, tuple)):
            res = res + tuple(arg)

        else:
            res = res + (arg,)
    return res


def _impl_concat(*args):
    if len(args) < 3:
        raise ValueError("Concat requires at least two tensors and one axis")
    *values, axis = args
    axis_array = np.asarray(axis)
    if axis_array.size != 1:
        raise ValueError("Concat axis must be scalar")
    return np.concatenate(
        tuple(np.asarray(value) for value in values),
        axis=int(axis_array.reshape(-1)[0]),
    )


def _impl_pad(x, paddings, constant_values=0):
    value = np.asarray(x)
    pads = np.asarray(paddings).astype(np.int64).reshape(-1, 2)
    if pads.shape[0] != value.ndim:
        raise ValueError(f"Pad rank mismatch: {pads.shape[0]} versus {value.ndim}")
    fill = np.asarray(constant_values).reshape(-1)
    return np.pad(value, tuple((int(a), int(b)) for a, b in pads), mode="constant", constant_values=float(fill[0] if fill.size else 0))


def _impl_pad_shape(shape, paddings):
    dims = np.asarray(shape).reshape(-1).astype(np.int64)
    pads = np.asarray(paddings).reshape(-1, 2).astype(np.int64)
    if len(dims) != len(pads):
        raise ValueError("PadShape rank mismatch")
    return tuple(int(dim + pair[0] + pair[1]) for dim, pair in zip(dims, pads))


def _impl_tile(x, repeat_times):
    repeats = tuple(int(item) for item in np.asarray(repeat_times).reshape(-1))
    return np.tile(np.asarray(x), repeats)


def _impl_expand_dims(x, axis):
    axes = tuple(int(item) for item in np.asarray(axis).reshape(-1))
    return np.expand_dims(np.asarray(x), axis=axes[0] if len(axes) == 1 else axes)


def _impl_gather(params, indices, axis=0):
    return np.take(np.asarray(params), np.asarray(indices).astype(np.int64), axis=int(np.asarray(axis).reshape(-1)[0]))


def _impl_slice_axes(x, axes, starts, ends):
    value = np.asarray(x)
    slices = [slice(None)] * value.ndim
    for axis, start, end in zip(
        np.asarray(axes).reshape(-1),
        np.asarray(starts).reshape(-1),
        np.asarray(ends).reshape(-1),
    ):
        slices[int(axis)] = slice(int(start), int(end))
    return value[tuple(slices)]


def _impl_einsum(*args):
    if len(args) < 2:
        raise ValueError("Einsum requires operands and an equation")
    *operands, equation = args
    equation = str(equation).strip("'\"")
    return np.einsum(equation, *[np.asarray(value) for value in operands])


def _impl_range(limit):
    return np.arange(int(np.asarray(limit).reshape(-1)[0]))


def _impl_python_getitem(value, index_spec, *dynamic_values):
    if isinstance(index_spec, np.ndarray) and index_spec.ndim == 0:
        index_spec = index_spec.item()
    if isinstance(index_spec, dict):
        if index_spec.get("kind") != "python_index":
            raise ValueError(f"invalid structured Python index: {index_spec!r}")

        def resolve_bound(bound):
            if bound is None:
                return None
            if isinstance(bound, dict) and set(bound) == {"input_index"}:
                index = int(bound["input_index"]) - 1
                if index < 0 or index >= len(dynamic_values):
                    raise ValueError(
                        f"dynamic Python index input is unavailable: {index + 1}"
                    )
                array = np.asarray(dynamic_values[index]).reshape(-1)
                if array.size != 1:
                    raise ValueError("dynamic Python index bound must be scalar")
                return int(array[0])
            return int(bound)

        components = []
        for component in index_spec.get("components", ()):
            kind = component.get("kind")
            if kind == "newaxis":
                components.append(None)
            elif kind == "ellipsis":
                components.append(Ellipsis)
            elif kind == "integer":
                components.append(resolve_bound(component.get("value")))
            elif kind == "slice":
                components.append(slice(
                    resolve_bound(component.get("start")),
                    resolve_bound(component.get("stop")),
                    resolve_bound(component.get("step")),
                ))
            else:
                raise ValueError(
                    f"invalid structured Python index component: {component!r}"
                )
        return np.asarray(value)[tuple(components)]
    components = []
    for token in str(index_spec).strip("'\"").split("|"):
        if token == "n":
            components.append(None)
        elif token == "e":
            components.append(Ellipsis)
        elif token.startswith("i:"):
            components.append(int(token[2:]))
        elif token.startswith("s:"):
            parts = token[2:].split(":")
            if len(parts) != 3:
                raise ValueError(f"invalid slice token: {token}")
            components.append(
                slice(*[None if part == "" else int(part) for part in parts])
            )
        else:
            raise ValueError(f"invalid Python index token: {token}")
    return np.asarray(value)[tuple(components)]


def _impl_scatter_nd_update(value, indices, updates):
    output = np.array(value, copy=True)
    index_array = np.asarray(indices, dtype=np.int64)
    if index_array.ndim < 1:
        raise ValueError("ScatterND indices must have rank at least one")
    depth = int(index_array.shape[-1])
    flat_indices = index_array.reshape(-1, depth)
    update_array = np.asarray(updates)
    for offset, coordinate in enumerate(flat_indices):
        output[tuple(int(item) for item in coordinate)] = update_array[offset]
    return output


def _impl_scatter_nd(indices, updates, shape):
    shape_values = tuple(int(item) for item in np.asarray(shape).reshape(-1))
    output = np.zeros(shape_values, dtype=np.asarray(updates).dtype)
    index_array = np.asarray(indices, dtype=np.int64)
    depth = int(index_array.shape[-1])
    flat_indices = index_array.reshape(-1, depth)
    update_array = np.asarray(updates)
    for offset, coordinate in enumerate(flat_indices):
        output[tuple(int(item) for item in coordinate)] += update_array[offset]
    return output


def _impl_scatter(value, indices, updates, overwrite=True):
    output = np.array(value, copy=True)
    index_array = np.asarray(indices, dtype=np.int64).reshape(-1)
    update_array = np.asarray(updates).reshape((len(index_array),) + np.asarray(updates).shape[1:])
    for offset, index in enumerate(index_array):
        if overwrite:
            output[int(index)] = update_array[offset]
        else:
            output[int(index)] += update_array[offset]
    return output


def _impl_repeat(value, repeats, axis=None):
    repeat_values = np.asarray(repeats, dtype=np.int64)
    if repeat_values.ndim == 0:
        repeat_values = int(repeat_values)
    return np.repeat(np.asarray(value), repeat_values, axis=axis)


def _impl_reduce_prod(x, axis=None, keepdim=False):
    parsed_axis = None if axis is None else _parse_axis((axis,))
    return np.prod(np.asarray(x), axis=parsed_axis, keepdims=bool(keepdim))


def _impl_strided_slice(x, begin, end, strides):
    value = np.asarray(x)
    starts = np.asarray(begin).reshape(-1).astype(np.int64)
    stops = np.asarray(end).reshape(-1).astype(np.int64)
    steps = np.asarray(strides).reshape(-1).astype(np.int64)
    if not (len(starts) == len(stops) == len(steps) == value.ndim):
        raise ValueError("StridedSlice rank mismatch")
    slices = tuple(slice(int(a), int(b), int(c)) for a, b, c in zip(starts, stops, steps))
    return value[slices]


def _impl_splitv(x, sizes, axis):
    value = np.asarray(x)
    axis = int(np.asarray(axis).reshape(-1)[0])
    sizes = np.asarray(sizes).reshape(-1).astype(np.int64).tolist()
    if -1 in sizes:
        sizes[sizes.index(-1)] = value.shape[axis] - sum(item for item in sizes if item >= 0)
    offsets = np.cumsum([0] + sizes)
    return tuple(np.take(value, range(int(offsets[i]), int(offsets[i + 1])), axis=axis) for i in range(len(sizes)))


def _impl_resize_nearest(x, size):
    value = np.asarray(x)
    target = np.asarray(size).reshape(-1).astype(np.int64)
    if value.ndim != 4 or target.size < 2:
        raise ValueError("ResizeNearestNeighbor currently requires rank-4 input and two spatial sizes")
    h, w = int(target[0]), int(target[1])
    yi = np.minimum((np.arange(h) * value.shape[1] // h), value.shape[1] - 1)
    xi = np.minimum((np.arange(w) * value.shape[2] // w), value.shape[2] - 1)
    return value[:, yi[:, None], xi[None, :], :]


def _impl_resize2d(
    x,
    scale_factor,
    mode="nearest",
    align_corners=False,
    align_mode=0,
    data_format="channels_last",
):
    value = np.asarray(x)
    factors = np.asarray(scale_factor).reshape(-1)
    if value.ndim != 4 or factors.size != 2:
        raise ValueError("Resize2D requires rank-4 input and two scale factors")
    factors = tuple(float(item) for item in factors)
    if any(item <= 0 for item in factors):
        raise ValueError(f"Resize2D scale factors must be positive: {factors}")
    layout = str(data_format).strip("'\"").lower()
    channels_last = layout in {"channels_last", "nhwc"}
    native = value if channels_last else np.transpose(value, (0, 2, 3, 1))
    input_h, input_w = native.shape[1:3]
    output_h = int(round(input_h * factors[0]))
    output_w = int(round(input_w * factors[1]))
    normalized_mode = str(mode).strip("'\"").lower()

    def coordinates(output_size, input_size):
        if bool(align_corners) and output_size > 1:
            result = np.arange(output_size, dtype=np.float64) * (
                (input_size - 1) / (output_size - 1)
            )
        elif int(align_mode) == 0:
            result = (
                (np.arange(output_size, dtype=np.float64) + 0.5)
                * input_size
                / output_size
                - 0.5
            )
        else:
            result = (
                np.arange(output_size, dtype=np.float64)
                * input_size
                / output_size
            )
        return np.clip(result, 0.0, max(input_size - 1, 0))

    if normalized_mode == "nearest":
        yi = np.minimum(
            np.floor(
                np.arange(output_h, dtype=np.float64) * input_h / output_h
            ).astype(np.int64),
            input_h - 1,
        )
        xi = np.minimum(
            np.floor(
                np.arange(output_w, dtype=np.float64) * input_w / output_w
            ).astype(np.int64),
            input_w - 1,
        )
        output = native[:, yi[:, None], xi[None, :], :]
    elif normalized_mode == "bilinear":
        y = coordinates(output_h, input_h)
        x_coord = coordinates(output_w, input_w)
        y0 = np.floor(y).astype(np.int64)
        x0 = np.floor(x_coord).astype(np.int64)
        y1 = np.minimum(y0 + 1, input_h - 1)
        x1 = np.minimum(x0 + 1, input_w - 1)
        wy = (y - y0).astype(np.float64)
        wx = (x_coord - x0).astype(np.float64)
        top_left = native[:, y0[:, None], x0[None, :], :]
        top_right = native[:, y0[:, None], x1[None, :], :]
        bottom_left = native[:, y1[:, None], x0[None, :], :]
        bottom_right = native[:, y1[:, None], x1[None, :], :]
        output = (
            top_left * (1.0 - wy)[None, :, None, None]
            * (1.0 - wx)[None, None, :, None]
            + top_right * (1.0 - wy)[None, :, None, None]
            * wx[None, None, :, None]
            + bottom_left * wy[None, :, None, None]
            * (1.0 - wx)[None, None, :, None]
            + bottom_right * wy[None, :, None, None]
            * wx[None, None, :, None]
        ).astype(native.dtype, copy=False)
    else:
        raise ValueError(f"Resize2D mode is not qualified: {normalized_mode}")
    return output if channels_last else np.transpose(output, (0, 3, 1, 2))


def _impl_cast(x, dtype):
    """Cast a MathIR value using an explicit, framework-neutral dtype name."""
    if isinstance(dtype, np.ndarray) and dtype.ndim == 0:
        dtype = dtype.item()
    if isinstance(dtype, bytes):
        dtype = dtype.decode("utf-8")

    try:
        target_dtype = np.dtype(dtype)
    except TypeError:
        target_dtype = None
    if target_dtype is not None:
        return np.asarray(x).astype(target_dtype, copy=False)

    dtype_name = getattr(dtype, "name", dtype)
    dtype_name = str(dtype_name).strip("'\"").lower()
    for prefix in ("torch.", "paddle.", "tensorflow.", "tf.", "numpy.", "np."):
        if dtype_name.startswith(prefix):
            dtype_name = dtype_name[len(prefix):]
            break

    aliases = {
        "bool_": "bool",
        "half": "float16",
        "float": "float32",
        "double": "float64",
        "short": "int16",
        "int": "int32",
        "long": "int64",
    }
    dtype_name = aliases.get(dtype_name, dtype_name)
    try:
        target_dtype = np.dtype(dtype_name)
    except TypeError as exc:
        raise ValueError(f"Unsupported MathIR cast dtype: {dtype!r}") from exc
    return np.asarray(x).astype(target_dtype, copy=False)


def _impl_embedding_lookup(weight, indices, padding_idx=-1):
    weight_arr = np.asarray(weight)
    index_arr = np.asarray(indices, dtype=np.int64)
    if np.any(index_arr < 0) or np.any(index_arr >= weight_arr.shape[0]):
        raise IndexError("EmbeddingLookup index is outside the weight vocabulary")

    result = weight_arr[index_arr]
    padding = int(np.asarray(padding_idx).reshape(-1)[0])
    if padding >= 0:
        result = np.where(
            (index_arr == padding)[..., np.newaxis],
            np.zeros((), dtype=result.dtype),
            result,
        )
    return result


class NumpyBackend:
    @staticmethod
    def get_ops(context=None):
        ops = {
            'logical_and': _smart_and,
            'logical_or': _smart_or,

            'And': _impl_and,
            'Or': _impl_or,

            'GetItem': _impl_hybrid_getitem,

            'Add': np.add, 'Sub': np.subtract, 'Mul': np.multiply,
            'Div': np.divide, 'Pow': np.power,
            'Mod': lambda x, y: np.mod(x, y).astype(np.int64),
            'floor': lambda x: np.floor(np.round(x * 1e5) / 1e5).astype(np.int64),
            'ceiling': lambda x: np.ceil(np.round(x * 1e5) / 1e5).astype(np.int64),
            'Abs': np.abs, 'Sqrt': np.sqrt, 'Exp': np.exp, 'Log': np.log, 'Round': np.round,

            'erf': sp.erf,
            'Erf': sp.erf,

            'Eq': np.equal, 'Ne': np.not_equal,
            'Gt': np.greater, 'Lt': np.less,
            'Ge': np.greater_equal, 'Le': np.less_equal,
            'Not': np.logical_not,
            'Piecewise': _impl_piecewise,

            'Min': _impl_min,
            'Max': _impl_max,
            'ArgMin': _impl_argmin,
            'ArgMax': _impl_argmax,
            'Dim': _impl_dim,
            'Rank': _impl_rank,
            'Shape': _impl_shape,
            'Range': _impl_range,
            'PythonGetItem': _impl_python_getitem,
            'ScatterNDUpdate': _impl_scatter_nd_update,
            'ScatterND': _impl_scatter_nd,
            'Scatter': _impl_scatter,
            'Repeat': _impl_repeat,
            'OnesLike': lambda x, dtype=None: np.ones_like(np.asarray(x), dtype=dtype),
            'ZerosLike': lambda x: np.zeros_like(np.asarray(x)),
            'Expand': lambda x, shape: np.broadcast_to(np.asarray(x), tuple(int(item) for item in np.asarray(shape).reshape(-1))),
            'BroadcastTo': lambda x, shape: np.broadcast_to(np.asarray(x), tuple(int(item) for item in np.asarray(shape).reshape(-1))),
            'TakeAlongAxis': lambda x, indices, axis: np.take_along_axis(np.asarray(x), np.asarray(indices, dtype=np.int64), axis=int(axis)),
            'Tuple': lambda *args: tuple(args),
            'Reshape': _impl_reshape,
            'SliceFill': _impl_slice_fill,
            'Pad': _impl_pad,
            'PadShape': _impl_pad_shape,
            'Tile': _impl_tile,
            'StridedSlice': _impl_strided_slice,
            'ExpandDims': _impl_expand_dims,
            'Gather': _impl_gather,
            'Slice': _impl_slice_axes,
            'Einsum': _impl_einsum,
            'SplitV': _impl_splitv,
            'ResizeNearestNeighbor': _impl_resize_nearest,
            'Resize2D': _impl_resize2d,
            'Concat': _impl_concat,
            'Transpose': _impl_transpose,
            'make_swap_perm': _impl_make_swap_perm,
            'AxisBroadcastShape': _impl_axis_broadcast_shape,
            'Squeeze': _impl_squeeze,
            'ReduceMean': _impl_reduce_mean,
            'ReduceMax': _impl_reduce_max,
            'ReduceProd': _impl_reduce_prod,
            'Mean': lambda x, *args: np.mean(x, axis=_parse_axis(args), keepdims=True),
            'Var': lambda x, *args: np.var(x, axis=_parse_axis(args), keepdims=True),
            'join': _impl_join,
            'is_list': lambda x: isinstance(x, (list, tuple, np.ndarray)),

            'Sum': _impl_tensor_sum,
            'Prod': _impl_tensor_prod,
            'TensorSum': _impl_tensor_sum,
            'TensorProd': _impl_tensor_prod,
            'TensorMin': _impl_tensor_min,
            'TensorMax': _impl_tensor_max,

            # Ops
            'MatMul': np.matmul,
            'EmbeddingLookup': _impl_embedding_lookup,
            'Softmax': _impl_softmax,
            'Gelu': _impl_gelu,
            'Cast': _impl_cast,
            'Minimum': np.minimum,
            'Maximum': np.maximum,

            # Bitwise
            'BitwiseAnd': _impl_bitwise_and, 'BitwiseOr': _impl_bitwise_or,
            'BitwiseXor': _impl_bitwise_xor, 'BitwiseNot': np.bitwise_not,
            'BitwiseLeftShift': np.left_shift, 'BitwiseRightShift': np.right_shift,

            'Equality': np.equal,
            'GreaterThan': np.greater,
            'LessThan': np.less,
            'GreaterEqual': np.greater_equal,
            'LessEqual': np.less_equal,
        }

        out_rank = context.get('out_rank') if context else None

        if out_rank is not None:
            def make_smart_reduce(np_func, default_impl):
                def wrapper(x, *args):
                    x_arr = np.asarray(x)
                    if args:
                        return default_impl(x_arr, *args)
                    dim = _parse_axis(args)
                    if dim is not None:
                        return default_impl(x_arr, *args)
                    if x_arr.ndim > out_rank:
                        auto_axes = tuple(range(out_rank, x_arr.ndim))
                        return np_func(x_arr, axis=auto_axes, keepdims=True)

                    return default_impl(x_arr, *args)

                return wrapper

            ops['Sum'] = make_smart_reduce(np.sum, _impl_tensor_sum)
            ops['TensorSum'] = make_smart_reduce(np.sum, _impl_tensor_sum)
            ops['TensorMax'] = make_smart_reduce(np.max, _impl_tensor_max)
            ops['TensorMin'] = make_smart_reduce(np.min, _impl_tensor_min)
            ops['Prod'] = make_smart_reduce(np.prod, _impl_tensor_prod)
            ops['TensorProd'] = make_smart_reduce(np.prod, _impl_tensor_prod)

        return ops
