"""Formula-driven pooling plans and FX/ATen rewrites for the capability gate."""

from __future__ import annotations

import json
import operator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


Pair = Tuple[int, int]


class LoweringRejection(ValueError):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details = details

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "details": self.details,
        }


@dataclass(frozen=True)
class Pool2DPlan:
    source_operation: str
    source_mathir_key: str
    input_shape: Tuple[int, ...]
    kernel_size: Pair
    stride: Pair
    padding: Pair
    dilation: Pair
    output_size: Pair
    extra_padding_after: Pair
    target_mathir_primitives: Tuple[str, ...]
    instantiated_source_formula: str
    instantiated_target_formula: str
    equivalence_preconditions: Tuple[str, ...]
    count_include_pad: Optional[bool] = None
    divisor_override: Optional[int] = None


@dataclass(frozen=True)
class AdaptiveMaxPlan:
    source_operation: str
    source_mathir_key: str
    input_shape: Tuple[int, ...]
    spatial_axes: Tuple[int, ...]
    output_size: Tuple[int, ...]
    target_mathir_primitives: Tuple[str, ...]
    instantiated_source_formula: str
    instantiated_target_formula: str
    equivalence_preconditions: Tuple[str, ...]


@dataclass
class LoweringRecord:
    node: str
    source_operation: str
    status: str
    plan: Optional[Dict[str, Any]] = None
    replacement_nodes: List[str] = field(default_factory=list)
    rejection: Optional[Dict[str, Any]] = None


@dataclass
class LoweringReport:
    records: List[LoweringRecord]
    graph_before: str
    graph_after: str

    @property
    def applied(self) -> List[LoweringRecord]:
        return [record for record in self.records if record.status == "APPLIED"]

    @property
    def rejected(self) -> List[LoweringRecord]:
        return [record for record in self.records if record.status == "REJECTED"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "records": [asdict(record) for record in self.records],
            "graph_before": self.graph_before,
            "graph_after": self.graph_after,
            "applied_count": len(self.applied),
            "rejected_count": len(self.rejected),
        }


class CapabilityRegistry:
    def __init__(self, entries: Sequence[Mapping[str, Any]]) -> None:
        self.entries = {
            str(entry.get("mathir_key")): dict(entry) for entry in entries
        }

    @classmethod
    def load(cls, path: Path) -> "CapabilityRegistry":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def require(self, key: str) -> Mapping[str, Any]:
        entry = self.entries.get(key)
        if entry is None:
            raise LoweringRejection(
                "CAPABILITY_MISSING", f"capability registry has no {key}", key=key
            )
        if not entry.get("eligible_for_lowering"):
            raise LoweringRejection(
                "CAPABILITY_INELIGIBLE",
                f"{key} is not eligible for lowering",
                key=key,
                native_test_status=entry.get("native_test_status"),
                parser_roundtrip_status=entry.get("parser_roundtrip_status"),
            )
        return entry

    def require_domain_value(self, key: str, field_name: str, value: Any) -> None:
        entry = self.require(key)
        domain = dict(entry.get("verified_domain") or {})
        allowed = domain.get(field_name)
        if allowed is None:
            raise LoweringRejection(
                "CAPABILITY_DOMAIN_UNRECORDED",
                f"{key} has no recorded {field_name} domain",
                key=key,
                field=field_name,
                value=value,
            )
        comparable = list(value) if isinstance(value, tuple) else value
        if isinstance(allowed, list) and comparable not in allowed:
            raise LoweringRejection(
                "CAPABILITY_OUTSIDE_VERIFIED_DOMAIN",
                f"{key} {field_name}={value!r} is outside the verified domain",
                key=key,
                field=field_name,
                value=comparable,
                allowed=allowed,
            )


def _pair(value: Any, *, default: Optional[Any] = None) -> Pair:
    if value in (None, [], ()):
        value = default
    if isinstance(value, int):
        return int(value), int(value)
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return int(value[0]), int(value[0])
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise LoweringRejection(
        "INVALID_PAIR_ATTRIBUTE", f"expected scalar or pair, received {value!r}"
    )


def pool_output_size(
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int = 1,
    *,
    ceil_mode: bool,
) -> int:
    values = (input_size, kernel_size, stride, padding, dilation)
    if any(not isinstance(value, int) for value in values):
        raise LoweringRejection(
            "DYNAMIC_POOL_ATTRIBUTE", "pooling qualification requires static integers"
        )
    if input_size <= 0 or kernel_size <= 0 or stride <= 0 or dilation <= 0:
        raise LoweringRejection(
            "INVALID_POOL_ATTRIBUTE", "input, kernel, stride, and dilation must be positive"
        )
    if padding < 0:
        raise LoweringRejection("INVALID_POOL_ATTRIBUTE", "padding must be non-negative")
    effective_kernel = dilation * (kernel_size - 1) + 1
    ceil_adjustment = stride - 1 if ceil_mode else 0
    output = (
        input_size + 2 * padding - effective_kernel + ceil_adjustment
    ) // stride + 1
    if ceil_mode and output > 0 and (output - 1) * stride >= input_size + padding:
        output -= 1
    if output <= 0:
        raise LoweringRejection(
            "EMPTY_POOL_OUTPUT",
            "pooling parameters produce a non-positive output size",
            input_size=input_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
    return output


def ceil_mode_extra_padding(
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int = 1,
) -> Tuple[int, int]:
    output = pool_output_size(
        input_size,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode=True,
    )
    effective_kernel = dilation * (kernel_size - 1) + 1
    extra = max(
        0,
        (output - 1) * stride
        + effective_kernel
        - (input_size + 2 * padding),
    )
    floor_output = pool_output_size(
        input_size + extra,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode=False,
    )
    if floor_output != output:
        raise AssertionError(
            f"explicit padding derivation failed: ceil={output}, floor={floor_output}"
        )
    return output, extra


def max_pool2d_ceil_decomposition(
    value: Any,
    kernel_size: Any,
    stride: Any = None,
    padding: Any = 0,
    dilation: Any = 1,
) -> Any:
    import torch

    kernel = _pair(kernel_size)
    strides = _pair(stride, default=kernel)
    pads = _pair(padding)
    dilations = _pair(dilation)
    if value.ndim != 4:
        raise LoweringRejection("MAX_POOL_RANK", "max-pool lowering requires rank-4 NCHW")
    _, extra_h = ceil_mode_extra_padding(
        int(value.shape[-2]), kernel[0], strides[0], pads[0], dilations[0]
    )
    _, extra_w = ceil_mode_extra_padding(
        int(value.shape[-1]), kernel[1], strides[1], pads[1], dilations[1]
    )
    padded = value
    if extra_h or extra_w:
        padded = torch.ops.aten.constant_pad_nd.default(
            value, [0, extra_w, 0, extra_h], float("-inf")
        )
    return torch.ops.aten.max_pool2d.default(
        padded,
        list(kernel),
        list(strides),
        list(pads),
        list(dilations),
        False,
    )


def avg_pool2d_ceil_decomposition(
    value: Any,
    kernel_size: Any,
    stride: Any = None,
    padding: Any = 0,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
) -> Any:
    import torch

    kernel = _pair(kernel_size)
    strides = _pair(stride, default=kernel)
    pads = _pair(padding)
    if value.ndim != 4:
        raise LoweringRejection("AVG_POOL_RANK", "avg-pool lowering requires rank-4 NCHW")
    _, extra_h = ceil_mode_extra_padding(
        int(value.shape[-2]), kernel[0], strides[0], pads[0]
    )
    _, extra_w = ceil_mode_extra_padding(
        int(value.shape[-1]), kernel[1], strides[1], pads[1]
    )
    if not (extra_h or extra_w):
        return torch.ops.aten.avg_pool2d.default(
            value,
            list(kernel),
            list(strides),
            list(pads),
            False,
            bool(count_include_pad),
            divisor_override,
        )

    explicit_padding = [
        pads[1], pads[1] + extra_w, pads[0], pads[0] + extra_h
    ]
    padded_value = torch.ops.aten.constant_pad_nd.default(
        value, explicit_padding, 0.0
    )
    numerator_average = torch.ops.aten.avg_pool2d.default(
        padded_value,
        list(kernel),
        list(strides),
        [0, 0],
        False,
        True,
    )
    kernel_area = kernel[0] * kernel[1]
    if divisor_override is not None:
        if divisor_override <= 0:
            raise LoweringRejection(
                "INVALID_DIVISOR_OVERRIDE", "divisor_override must be positive"
            )
        return torch.ops.aten.mul.Scalar(
            numerator_average, float(kernel_area) / float(divisor_override)
        )

    mask = torch.ops.aten.mul.Scalar(value, 0.0)
    mask = torch.ops.aten.add.Scalar(mask, 1.0)
    declared_fill = 1.0 if count_include_pad else 0.0
    if pads != (0, 0):
        mask = torch.ops.aten.constant_pad_nd.default(
            mask,
            [pads[1], pads[1], pads[0], pads[0]],
            declared_fill,
        )
    if extra_h or extra_w:
        mask = torch.ops.aten.constant_pad_nd.default(
            mask, [0, extra_w, 0, extra_h], 0.0
        )
    denominator_average = torch.ops.aten.avg_pool2d.default(
        mask,
        list(kernel),
        list(strides),
        [0, 0],
        False,
        True,
    )
    return torch.ops.aten.div.Tensor(numerator_average, denominator_average)


def _node_target(node: Any) -> str:
    return str(getattr(node, "target", ""))


def _static_shape(node: Any) -> Tuple[int, ...]:
    value = getattr(node, "meta", {}).get("val")
    if value is None or not hasattr(value, "shape"):
        raise LoweringRejection(
            "MISSING_TENSOR_META", "localized source node has no tensor shape metadata"
        )
    try:
        return tuple(int(item) for item in value.shape)
    except Exception as exc:
        raise LoweringRejection(
            "DYNAMIC_INPUT_SHAPE", "qualification lowering requires a static input shape"
        ) from exc


def _require_float32(node: Any) -> None:
    value = getattr(node, "meta", {}).get("val")
    dtype = str(getattr(value, "dtype", ""))
    if dtype != "torch.float32":
        raise LoweringRejection(
            "DTYPE_OUTSIDE_VERIFIED_DOMAIN",
            f"qualification lowering requires torch.float32, received {dtype}",
        )


def _domain_pair(
    registry: CapabilityRegistry, key: str, field_name: str, value: Pair
) -> None:
    registry.require_domain_value(key, field_name, list(value))


def plan_adaptive_global_max(
    node: Any, registry: CapabilityRegistry
) -> AdaptiveMaxPlan:
    target = _node_target(node)
    dimensions = {
        "aten.adaptive_max_pool1d.default": 1,
        "aten.adaptive_max_pool2d.default": 2,
        "aten.adaptive_max_pool3d.default": 3,
    }
    if target not in dimensions:
        raise LoweringRejection(
            "NOT_ADAPTIVE_MAX", f"unsupported adaptive-max target {target}"
        )
    spatial_rank = dimensions[target]
    output_size = tuple(int(item) for item in node.args[1])
    if output_size != (1,) * spatial_rank:
        raise LoweringRejection(
            "ADAPTIVE_MAX_NOT_GLOBAL",
            "only output_size=1 is equivalent to whole-axis reductions",
            output_size=output_size,
        )
    input_node = node.args[0]
    shape = _static_shape(input_node)
    _require_float32(input_node)
    if len(shape) != spatial_rank + 2:
        raise LoweringRejection(
            "ADAPTIVE_MAX_RANK",
            "adaptive-max tensor rank does not match the ATen overload",
            shape=shape,
            spatial_rank=spatial_rank,
        )
    reduce_entry = registry.require("TensorRTReduceMax")
    unsqueeze_entry = registry.require("TensorRTUnsqueeze")
    if len(shape) not in reduce_entry.get("verified_domain", {}).get("input_rank", []):
        raise LoweringRejection(
            "CAPABILITY_OUTSIDE_VERIFIED_DOMAIN",
            "reduce input rank is outside the verified domain",
            rank=len(shape),
        )
    reduced_rank = len(shape) - 1
    if reduced_rank not in unsqueeze_entry.get("verified_domain", {}).get("input_rank", []):
        raise LoweringRejection(
            "CAPABILITY_OUTSIDE_VERIFIED_DOMAIN",
            "unsqueeze input rank is outside the verified domain",
            rank=reduced_rank,
        )
    spatial_axes = tuple(range(2, len(shape)))
    primitives = tuple(
        primitive
        for _ in spatial_axes
        for primitive in ("TensorRTReduceMax", "TensorRTUnsqueeze")
    )
    axes_text = ", ".join(str(axis) for axis in spatial_axes)
    return AdaptiveMaxPlan(
        source_operation=target,
        source_mathir_key=f"AdaptiveMaxPool{spatial_rank}d",
        input_shape=shape,
        spatial_axes=spatial_axes,
        output_size=output_size,
        target_mathir_primitives=primitives,
        instantiated_source_formula=(
            f"global max over spatial axes ({axes_text}) with singleton outputs"
        ),
        instantiated_target_formula=(
            " -> ".join(
                f"ReduceMax(axis={axis}, keep_dimensions=False) -> Unsqueeze(axis={axis})"
                for axis in reversed(spatial_axes)
            )
        ),
        equivalence_preconditions=(
            "output_size is one on every spatial axis",
            "indices output has no live consumer",
            "each reduce and unsqueeze lies in the verified FP32 rank/axis domain",
        ),
    )


def _pool2d_args(node: Any, *, average: bool) -> Dict[str, Any]:
    args = list(node.args)
    kernel = _pair(args[1])
    stride = _pair(args[2] if len(args) > 2 else None, default=kernel)
    padding = _pair(args[3] if len(args) > 3 else 0)
    if average:
        return {
            "kernel": kernel,
            "stride": stride,
            "padding": padding,
            "dilation": (1, 1),
            "ceil_mode": bool(args[4]) if len(args) > 4 else False,
            "count_include_pad": bool(args[5]) if len(args) > 5 else True,
            "divisor_override": args[6] if len(args) > 6 else None,
        }
    return {
        "kernel": kernel,
        "stride": stride,
        "padding": padding,
        "dilation": _pair(args[4] if len(args) > 4 else 1),
        "ceil_mode": bool(args[5]) if len(args) > 5 else False,
    }


def plan_max_pool2d_ceil(node: Any, registry: CapabilityRegistry) -> Pool2DPlan:
    if _node_target(node) != "aten.max_pool2d.default":
        raise LoweringRejection("NOT_MAX_POOL2D", "node is not aten.max_pool2d.default")
    attrs = _pool2d_args(node, average=False)
    if not attrs["ceil_mode"]:
        raise LoweringRejection("CEIL_MODE_DISABLED", "source max pool is not a ceil-mode case")
    if attrs["dilation"] != (1, 1):
        raise LoweringRejection(
            "MAX_POOL_DILATION_UNVERIFIED",
            "TensorRTMaxPool registry has no dilated-pooling semantics",
            dilation=attrs["dilation"],
        )
    input_node = node.args[0]
    shape = _static_shape(input_node)
    _require_float32(input_node)
    if len(shape) != 4:
        raise LoweringRejection("MAX_POOL_RANK", "max-pool lowering requires rank-4 NCHW")
    output_h, extra_h = ceil_mode_extra_padding(
        shape[-2], attrs["kernel"][0], attrs["stride"][0], attrs["padding"][0]
    )
    output_w, extra_w = ceil_mode_extra_padding(
        shape[-1], attrs["kernel"][1], attrs["stride"][1], attrs["padding"][1]
    )
    registry.require_domain_value("TensorRTMaxPool", "input_rank", 4)
    _domain_pair(registry, "TensorRTMaxPool", "window_size", attrs["kernel"])
    _domain_pair(registry, "TensorRTMaxPool", "stride", attrs["stride"])
    _domain_pair(registry, "TensorRTMaxPool", "pre_padding", attrs["padding"])
    _domain_pair(registry, "TensorRTMaxPool", "post_padding", attrs["padding"])
    primitives: List[str] = []
    if extra_h or extra_w:
        registry.require_domain_value("TensorRTSliceFill", "rank", 4)
        registry.require_domain_value("TensorRTSliceFill", "fill_value", "-Infinity")
        primitives.append("TensorRTSliceFill")
    primitives.append("TensorRTMaxPool")
    return Pool2DPlan(
        source_operation="aten.max_pool2d.default",
        source_mathir_key="MaxPool2d",
        input_shape=shape,
        kernel_size=attrs["kernel"],
        stride=attrs["stride"],
        padding=attrs["padding"],
        dilation=attrs["dilation"],
        output_size=(output_h, output_w),
        extra_padding_after=(extra_h, extra_w),
        target_mathir_primitives=tuple(primitives),
        instantiated_source_formula=(
            f"ceil MaxPool2d(K={attrs['kernel']}, S={attrs['stride']}, "
            f"P={attrs['padding']}, input_hw={shape[-2:]})"
        ),
        instantiated_target_formula=(
            f"SliceFill(right={extra_w}, bottom={extra_h}, fill=-Infinity) -> "
            f"round-down MaxPool2d(output_hw={(output_h, output_w)})"
        ),
        equivalence_preconditions=(
            "rank-4 static FP32 NCHW",
            "dilation=(1,1)",
            "explicit padding makes floor output shape equal PyTorch ceil output shape",
            "-Infinity padding cannot change a max over real FP32 inputs",
        ),
    )


def plan_avg_pool2d_ceil(node: Any, registry: CapabilityRegistry) -> Pool2DPlan:
    if _node_target(node) != "aten.avg_pool2d.default":
        raise LoweringRejection("NOT_AVG_POOL2D", "node is not aten.avg_pool2d.default")
    attrs = _pool2d_args(node, average=True)
    if not attrs["ceil_mode"]:
        raise LoweringRejection("CEIL_MODE_DISABLED", "source average pool is not a ceil-mode case")
    input_node = node.args[0]
    shape = _static_shape(input_node)
    _require_float32(input_node)
    if len(shape) != 4:
        raise LoweringRejection("AVG_POOL_RANK", "avg-pool lowering requires rank-4 NCHW")
    output_h, extra_h = ceil_mode_extra_padding(
        shape[-2], attrs["kernel"][0], attrs["stride"][0], attrs["padding"][0]
    )
    output_w, extra_w = ceil_mode_extra_padding(
        shape[-1], attrs["kernel"][1], attrs["stride"][1], attrs["padding"][1]
    )
    registry.require_domain_value("TensorRTAveragePool", "input_rank", 4)
    _domain_pair(registry, "TensorRTAveragePool", "window_size", attrs["kernel"])
    _domain_pair(registry, "TensorRTAveragePool", "stride", attrs["stride"])
    _domain_pair(registry, "TensorRTAveragePool", "pre_padding", attrs["padding"])
    _domain_pair(registry, "TensorRTAveragePool", "post_padding", attrs["padding"])
    excludes_padding = not attrs["count_include_pad"]
    registry.require_domain_value(
        "TensorRTAveragePool",
        "average_count_excludes_padding",
        excludes_padding,
    )
    primitives: List[str] = []
    if extra_h or extra_w:
        registry.require_domain_value("TensorRTSliceFill", "rank", 4)
        registry.require_domain_value("TensorRTSliceFill", "fill_value", 0.0)
        primitives.extend(("TensorRTSliceFill", "TensorRTAveragePool"))
    else:
        primitives.append("TensorRTAveragePool")
    return Pool2DPlan(
        source_operation="aten.avg_pool2d.default",
        source_mathir_key="AvgPool2d",
        input_shape=shape,
        kernel_size=attrs["kernel"],
        stride=attrs["stride"],
        padding=attrs["padding"],
        dilation=(1, 1),
        output_size=(output_h, output_w),
        extra_padding_after=(extra_h, extra_w),
        target_mathir_primitives=tuple(primitives),
        instantiated_source_formula=(
            f"ceil AvgPool2d(K={attrs['kernel']}, S={attrs['stride']}, "
            f"P={attrs['padding']}, include_pad={attrs['count_include_pad']}, "
            f"divisor_override={attrs['divisor_override']})"
        ),
        instantiated_target_formula=(
            "round-down TensorRT average pool"
            if not (extra_h or extra_w)
            else "zero SliceFill + numerator/mask round-down pools + elementwise normalization"
        ),
        equivalence_preconditions=(
            "rank-4 static FP32 NCHW",
            "ceil-only extra padding is excluded from the denominator",
            "declared padding follows count_include_pad",
        ),
        count_include_pad=attrs["count_include_pad"],
        divisor_override=attrs["divisor_override"],
    )


def _meta_value(target: Any, args: Tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
    converted = []
    for arg in args:
        if hasattr(arg, "meta") and "val" in arg.meta:
            converted.append(arg.meta["val"])
        else:
            converted.append(arg)
    try:
        return target(*tuple(converted), **dict(kwargs))
    except Exception:
        return None


def _new_node(
    graph: Any,
    target: Any,
    args: Tuple[Any, ...],
    *,
    source_meta: Optional[Mapping[str, Any]] = None,
) -> Any:
    node = graph.call_function(target, args=args)
    if source_meta:
        for key in ("nn_module_stack", "source_fn_stack", "stack_trace"):
            if key in source_meta:
                node.meta[key] = source_meta[key]
    inferred = _meta_value(target, args, {})
    if inferred is not None:
        node.meta["val"] = inferred
    return node


def _adaptive_value_users(node: Any) -> Tuple[List[Any], List[Any]]:
    values: List[Any] = []
    indices: List[Any] = []
    for user in node.users:
        if user.op != "call_function" or user.target is not operator.getitem:
            raise LoweringRejection(
                "ADAPTIVE_MAX_OUTPUT_STRUCTURE",
                "adaptive max output has a non-getitem consumer",
                user=user.name,
            )
        index = int(user.args[1])
        if index == 0:
            values.append(user)
        elif index == 1:
            indices.append(user)
        else:
            raise LoweringRejection(
                "ADAPTIVE_MAX_OUTPUT_STRUCTURE",
                f"unexpected adaptive max tuple index {index}",
            )
    if not values:
        raise LoweringRejection(
            "ADAPTIVE_MAX_VALUES_UNUSED", "adaptive max values output has no consumer"
        )
    live_indices = [user.name for user in indices if user.users]
    if live_indices:
        raise LoweringRejection(
            "ADAPTIVE_MAX_INDICES_LIVE",
            "indices reconstruction is not implemented",
            live_indices=live_indices,
        )
    return values, indices


def _apply_adaptive_global_max(
    graph: Any, node: Any, plan: AdaptiveMaxPlan
) -> List[str]:
    import torch

    value_users, index_users = _adaptive_value_users(node)
    current = node.args[0]
    replacements: List[str] = []
    with graph.inserting_before(node):
        for axis in reversed(plan.spatial_axes):
            reduced = _new_node(
                graph,
                torch.ops.aten.amax.default,
                (current, [axis], False),
                source_meta=node.meta,
            )
            expanded = _new_node(
                graph,
                torch.ops.aten.unsqueeze.default,
                (reduced, axis),
                source_meta=node.meta,
            )
            replacements.extend((reduced.name, expanded.name))
            current = expanded
    for user in value_users:
        if "val" in user.meta:
            current.meta["val"] = user.meta["val"]
        user.replace_all_uses_with(current)
    for user in value_users + index_users:
        if not user.users:
            graph.erase_node(user)
    if node.users:
        raise AssertionError(f"adaptive max node still has users: {list(node.users)}")
    graph.erase_node(node)
    return replacements


def _apply_max_pool2d(graph: Any, node: Any, plan: Pool2DPlan) -> List[str]:
    import torch

    current = node.args[0]
    replacements: List[str] = []
    extra_h, extra_w = plan.extra_padding_after
    with graph.inserting_before(node):
        if extra_h or extra_w:
            current = _new_node(
                graph,
                torch.ops.aten.constant_pad_nd.default,
                (current, [0, extra_w, 0, extra_h], float("-inf")),
                source_meta=node.meta,
            )
            replacements.append(current.name)
        pooled = _new_node(
            graph,
            torch.ops.aten.max_pool2d.default,
            (
                current,
                list(plan.kernel_size),
                list(plan.stride),
                list(plan.padding),
                # Torch-TensorRT 2.7's validator accepts the scalar default
                # ``dilation=1`` but rejects an equivalent rank-2 list.
                # The lowering rule already rejects non-unit dilation, so the
                # scalar preserves the exact semantics while selecting the
                # strict TensorRT converter.
                1 if plan.dilation == (1, 1) else list(plan.dilation),
                False,
            ),
            source_meta=node.meta,
        )
        replacements.append(pooled.name)
    pooled.meta.update(node.meta)
    node.replace_all_uses_with(pooled)
    graph.erase_node(node)
    return replacements


def _apply_avg_pool2d(graph: Any, node: Any, plan: Pool2DPlan) -> List[str]:
    import torch

    if plan.extra_padding_after != (0, 0):
        raise LoweringRejection(
            "AVG_CEIL_MASK_TARGET_BRIDGE_UNVERIFIED",
            "the full numerator/mask formula is tested, but its scalar-mask target "
            "MathIR/parser bridge is not yet eligible",
            extra_padding_after=plan.extra_padding_after,
        )
    if plan.divisor_override is not None:
        raise LoweringRejection(
            "AVG_DIVISOR_OVERRIDE_TARGET_BRIDGE_UNVERIFIED",
            "TensorRT average-pool registry has no divisor_override primitive",
            divisor_override=plan.divisor_override,
        )
    with graph.inserting_before(node):
        pooled = _new_node(
            graph,
            torch.ops.aten.avg_pool2d.default,
            (
                node.args[0],
                list(plan.kernel_size),
                list(plan.stride),
                list(plan.padding),
                False,
                bool(plan.count_include_pad),
                None,
            ),
            source_meta=node.meta,
        )
    pooled.meta.update(node.meta)
    node.replace_all_uses_with(pooled)
    graph.erase_node(node)
    return [pooled.name]


def apply_pooling_lowering(
    graph_module: Any,
    capability_registry: CapabilityRegistry,
    *,
    allowed_nodes: Optional[Iterable[str]] = None,
) -> LoweringReport:
    graph = graph_module.graph
    before = str(graph)
    records: List[LoweringRecord] = []
    allowed = None if allowed_nodes is None else {str(item) for item in allowed_nodes}
    targets = {
        "aten.adaptive_max_pool1d.default",
        "aten.adaptive_max_pool2d.default",
        "aten.adaptive_max_pool3d.default",
        "aten.max_pool2d.default",
        "aten.avg_pool2d.default",
    }
    for node in list(graph.nodes):
        target = _node_target(node)
        if target not in targets:
            continue
        if allowed is not None and node.name not in allowed:
            continue
        try:
            if target.startswith("aten.adaptive_max_pool"):
                plan = plan_adaptive_global_max(node, capability_registry)
                replacement = _apply_adaptive_global_max(graph, node, plan)
            elif target == "aten.max_pool2d.default":
                attrs = _pool2d_args(node, average=False)
                if not attrs["ceil_mode"]:
                    continue
                plan = plan_max_pool2d_ceil(node, capability_registry)
                replacement = _apply_max_pool2d(graph, node, plan)
            else:
                attrs = _pool2d_args(node, average=True)
                if not attrs["ceil_mode"]:
                    continue
                plan = plan_avg_pool2d_ceil(node, capability_registry)
                replacement = _apply_avg_pool2d(graph, node, plan)
            records.append(
                LoweringRecord(
                    node=node.name,
                    source_operation=target,
                    status="APPLIED",
                    plan=asdict(plan),
                    replacement_nodes=replacement,
                )
            )
        except LoweringRejection as exc:
            records.append(
                LoweringRecord(
                    node=node.name,
                    source_operation=target,
                    status="REJECTED",
                    rejection=exc.to_dict(),
                )
            )
    graph.eliminate_dead_code()
    graph.lint()
    graph_module.recompile()
    return LoweringReport(records, before, str(graph))


__all__ = [
    "AdaptiveMaxPlan",
    "CapabilityRegistry",
    "LoweringRecord",
    "LoweringRejection",
    "LoweringReport",
    "Pool2DPlan",
    "apply_pooling_lowering",
    "avg_pool2d_ceil_decomposition",
    "ceil_mode_extra_padding",
    "max_pool2d_ceil_decomposition",
    "plan_adaptive_global_max",
    "plan_avg_pool2d_ceil",
    "plan_max_pool2d_ceil",
    "pool_output_size",
]
