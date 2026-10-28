"""Verified rank-5/6 ``aten.unfold`` decomposition for strict TensorRT repair."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .pooling_lowering import (
    CapabilityRegistry,
    LoweringRecord,
    LoweringRejection,
    LoweringReport,
)


@dataclass(frozen=True)
class TensorUnfoldPlan:
    source_operation: str
    source_mathir_key: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    dimension: int
    size: int
    step: int
    window_count: int
    target_mathir_primitives: Tuple[str, ...]
    instantiated_source_formula: str
    instantiated_target_formula: str
    equivalence_preconditions: Tuple[str, ...]


def _node_target(node: Any) -> str:
    return str(getattr(node, "target", ""))


def _tensor_meta(node: Any) -> Any:
    value = getattr(node, "meta", {}).get("val")
    if value is None or not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise LoweringRejection(
            "MISSING_TENSOR_META",
            "localized unfold input or output has no tensor metadata",
            node=str(getattr(node, "name", "")),
        )
    return value


def _static_shape(node: Any) -> Tuple[int, ...]:
    value = _tensor_meta(node)
    try:
        shape = tuple(int(item) for item in value.shape)
    except Exception as exc:
        raise LoweringRejection(
            "DYNAMIC_INPUT_SHAPE",
            "TensorUnfold lowering requires a static input shape",
        ) from exc
    if any(item <= 0 for item in shape):
        raise LoweringRejection(
            "DYNAMIC_INPUT_SHAPE",
            "TensorUnfold lowering requires positive static dimensions",
            shape=shape,
        )
    return shape


def _integer_arg(node: Any, position: int, keyword: str) -> int:
    value = node.args[position] if len(node.args) > position else node.kwargs.get(keyword)
    if isinstance(value, bool) or not isinstance(value, int):
        raise LoweringRejection(
            "DYNAMIC_UNFOLD_ATTRIBUTE",
            f"TensorUnfold {keyword} must be a static integer",
            attribute=keyword,
            value=repr(value),
        )
    return int(value)


def _mutable_or_alias_sensitive_users(node: Any) -> List[Dict[str, str]]:
    rejected = []
    for user in node.users:
        target = getattr(user, "target", None)
        schema = getattr(target, "_schema", None)
        mutable = bool(getattr(schema, "is_mutable", False))
        text = str(target)
        alias_sensitive = any(
            token in text
            for token in (
                "copy_",
                "index_put_",
                "scatter_",
                "masked_fill_",
                "as_strided_",
                "detach_",
                "set_",
            )
        ) or (user.op == "call_method" and str(target).endswith("_"))
        if mutable or alias_sensitive:
            rejected.append(
                {
                    "node": str(user.name),
                    "target": text,
                    "reason": "mutable_schema" if mutable else "alias_sensitive_target",
                }
            )
    return rejected


def _require_registry_domain(
    registry: CapabilityRegistry,
    key: str,
    field: str,
    value: Any,
) -> None:
    registry.require_domain_value(key, field, value)


def plan_tensor_unfold(
    node: Any,
    capability_registry: CapabilityRegistry,
) -> TensorUnfoldPlan:
    if _node_target(node) != "aten.unfold.default":
        raise LoweringRejection(
            "NOT_TENSOR_UNFOLD",
            "node is not the exact aten.unfold.default overload",
            target=_node_target(node),
        )
    if not node.args:
        raise LoweringRejection("MISSING_UNFOLD_INPUT", "aten.unfold has no tensor input")
    input_node = node.args[0]
    input_shape = _static_shape(input_node)
    input_meta = _tensor_meta(input_node)
    if str(input_meta.dtype) != "torch.float32":
        raise LoweringRejection(
            "DTYPE_OUTSIDE_VERIFIED_DOMAIN",
            "TensorUnfold lowering requires torch.float32",
            dtype=str(input_meta.dtype),
        )
    input_rank = len(input_shape)
    if input_rank not in (5, 6):
        raise LoweringRejection(
            "RANK_OUTSIDE_VERIFIED_DOMAIN",
            "TensorUnfold lowering is qualified only for rank 5 or 6",
            rank=input_rank,
        )

    raw_dimension = _integer_arg(node, 1, "dimension")
    dimension = raw_dimension + len(input_shape) if raw_dimension < 0 else raw_dimension
    if dimension not in (2, 3):
        raise LoweringRejection(
            "DIMENSION_OUTSIDE_VERIFIED_DOMAIN",
            "TensorUnfold lowering is qualified only for normalized dimensions 2 and 3",
            raw_dimension=raw_dimension,
            normalized_dimension=dimension,
        )
    size = _integer_arg(node, 2, "size")
    step = _integer_arg(node, 3, "step")
    if size <= 0 or step <= 0:
        raise LoweringRejection(
            "INVALID_UNFOLD_ATTRIBUTE",
            "TensorUnfold size and step must be positive",
            size=size,
            step=step,
        )
    if size > input_shape[dimension]:
        raise LoweringRejection(
            "UNFOLD_WINDOW_EXCEEDS_INPUT",
            "TensorUnfold window does not fit the selected input dimension",
            size=size,
            input_extent=input_shape[dimension],
        )

    mutable_users = _mutable_or_alias_sensitive_users(node)
    if mutable_users:
        raise LoweringRejection(
            "ALIAS_SENSITIVE_CONSUMER",
            "TensorUnfold output has a mutable or alias-sensitive consumer",
            consumers=mutable_users,
        )

    window_count = (input_shape[dimension] - size) // step + 1
    expected_output = list(input_shape)
    expected_output[dimension] = window_count
    expected_output.append(size)
    observed_output = _static_shape(node)
    if tuple(expected_output) != observed_output:
        raise LoweringRejection(
            "OUTPUT_SHAPE_MISMATCH",
            "TensorUnfold metadata does not match the mathematical output shape",
            expected=expected_output,
            observed=observed_output,
        )

    output_rank = input_rank + 1
    _require_registry_domain(
        capability_registry, "TensorRTSlice", "rank", input_rank
    )
    _require_registry_domain(
        capability_registry, "TensorRTSlice", "slice_mode", "STRICT_BOUNDS"
    )
    _require_registry_domain(
        capability_registry, "TensorRTUnsqueeze", "input_rank", input_rank
    )
    _require_registry_domain(
        capability_registry, "TensorRTUnsqueeze", "output_rank", output_rank
    )
    primitives: List[str] = []
    for _ in range(size):
        primitives.extend(("TensorRTSlice", "TensorRTUnsqueeze"))
    if size > 1:
        _require_registry_domain(
            capability_registry, "TensorRTConcatenation", "input_count", 2
        )
        _require_registry_domain(
            capability_registry, "TensorRTConcatenation", "input_rank", output_rank
        )
        _require_registry_domain(
            capability_registry, "TensorRTConcatenation", "axis", input_rank
        )
        primitives.extend("TensorRTConcatenation" for _ in range(size - 1))

    return TensorUnfoldPlan(
        source_operation="aten.unfold.default",
        source_mathir_key="TensorUnfold",
        input_shape=input_shape,
        output_shape=observed_output,
        dimension=dimension,
        size=size,
        step=step,
        window_count=window_count,
        target_mathir_primitives=tuple(primitives),
        instantiated_source_formula=(
            f"Y[..., window_index, ..., offset] = X[..., "
            f"window_index * {step} + offset, ...], dimension={dimension}, "
            f"size={size}"
        ),
        instantiated_target_formula=(
            f"for offset in [0,{size}): Slice(axis={dimension}, start=offset, "
            f"size={window_count}, stride={step}) -> Unsqueeze(axis={input_rank}); "
            f"ordered binary Concatenation(axis={input_rank})"
        ),
        equivalence_preconditions=(
            "exact aten.unfold.default overload",
            "rank-5 or rank-6 static FP32 input",
            "normalized dimension is 2 or 3",
            "positive size and step with the window inside the input",
            "reconstructed output shape equals exported node metadata",
            "no mutable or alias-sensitive consumer",
            "Slice, Unsqueeze, and binary Concatenation lie in the verified TensorRT registry domain",
        ),
    )


def _infer_meta(target: Any, args: Sequence[Any]) -> Any:
    values = []
    for arg in args:
        if hasattr(arg, "meta") and "val" in arg.meta:
            values.append(arg.meta["val"])
        elif isinstance(arg, (list, tuple)):
            values.append(
                type(arg)(
                    item.meta["val"] if hasattr(item, "meta") and "val" in item.meta else item
                    for item in arg
                )
            )
        else:
            values.append(arg)
    try:
        return target(*values)
    except Exception:
        return None


def _call_function(
    graph: Any,
    target: Any,
    args: Tuple[Any, ...],
    source_meta: Mapping[str, Any],
) -> Any:
    node = graph.call_function(target, args=args)
    for key in ("nn_module_stack", "source_fn_stack", "stack_trace"):
        if key in source_meta:
            node.meta[key] = source_meta[key]
    inferred = _infer_meta(target, args)
    if inferred is not None:
        node.meta["val"] = inferred
    return node


def _rewrite_tensor_unfold(node: Any, plan: TensorUnfoldPlan) -> List[str]:
    import torch

    graph = node.graph
    input_node = node.args[0]
    replacement_nodes: List[str] = []
    parts = []
    with graph.inserting_before(node):
        for offset in range(plan.size):
            stop = offset + plan.window_count * plan.step
            sliced = _call_function(
                graph,
                torch.ops.aten.slice.Tensor,
                (input_node, plan.dimension, offset, stop, plan.step),
                node.meta,
            )
            replacement_nodes.append(sliced.name)
            unsqueezed = _call_function(
                graph,
                torch.ops.aten.unsqueeze.default,
                (sliced, len(plan.input_shape)),
                node.meta,
            )
            replacement_nodes.append(unsqueezed.name)
            parts.append(unsqueezed)

        replacement = parts[0]
        for part in parts[1:]:
            replacement = _call_function(
                graph,
                torch.ops.aten.cat.default,
                ([replacement, part], len(plan.input_shape)),
                node.meta,
            )
            replacement_nodes.append(replacement.name)

    replacement.meta.update(node.meta)
    node.replace_all_uses_with(replacement)
    graph.erase_node(node)
    return replacement_nodes


def apply_tensor_unfold_lowering(
    graph_module: Any,
    capability_registry: CapabilityRegistry,
    *,
    allowed_nodes: Optional[Iterable[str]] = None,
) -> LoweringReport:
    graph = graph_module.graph
    before = str(graph)
    records: List[LoweringRecord] = []
    allowed = None if allowed_nodes is None else {str(item) for item in allowed_nodes}
    for node in list(graph.nodes):
        if _node_target(node) != "aten.unfold.default":
            continue
        if allowed is not None and node.name not in allowed:
            continue
        try:
            plan = plan_tensor_unfold(node, capability_registry)
            replacement_nodes = _rewrite_tensor_unfold(node, plan)
            records.append(
                LoweringRecord(
                    node=node.name,
                    source_operation="aten.unfold.default",
                    status="APPLIED",
                    plan=asdict(plan),
                    replacement_nodes=replacement_nodes,
                )
            )
        except LoweringRejection as exc:
            records.append(
                LoweringRecord(
                    node=node.name,
                    source_operation="aten.unfold.default",
                    status="REJECTED",
                    rejection=exc.to_dict(),
                )
            )
    graph.lint()
    graph_module.recompile()
    return LoweringReport(records=records, graph_before=before, graph_after=str(graph))


__all__ = [
    "TensorUnfoldPlan",
    "apply_tensor_unfold_lowering",
    "plan_tensor_unfold",
]
