"""Direct identity checks for the current compiler-conversion protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping


def files_are_byte_identical(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    if not left.is_file() or not right.is_file():
        return False
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_handle, right.open("rb") as right_handle:
        while True:
            left_block = left_handle.read(1024 * 1024)
            right_block = right_handle.read(1024 * 1024)
            if left_block != right_block:
                return False
            if not left_block:
                return True


def graph_record(value: Any) -> Dict[str, Any]:
    graph_module = _graph_module(value)
    if graph_module is None:
        raise TypeError(f"value has no FX graph: {type(value).__name__}")
    graph_nodes = tuple(graph_module.graph.nodes)
    node_indices = {node: index for index, node in enumerate(graph_nodes)}
    nodes = []
    for node in graph_nodes:
        nodes.append(
            {
                "op": node.op,
                "target": str(node.target),
                "args": _canonical_graph_value(node.args, node_indices),
                "kwargs": _canonical_graph_value(node.kwargs, node_indices),
            }
        )
    return {
        "structural_schema_version": 2,
        "node_count": len(nodes),
        "nodes": nodes,
    }


def graphs_are_structurally_identical(left: Any, right: Any) -> bool:
    return graph_record(left) == graph_record(right)


def _canonical_graph_value(value: Any, node_indices: Mapping[Any, int]) -> Any:
    """Encode FX arguments by topology, independent of generated node names."""
    try:
        from torch.fx import Node
    except ImportError:  # pragma: no cover - graph_record already requires torch
        Node = ()

    if isinstance(value, Node):
        return {"node_index": node_indices[value]}
    if isinstance(value, tuple):
        return {"tuple": [_canonical_graph_value(item, node_indices) for item in value]}
    if isinstance(value, list):
        return {"list": [_canonical_graph_value(item, node_indices) for item in value]}
    if isinstance(value, dict):
        return {
            "dict": [
                {
                    "key": _canonical_graph_value(key, node_indices),
                    "value": _canonical_graph_value(item, node_indices),
                }
                for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
            ]
        }
    if isinstance(value, slice):
        return {
            "slice": [
                _canonical_graph_value(value.start, node_indices),
                _canonical_graph_value(value.stop, node_indices),
                _canonical_graph_value(value.step, node_indices),
            ]
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    try:
        import torch

        if torch.is_tensor(value):
            tensor = value.detach().cpu()
            return {
                "tensor": {
                    "dtype": str(tensor.dtype),
                    "shape": list(tensor.shape),
                    "values": tensor.tolist(),
                }
            }
    except ImportError:  # pragma: no cover - graph_record already requires torch
        pass
    return {"type": type(value).__name__, "value": str(value)}


def persist_model_state_reference(model: Any, path: Path) -> Dict[str, Any]:
    import torch

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    torch.save(state, str(path))
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "tensor_count": len(state),
        "tensors": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in state.items()
        },
    }


def model_matches_state_reference(model: Any, record: Mapping[str, Any]) -> bool:
    import torch

    path = Path(str(record.get("path") or "")).resolve()
    if not path.is_file() or path.stat().st_size != record.get("size_bytes"):
        return False
    expected = torch.load(str(path), map_location="cpu", weights_only=True)
    actual = model.state_dict()
    if tuple(expected) != tuple(actual):
        return False
    return all(
        torch.equal(expected[name], actual[name].detach().cpu())
        for name in expected
    )


def _graph_module(value: Any) -> Any:
    if value is None:
        return None
    if type(value).__name__ == "ExportedProgram":
        return value.graph_module
    graph_module = getattr(value, "graph_module", None)
    if graph_module is not None:
        return graph_module
    if getattr(value, "graph", None) is not None:
        return value
    return None
