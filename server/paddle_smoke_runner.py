"""Execute exactly the Paddle source bundle delivered by a conversion task.

Runs in a separate process so Paddle failures and loaded helper modules cannot
affect another conversion. This checks executability, not numerical equivalence
to PyTorch or performance on a labelled dataset.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import time
import types
from pathlib import Path


RESULT_PREFIX = "__PADDLE_RESULT__"


def emit(result: dict) -> None:
    print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False), flush=True)


def resolve(module, value):
    """Resolve explicitly configured constructor arguments without eval()."""
    if isinstance(value, dict) and set(value) == {"ref"}:
        obj = module
        for part in value["ref"].split("."):
            obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
        return obj
    if isinstance(value, dict) and "call" in value:
        fn = resolve(module, {"ref": value["call"]})
        args = [resolve(module, arg) for arg in value.get("args", [])]
        kwargs = {k: resolve(module, arg) for k, arg in value.get("kwargs", {}).items()}
        return fn(*args, **kwargs)
    if isinstance(value, list):
        return [resolve(module, item) for item in value]
    if isinstance(value, dict):
        return {key: resolve(module, item) for key, item in value.items()}
    return value


def main() -> None:
    started = time.perf_counter()
    code_path = Path(sys.argv[1]).resolve()
    profile = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    result = {
        "attempted": True, "available": True, "import_ok": False,
        "construction_attempted": False, "construction_ok": False,
        "forward_attempted": False, "forward_ok": False,
        "input_shape": profile.get("input_shape", [1, 3, 224, 224]),
    }
    try:
        # A task-local package preserves ModelX's relative imports of generated
        # modified_*.py modules, including in the user-visible source artifact.
        package_name = "_modelx_task"
        package = types.ModuleType(package_name)
        package.__path__ = [str(code_path.parent)]
        sys.modules[package_name] = package
        sys.path.insert(0, str(code_path.parent))
        spec = importlib.util.spec_from_file_location(package_name + ".model", str(code_path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        import paddle

        paddle.set_device("cpu")
        paddle.seed(20261024)
        result["import_ok"] = True
        if profile.get("class_name"):
            cls = getattr(module, profile["class_name"])
        else:
            # Prefer exported top-level models, never an arbitrary helper Layer.
            names = getattr(module, "__all__", list(vars(module)))
            candidates = [getattr(module, name, None) for name in names]
            candidates = [
                cls for cls in candidates if inspect.isclass(cls)
                and cls.__module__ == module.__name__ and issubclass(cls, paddle.nn.Layer)
            ]
            candidates = [cls for cls in candidates if not any(
                name != "self" and param.default is inspect.Parameter.empty
                and param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
                for name, param in inspect.signature(cls.__init__).parameters.items()
            )]
            if len(candidates) != 1:
                result["skip_reason"] = "需要指定主模型类及构造参数；未执行前向检查"
                result["elapsed_s"] = round(time.perf_counter() - started, 3)
                emit(result)
                return
            cls = candidates[0]

        if not issubclass(cls, paddle.nn.Layer):
            raise TypeError("Configured main class is not paddle.nn.Layer")
        result["class_name"] = cls.__name__
        result["profile_label"] = profile.get("label", cls.__name__)
        result["construction_attempted"] = True
        model = cls(**resolve(module, profile.get("kwargs", {})))
        result["construction_ok"] = True
        model.eval()
        result["params"] = sum(int(p.numel()) for p in model.parameters())
        result["forward_attempted"] = True
        with paddle.no_grad():
            output = model(paddle.rand(result["input_shape"], dtype="float32"))
        if isinstance(output, (list, tuple)):
            output = output[0]
        if not isinstance(output, paddle.Tensor):
            raise TypeError("Model output is not a tensor")
        result["output_shape"] = list(output.shape)
        result["finite"] = bool(paddle.isfinite(output).all().item())
        if not result["finite"]:
            raise ValueError("Model output contains NaN or infinity")
        if profile.get("expected_output_shape"):
            expected = profile["expected_output_shape"]
            result["expected_output_shape"] = expected
            result["output_shape_ok"] = result["output_shape"] == expected
            if not result["output_shape_ok"]:
                raise ValueError(f"Output shape {result['output_shape']} != expected {expected}")
        result["forward_ok"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_s"] = round(time.perf_counter() - started, 3)
    emit(result)


if __name__ == "__main__":
    main()
