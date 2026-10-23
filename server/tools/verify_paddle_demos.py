"""Re-run the real web-task conversion/assessment for bundled Paddle cases.

    python server/tools/verify_paddle_demos.py
    python server/tools/verify_paddle_demos.py --models densenet inception
    python server/tools/verify_paddle_demos.py --build-presets

Success requires real ModelX conversion (no replay), syntax-valid main/helper
modules, no failed automatic checks, and an actual Paddle forward of the
configured top-level model with finite output of the expected shape. This is
an executability regression check, not a dataset accuracy measurement.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from server import app as backend
from server.paddle_converter import paddle_demo_cases


def main() -> None:
    cases = paddle_demo_cases()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(cases), default=list(cases))
    parser.add_argument("--build-presets", action="store_true")
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts" / "paddle-demo-validation.json")
    args = parser.parse_args()
    rows = []
    for name in args.models:
        started = time.perf_counter()
        task_id = backend._task_create("paddle")
        backend._run_paddle(task_id, backend.GenerateRequest(
            kb="paddle", source=f"pytorch/vision/{name}", mode="sample",
        ))
        task = backend.tasks[task_id]
        result = task.get("result") or {}
        evaluation = result.get("evaluation") or {}
        runtime = evaluation.get("paddle_runtime") or {}
        files = result.get("files") or []
        checks = evaluation.get("checks") or {}
        errors = []
        if task["status"] != "completed":
            errors.append(task.get("error") or task["status"])
        if result.get("pre_generated"):
            errors.append("Pre-generated fallback is not accepted as a live conversion test")
        if not checks or not all(checks.values()):
            errors.append("Failed automatic checks: " + ", ".join(k for k, ok in checks.items() if not ok))
        if not runtime.get("forward_attempted") or not runtime.get("forward_ok"):
            errors.append(runtime.get("error") or runtime.get("skip_reason") or "Forward not executed")
        if runtime.get("class_name") != cases[name]["class_name"]:
            errors.append("Wrong top-level model was tested")
        if runtime.get("output_shape") != cases[name]["expected_output_shape"]:
            errors.append("Unexpected output shape")
        names = {file["name"] for file in files}
        for companion in cases[name]["expected_companions"]:
            if companion not in names:
                errors.append("Missing underlying-library companion: " + companion)
        for file in files:
            try:
                ast.parse(file["content"], filename=file["name"])
            except SyntaxError as exc:
                errors.append(str(exc))
        row = {
            "model": name, "label": cases[name]["label"],
            "passed": not errors, "errors": errors,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "pre_generated": result.get("pre_generated"),
            "stats": evaluation.get("stats", {}),
            "checks": checks, "scores": evaluation.get("scores", {}),
            "runtime": runtime,
            "library_modifications": evaluation.get("library_modifications", []),
            "compatibility_fixes": evaluation.get("compatibility_fixes", []),
            "files": [{"name": file["name"], "sha256": hashlib.sha256(
                file["content"].encode("utf-8")).hexdigest()} for file in files],
            "source_sha256": hashlib.sha256((backend.OPALIGN_MODELS / "pytorch" / "vision" /
                "sourceModels" / f"{name}.py").read_bytes()).hexdigest(),
        }
        rows.append(row)
        if args.build_presets and row["passed"]:
            for file in files:
                if file["name"] == f"{name}_paddle.py":
                    target = backend.PADDLE_PRESET_DIR / f"{name}_to_paddle.py"
                else:
                    target = backend.PADDLE_PRESET_DIR / name / file["name"]
                target.parent.mkdir(parents=True, exist_ok=True)
                # Keep generated repository presets LF-only and free of
                # astor's soft-wrap trailing whitespace on Windows.
                preset_text = "\n".join(line.rstrip() for line in file["content"].splitlines()).rstrip() + "\n"
                target.write_bytes(preset_text.encode("utf-8"))
        print(json.dumps({"model": name, "passed": row["passed"], "seconds": row["elapsed_s"],
                          "class_name": runtime.get("class_name"), "output": runtime.get("output_shape"),
                          "companions": cases[name]["expected_companions"], "errors": errors}, ensure_ascii=True), flush=True)

    paddle_version = None
    for distribution in ("paddlepaddle", "paddlepaddle-gpu"):
        try:
            paddle_version = importlib.metadata.version(distribution)
            break
        except importlib.metadata.PackageNotFoundError:
            continue
    report = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "paddle": paddle_version,
        "scope": "Real ModelX conversion + main/helper syntax + CPU dummy forward; no numerical equivalence claim",
        "passed": sum(row["passed"] for row in rows), "total": len(rows), "cases": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_bytes((json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"passed": report["passed"], "total": report["total"],
                      "report": str(args.report)}, ensure_ascii=True), flush=True)
    raise SystemExit(0 if report["passed"] == report["total"] else 1)


if __name__ == "__main__":
    main()
