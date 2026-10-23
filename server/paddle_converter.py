"""ModelX CMC integration for the PyTorch to Paddle demo.

The original ModelX project is vendored under ``vendor/modelx/CMC``.  This
module keeps the web server independent from ModelX's conversion dependencies:
the converter is imported only when a Paddle conversion is requested, and the
caller receives a useful error when its dependencies are not installed.  The
conversion itself is deterministic and does not call an LLM.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import logging
import os
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional


MODELX_ROOT = Path(__file__).resolve().parent.parent / "vendor" / "modelx"
DEMO_CASES_PATH = Path(__file__).with_name("paddle_demo_cases.json")
_MODELX_LOCK = threading.Lock()


@dataclass
class ConversionResult:
    """A serialisable summary of one ModelX conversion."""

    code: str
    source_file: str
    success_api_count: int = 0
    failed_api_count: int = 0
    unsupported: dict[str, int] = field(default_factory=dict)
    extra_files: dict[str, str] = field(default_factory=dict)
    compatibility_fixes: list[str] = field(default_factory=list)
    converter: str = "ModelX CMC"

    @property
    def api_total(self) -> int:
        return self.success_api_count + self.failed_api_count

    @property
    def api_success_rate(self) -> float:
        return round(self.success_api_count / self.api_total * 100, 1) if self.api_total else 0.0


@contextlib.contextmanager
def _modelx_import_path() -> Iterator[None]:
    """Make the vendored ModelX package importable for the duration of a call."""

    root = str(MODELX_ROOT)
    inserted = root not in sys.path
    if inserted:
        sys.path.insert(0, root)
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(root)
            except ValueError:
                pass


def _primary_source(source: Path) -> Path:
    """Choose the model module when a torchvision model is stored in a folder."""

    if source.is_file():
        return source
    files = [p for p in sorted(source.rglob("*.py")) if p.name != "__init__.py"]
    if not files:
        raise FileNotFoundError(f"No Python source file found in {source}")
    preferred = [
        p for p in files
        if p.stem.lower() in {source.name.lower(), f"{source.name.lower()}v2", "model"}
    ]
    return preferred[0] if preferred else max(files, key=lambda p: p.stat().st_size)


def _normalize_modelx_code(content: str) -> str:
    """Repair compatibility helpers omitted by the bundled torchvision cases."""

    if "self.eager_outputs(" not in content or "def eager_outputs" in content:
        return content
    if "class Inception3" in content:
        helper = (
            "    def eager_outputs(self, x, aux):\n"
            "        if self.training and self.aux_logits:\n"
            "            return InceptionOutputs(x, aux)\n"
            "        return x\n\n"
        )
    elif "class GoogLeNet" in content:
        helper = (
            "    def eager_outputs(self, x, aux2, aux1):\n"
            "        if self.training and self.aux_logits:\n"
            "            return GoogLeNetOutputs(x, aux2, aux1)\n"
            "        return x\n\n"
        )
    else:
        return content
    marker = "    def _transform_input"
    return content.replace(marker, helper + marker, 1)


def paddle_demo_cases() -> dict[str, dict]:
    """Explicit constructors and input contracts for the bundled demo models."""
    return json.loads(DEMO_CASES_PATH.read_text(encoding="utf-8"))


def convert_pytorch_source(source: Path, output_dir: Path) -> ConversionResult:
    """Run ModelX's AST converter and return its generated Python source.

    ModelX's public converter accepts a source file.  For a model package we
    convert the primary module; imports and companion modules remain available
    in the original source tree for users who want to continue the conversion.
    ``output_dir`` is task-scoped and may safely be recreated by ModelX.
    """

    source_file = _primary_source(source).resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ModelX keeps a process-level CodeFileGenerator singleton.  Serialising
    # calls and resetting it makes repeated web tasks independent while
    # retaining the original converter implementation.
    with _MODELX_LOCK, _modelx_import_path():
        from CMC.converter import Converter
        from CMC.utils import CodeFileGenerator

        CodeFileGenerator._instance = None
        CodeFileGenerator._file_initialized = False
        old_log_disable = logging.root.manager.disable
        output_path = str(output_dir)
        old_output_path_count = sys.path.count(output_path)
        try:
            converter = Converter(
                str(source_file),
                str(output_dir),
                log_dir="disable",
                show_unsupport=False,
                sourceDLClass="pytorch",
                targetDLClass="paddlepaddle",
            )
            success, failed = converter.run()
        finally:
            # ModelX's CLI disables process-wide logging; restore it for the
            # web server after every conversion, including failed conversions.
            logging.disable(old_log_disable)
            # CMC appends its task output directory while resolving helpers.
            # Remove only this call's additions, not other threads' paths.
            while sys.path.count(output_path) > old_output_path_count:
                sys.path.remove(output_path)

    generated = output_dir / source_file.name
    if not generated.exists():
        candidates = sorted(output_dir.rglob("*.py"), key=lambda p: p.stat().st_mtime)
        candidates = [p for p in candidates if p.name != "codeGenerationInfo.py"]
        if not candidates:
            raise RuntimeError("ModelX did not produce a converted Python file")
        generated = candidates[-1]

    raw_code = generated.read_text(encoding="utf-8", errors="replace")
    code = _normalize_modelx_code(raw_code)
    extra_files = {
        p.relative_to(output_dir).as_posix(): p.read_text(encoding="utf-8", errors="replace")
        for p in sorted(output_dir.rglob("*.py"))
        if p != generated and p.name != "codeGenerationInfo.py" and "__pycache__" not in p.parts
    }
    # Keep the on-disk artifact identical to the version displayed/downloaded.
    generated.write_text(code, encoding="utf-8")
    return ConversionResult(
        code=code,
        source_file=str(source_file),
        success_api_count=int(success or 0),
        failed_api_count=int(failed or 0),
        unsupported={str(k): int(v) for k, v in getattr(converter, "unsupport_map", {}).items()},
        extra_files=extra_files,
        compatibility_fixes=["补齐仓库裁剪源码缺失的 eager_outputs 方法"] if raw_code != code else [],
    )


def paddle_runtime_available() -> bool:
    """Return whether the Paddle runtime can be imported."""

    try:
        return importlib.util.find_spec("paddle") is not None
    except (ImportError, ValueError):
        return False


def run_paddle_smoke_test(code_path: Path, timeout: int = 90, profile: Optional[dict] = None) -> dict[str, Any]:
    """Import a generated module and run a best-effort dummy forward pass.

    Converted models have different constructor signatures.  We always test
    syntax/import and run a forward pass when a Layer with only optional
    constructor arguments is available.  A skipped forward is reported as a
    measured result instead of being treated as a conversion failure.
    """

    if not paddle_runtime_available():
        return {
            "attempted": False,
            "available": False,
            "import_ok": False,
            "forward_attempted": False,
            "forward_ok": False,
            "skip_reason": "当前环境未安装 PaddlePaddle，已完成静态评估",
        }

    code_path = code_path.resolve()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "OMP_NUM_THREADS": "1"}
    try:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("paddle_smoke_runner.py")),
             str(code_path), json.dumps(profile or {}, ensure_ascii=False)],
            cwd=str(code_path.parent), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout, env=env,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return {"attempted": True, "available": True, "import_ok": False,
                "forward_attempted": False, "forward_ok": False,
                "error": f"Paddle 冒烟测试超时（{timeout}s）"}

    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith("__PADDLE_RESULT__"):
            result = json.loads(line[len("__PADDLE_RESULT__"):])
            if proc.returncode:
                result["forward_ok"] = False
                result["error"] = (proc.stderr or f"Paddle subprocess exit {proc.returncode}")[-600:]
            return result
    return {
        "attempted": True, "available": True, "import_ok": False,
        "forward_attempted": False, "forward_ok": False,
        "error": (proc.stderr or proc.stdout or "Paddle 冒烟测试没有返回结果")[-800:],
    }
