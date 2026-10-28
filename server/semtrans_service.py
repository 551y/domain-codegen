"""Backend orchestration for the vendored SemTrans worker."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable


EVENT_PREFIX = "__SEMTRANS_EVENT__"
RESULT_PREFIX = "__SEMTRANS_RESULT__"
WORKER = Path(__file__).with_name("semtrans_worker.py")


class SemTransWorkerError(RuntimeError):
    pass


def run_semtrans_worker(
    *, request: dict, task_dir: Path, timeout_s: int,
    on_event: Callable[[dict], None],
) -> dict:
    python = os.environ.get("SEMTRANS_PYTHON", sys.executable)
    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    proc = subprocess.Popen(
        [python, str(WORKER)], cwd=str(task_dir), stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        encoding="utf-8", errors="replace", bufsize=1, env=env,
    )
    lines: queue.Queue = queue.Queue()
    stderr_tail: list[str] = []

    def read_stdout():
        try:
            for line in proc.stdout:
                lines.put(line.rstrip("\r\n"))
        finally:
            lines.put(None)

    def read_stderr():
        try:
            for line in proc.stderr:
                stderr_tail.append(line)
                if sum(map(len, stderr_tail)) > 12000:
                    del stderr_tail[: max(1, len(stderr_tail) // 2)]
        finally:
            pass

    out_thread = threading.Thread(target=read_stdout, name=f"semtrans-out-{uuid.uuid4().hex[:6]}", daemon=True)
    err_thread = threading.Thread(target=read_stderr, name=f"semtrans-err-{uuid.uuid4().hex[:6]}", daemon=True)
    out_thread.start()
    err_thread.start()
    try:
        proc.stdin.write(json.dumps(request, ensure_ascii=False))
        proc.stdin.close()
        deadline = time.monotonic() + timeout_s
        result = None
        stdout_done = False
        while not stdout_done:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                proc.kill()
                proc.wait(timeout=5)
                raise SemTransWorkerError(f"SemTrans 演示超过 {timeout_s} 秒总时限，worker 已终止")
            try:
                line = lines.get(timeout=min(0.5, remaining))
            except queue.Empty:
                if proc.poll() is not None and not out_thread.is_alive():
                    break
                continue
            if line is None:
                stdout_done = True
            elif line.startswith(EVENT_PREFIX):
                try:
                    on_event(json.loads(line[len(EVENT_PREFIX):]))
                except (json.JSONDecodeError, TypeError):
                    continue
            elif line.startswith(RESULT_PREFIX):
                try:
                    result = json.loads(line[len(RESULT_PREFIX):])
                except json.JSONDecodeError:
                    result = {"status": "failed", "verified": False, "error": "SemTrans worker 返回了无效 JSON"}
        return_code = proc.wait(timeout=5)
        err_thread.join(timeout=1)
        if return_code != 0 and result is None:
            raise SemTransWorkerError(
                f"SemTrans worker 异常退出（代码 {return_code}）：{''.join(stderr_tail)[-2500:]}"
            )
        if result is None:
            raise SemTransWorkerError("SemTrans worker 未返回最终评测结果")
        return result
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        out_thread.join(timeout=1)
        err_thread.join(timeout=1)
