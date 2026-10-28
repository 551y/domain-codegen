"""Task-isolated SemTrans generation, semantic diagnosis, repair and re-check."""

from __future__ import annotations

import contextlib
import io
import json
import math
import logging
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any


EVENT_PREFIX = "__SEMTRANS_EVENT__"
RESULT_PREFIX = "__SEMTRANS_RESULT__"
ROOT = Path(__file__).resolve().parents[1]
VENDORED = ROOT / "vendor" / "semtrans"
sys.path.insert(0, str(VENDORED))


def _emit(kind: str, payload: dict) -> None:
    print(EVENT_PREFIX + json.dumps({"kind": kind, **payload}, ensure_ascii=False), flush=True)


def _json_safe(value: Any, depth: int = 0):
    if depth > 7:
        return str(value)[:1000]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value if not isinstance(value, str) else value[:12000]
    if isinstance(value, dict):
        return {str(k)[:200]: _json_safe(v, depth + 1) for k, v in list(value.items())[:80]}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v, depth + 1) for v in value[:80]]
    return str(value)[:2000]


def _load_semtrans():
    from SemTrans.enhanced_conversion.framework_knowledge_retrivals import enhanced_conversion_agent
    from SemTrans.math_symbol_base.frontends.abstract import AbsTensor
    from SemTrans.math_symbol_base.frontends.materialize import create_symbolnet, load_model_instance
    from SemTrans.testing_and_repair.constraint_driven_repair import (
        assemble_batch_user_prompt,
        auto_repair_pipeline,
        call_llm_for_fix,
        get_repair_system_prompt,
    )
    from SemTrans.testing_and_repair.logging import CONVERSION_LOG
    return (enhanced_conversion_agent, AbsTensor, create_symbolnet, load_model_instance,
            assemble_batch_user_prompt, auto_repair_pipeline, call_llm_for_fix,
            get_repair_system_prompt, CONVERSION_LOG)


def _instantiate(source_path: Path, target_path: Path, profile: dict, api):
    AbsTensor, create_symbolnet, load_model_instance = api[1], api[2], api[3]
    class_name = profile["class_name"]
    init_kwargs = profile.get("init_kwargs", {})
    source = load_model_instance(str(source_path), class_name, "cpu", init_kwargs=init_kwargs)
    target = load_model_instance(str(target_path), class_name, "cpu", init_kwargs=init_kwargs)
    input_map = {
        "x": AbsTensor(list(profile["input_shape"]), profile.get("input_dtype", "float32")),
    }
    src_net = create_symbolnet(source, use_gradient=False, input_map=input_map, framework_type="pytorch")
    tgt_net = create_symbolnet(target, use_gradient=False, input_map=input_map, framework_type="paddle")
    src_net.to("cpu")
    tgt_net.to("cpu")
    return src_net, tgt_net


def _validate(source_path: Path, target_path: Path, profile: dict, config: dict,
              api, *, seed: int, phase: str, rtol: float, atol: float):
    _, AbsTensor, create_symbolnet, load_model_instance, _, auto_repair_pipeline, _, _, logger = api
    del AbsTensor, create_symbolnet, load_model_instance
    telemetry: list[dict] = []
    semantic_evidence: dict = {}
    numerical_checks: list[dict] = []
    search_config = {
        "bounded_demo": True,
        "max_sample": int(config.get("max_sample", 2)),
        "max_time_ms": int(config.get("search_budget_ms", 1200)),
        "target_count": int(config.get("target_count", 3)),
        "seed": seed,
        "telemetry": telemetry,
        "semantic_evidence": semantic_evidence,
        "numerical_checks": numerical_checks,
    }
    started = time.perf_counter()
    try:
        source_model, target_model = _instantiate(source_path, target_path, profile, api)
        with contextlib.redirect_stdout(io.StringIO()):
            success, issues = auto_repair_pipeline(
                source_model=source_model,
                target_model=target_model,
                logger=logger,
                atol=atol,
                rtol=rtol,
                use_cuda=False,
                is_static_source=False,
                input_search_config=search_config,
            )
    except Exception as exc:
        elapsed = round(time.perf_counter() - started, 3)
        status = "unverified" if isinstance(exc, (ImportError, ModuleNotFoundError, NotImplementedError)) else "failed"
        issue = {
            "cluster_key": "target_runtime_or_semtrans_error",
            "root_cause": f"{type(exc).__name__}: {exc}",
            "fault_layer": "model loading / SemTrans frontend / differential testing",
            "prompt_content": f"Runtime diagnostic: {type(exc).__name__}: {exc}",
            "case_count": 1,
            "evidence_kind": "runtime_or_structural_diagnostic",
        }
        return ({
            "phase": phase,
            "status": status,
            "verified": False,
            "reason": "SemTrans frontend, operator definition or runtime dependency is unavailable." if status == "unverified" else None,
            "input_search": telemetry,
            "valid_paired_cases": sum(row.get("valid_cases", 0) for row in telemetry),
            "search_attempts": sum(row.get("attempts", 0) for row in telemetry),
            "atol": atol,
            "rtol": rtol,
            "elapsed_s": elapsed,
            "issues": [_json_safe(issue)],
            "semantic_evidence": _json_safe(semantic_evidence),
            "numerical_checks": _json_safe(numerical_checks),
        }, [issue])
    elapsed = round(time.perf_counter() - started, 3)
    valid_cases = sum(row.get("valid_cases", 0) for row in telemetry)
    attempts = sum(row.get("attempts", 0) for row in telemetry)
    output_checks = [
        output
        for case in numerical_checks
        for output in case.get("outputs", [])
    ]
    report = {
        "phase": phase,
        "status": "passed" if success and valid_cases > 0 else "failed" if issues else "unverified",
        "verified": bool(success and valid_cases > 0),
        "input_search": telemetry,
        "valid_paired_cases": valid_cases,
        "search_attempts": attempts,
        "atol": atol,
        "rtol": rtol,
        "elapsed_s": elapsed,
        "issues": _json_safe(issues or []),
        "semantic_evidence": _json_safe(semantic_evidence),
        "numerical_checks": _json_safe(numerical_checks),
        "numerical_summary": {
            "checked_cases": len(numerical_checks),
            "checked_outputs": len(output_checks),
            "max_absolute_error": max((row.get("max_absolute_error", 0.0) for row in output_checks), default=None),
            "max_relative_error": max((row.get("max_relative_error", 0.0) for row in output_checks), default=None),
            "failed_elements": sum(row.get("failed_elements", 0) for row in output_checks),
            "checked_elements": sum(row.get("element_count", 0) for row in output_checks),
        },
    }
    if success and valid_cases <= 0:
        report["status"] = "unverified"
        report["reason"] = "SemTrans completed without producing a valid paired test input."
    return report, issues or []


def run(request: dict) -> dict:
    started = time.perf_counter()
    output_dir = Path(request["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / "source_model.py"
    target_path = output_dir / "paddle_model.py"
    source_path.write_text(request["source_code"], encoding="utf-8")
    profile = request["profile"]
    config = request["config"]
    stages = []
    api = _load_semtrans()
    generate, _, _, _, assemble_prompt, _, call_fix, repair_system, logger = api
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL)

    api_key = request["model_service"]["api_key"]
    provider = request["model_service"].get("provider", "")
    model = request["model_service"]["model"]
    from semtrans_llm import OpenAICompatibleClient
    client = OpenAICompatibleClient(
        base_url=request["model_service"]["base_url"],
        api_key=api_key,
        provider=request["model_service"].get("provider", ""),
        timeout=float(config.get("llm_timeout_s", 75)),
    )
    source_label = request.get("source_label", "source_model.py")
    _emit("stage", {"name": "SemTrans 数学语义与模型前端准备", "status": "running"})
    from SemTrans.math_symbol_base.frontends.abstract import AbsTensor
    from SemTrans.math_symbol_base.frontends.materialize import create_symbolnet, load_model_instance
    source_instance = load_model_instance(
        str(source_path), profile["class_name"], "cpu", init_kwargs=profile.get("init_kwargs", {})
    )
    source_input_map = {"x": AbsTensor(list(profile["input_shape"]), profile.get("input_dtype", "float32"))}
    source_symbolnet = create_symbolnet(
        source_instance, use_gradient=False, input_map=source_input_map, framework_type="pytorch"
    )
    source_symbolnet.to("cpu")
    import torch
    with torch.no_grad():
        source_instance.eval()
        source_output = source_instance(torch.zeros(profile["input_shape"], dtype=torch.float32))
    if source_output is None:
        raise RuntimeError("源 PyTorch 模型没有返回输出")
    stages.append({"name": "SemTrans 数学语义与模型前端准备", "status": "passed",
                   "detail": "加载 PyTorch/Paddle frontend 与包内数学定义"})
    _emit("stage", stages[-1])

    _emit("stage", {"name": "LLM 生成 Paddle 候选", "status": "running"})
    conversion_started = time.perf_counter()
    guidance = profile.get("task_guidance", "")
    with contextlib.redirect_stdout(io.StringIO()):
        candidate = generate(
            code_content=request["source_code"], sourceFramework="PyTorch",
            targetFramework="PaddlePaddle", logger=logger, is_JIT=False,
            LLM_api=client, LLM_type=model, task_guidance=guidance,
            compact_prompt=True,
        )
    if not isinstance(candidate, str) or not candidate.strip():
        raise RuntimeError("SemTrans LLM 未返回 Paddle 候选代码")
    candidate = candidate.strip() + "\n"
    # Paddle's tracer records Python ``Tensor.__add__`` residuals as a
    # captured const input.  Keep MNASNet's residual edge explicit through
    # Paddle's named add API so SemTrans can align it with PyTorch's add node.
    if str(profile.get("class_name", "")) == "MNASNet":
        candidate = re.sub(
            r"return\s+(self\.layers\((?P<var>[A-Za-z_]\w*)\))\s*\+\s*(?P=var)",
            r"return paddle.add(\1, \g<var>)",
            candidate,
        )
    generation_time = round(time.perf_counter() - conversion_started, 3)
    target_path.write_text(candidate, encoding="utf-8")
    stages.append({"name": "LLM 生成 Paddle 候选", "status": "passed",
                   "detail": f"OpenAI-compatible API / {model} · {generation_time:.2f} 秒"})
    _emit("stage", stages[-1])

    initial_candidate = candidate
    injected_fault = None
    if profile.get("fault_injection"):
        fault = profile["fault_injection"]
        pattern = re.compile(fault["pattern"])
        candidate, replacements = pattern.subn(fault["replacement"], candidate, count=1)
        if replacements != 1:
            raise RuntimeError("SemTrans 教学故障注入点未在 LLM 生成代码中唯一匹配，拒绝伪造该案例")
        injected_fault = {
            "label": fault["label"],
            "mutation": fault.get("mutation", f"{fault['pattern']} → {fault['replacement']}"),
            "scope": "仅限明确标记的语义诊断教学样例",
        }
        target_path.write_text(candidate, encoding="utf-8")

    rounds = []
    code_history = [{"round": 0, "stage": "initial_candidate", "content": candidate}]
    max_repairs = max(0, min(int(config.get("max_repairs", 2)), 2))
    current_issues = []
    source_framework, target_framework = "pytorch", "paddle"
    repair_round = 0
    final_validation = {"status": "unverified", "verified": False}
    while True:
        _emit("stage", {"name": "SemTrans 语义对齐与差异定位", "status": "running"})
        report, issues = _validate(
            source_path, target_path, profile, config, api,
            seed=int(config.get("seed", 20261027)) + repair_round * 1009,
            phase="initial" if repair_round == 0 else f"repair_{repair_round}",
            rtol=float(config.get("rtol", 1e-4)), atol=float(config.get("atol", 1e-4)),
        )
        rounds.append(report)
        current_issues = issues
        final_validation = report
        stage = {
            "name": "SemTrans 语义对齐与差异定位" if repair_round == 0 else f"修复后诊断检查（第 {repair_round} 轮）",
            "status": report["status"],
            "detail": f"有效配对输入 {report['valid_paired_cases']} 组 · {report['elapsed_s']:.2f} 秒",
        }
        stages.append(stage)
        _emit("stage", stage)

        if report["verified"]:
            _emit("stage", {"name": "独立输入复验", "status": "running"})
            holdout, holdout_issues = _validate(
                source_path, target_path, profile, config, api,
                seed=int(config.get("seed", 20261027)) + 900001 + repair_round * 1013,
                phase=f"held_out_{repair_round}",
                rtol=float(config.get("rtol", 1e-4)), atol=float(config.get("atol", 1e-4)),
            )
            rounds.append(holdout)
            final_validation = holdout
            current_issues = holdout_issues
            holdout_stage = {
                "name": "独立输入复验",
                "status": holdout["status"],
                "detail": f"独立有效配对输入 {holdout['valid_paired_cases']} 组 · {holdout['elapsed_s']:.2f} 秒",
            }
            stages.append(holdout_stage)
            _emit("stage", holdout_stage)
            if holdout["verified"]:
                break
            report["status"] = "failed"
            report["verified"] = False
            report["reason"] = "SemTrans training/diagnostic samples passed but the independent held-out samples did not."

        if report["status"] == "unverified" or final_validation["status"] == "unverified" or repair_round >= max_repairs:
            break
        repair_tasks = current_issues if isinstance(current_issues, list) else []
        if not repair_tasks:
            break
        _emit("stage", {"name": f"SemTrans 定向代码修复（第 {repair_round + 1} 轮）", "status": "running"})
        repair_started = time.perf_counter()
        source_text = source_path.read_text(encoding="utf-8")
        current_code = target_path.read_text(encoding="utf-8")
        repair_prompt = assemble_prompt(
            repair_tasks, current_code, source_text, False,
            sourceFramework=source_framework, targetFramework=target_framework,
            logger=logger, is_JIT=False, LLM_api=client, LLM_type=model,
        )
        repaired = call_fix(
            system_prompt=repair_system(source_framework, target_framework),
            user_prompt=repair_prompt, logger=logger, LLM_api=client, LLM_type=model,
        )
        if not isinstance(repaired, str) or not repaired.strip():
            raise RuntimeError("模型服务未返回修复后的 Paddle 代码")
        repaired = repaired.strip() + "\n"
        compile(repaired, str(target_path), "exec")
        target_path.write_text(repaired, encoding="utf-8")
        repair_round += 1
        code_history.append({"round": repair_round, "stage": "repaired_candidate", "content": repaired})
        repair_stage = {
            "name": f"SemTrans 定向代码修复（第 {repair_round} 轮）",
            "status": "passed",
            "detail": f"已生成修复版本 · {time.perf_counter() - repair_started:.2f} 秒",
        }
        stages.append(repair_stage)
        _emit("stage", repair_stage)

    result = {
        "status": final_validation["status"],
        "verified": final_validation.get("verified", False),
        "source_label": source_label,
        "profile_label": profile.get("label", profile.get("class_name", "")),
        "provider": provider,
        "input_shape": list(profile["input_shape"]),
        "model": model,
        "generation_elapsed_s": generation_time,
        "total_elapsed_s": round(time.perf_counter() - started, 3),
        "stages": stages,
        "validation_rounds": rounds,
        "repair_rounds": repair_round,
        "code_history": code_history,
        "diagnostic": _json_safe(current_issues),
        "files": [
            {"name": "paddle_model.py", "language": "python", "content": target_path.read_text(encoding="utf-8")},
            {"name": "initial_candidate.py", "language": "python", "content": initial_candidate},
            {"name": "source_model.py", "language": "python", "content": source_path.read_text(encoding="utf-8")},
        ],
        "limitations": [
            "经验性有限输入语义检查，不是形式化证明。",
            "SemTrans-JIT 依赖已移植的算子定义；未覆盖运算会返回受限/未验证。",
        ],
    }
    if injected_fault:
        result["fault_injection"] = injected_fault
    return result


def main() -> None:
    try:
        request = json.load(sys.stdin)
        result = run(request)
        print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as exc:
        print(RESULT_PREFIX + json.dumps({
            "status": "failed", "verified": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-10000:],
        }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
