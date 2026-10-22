"""
评测对比路由
"""

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from herta.api.task_store import tasks
from herta.utils.paths import project_relative
from herta.logging import get_logger
from herta.api.guards import validate_name

logger = get_logger(__name__)
router = APIRouter()


class CompareRequest(BaseModel):
    target: str


@router.post("/run")
async def run_compare(req: CompareRequest):
    """运行评测对比（后台任务）"""
    task_id = tasks.create("compare")

    asyncio.get_event_loop().run_in_executor(
        None, _run_compare, task_id, req
    )

    return {"task_id": task_id, "status": "pending"}


def _run_compare(task_id: str, req: CompareRequest):
    """线程池中执行评测对比"""
    try:
        tasks[task_id]["status"] = "running"
        tasks[task_id]["message"] = "正在比较 Config JSON..."
        tasks[task_id]["progress"] = 20

        target = req.target
        gen_config = project_relative(f"output/generated/{target}/{target}_config.json")
        gt_config = project_relative(f"data/target_proto/{target}/{target}_config.json")
        gen_handler = project_relative(f"output/generated/{target}/{target}_handler.c")
        gt_handler = project_relative(f"data/target_proto/{target}/{target}_handler.c")

        from herta.compare import compare_configs, parse_c_functions, match_functions

        config_result = compare_configs(str(gen_config), str(gt_config))

        tasks[task_id]["message"] = "正在解析 Handler C 代码..."
        tasks[task_id]["progress"] = 50

        gen_functions = parse_c_functions(str(gen_handler))
        gt_functions = parse_c_functions(str(gt_handler))

        tasks[task_id]["message"] = "正在匹配函数..."
        tasks[task_id]["progress"] = 70

        handler_result = match_functions(gen_functions, gt_functions)

        tasks[task_id]["message"] = "正在生成报告..."
        tasks[task_id]["progress"] = 90

        # 生成报告
        report_dir = project_relative(f"output/reports/{target}")
        report_dir.mkdir(parents=True, exist_ok=True)

        from herta.compare import generate_comparison_report
        generate_comparison_report(
            config_result=config_result,
            handler_result=handler_result,
            target=target,
            output_dir=str(report_dir),
        )

        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = 100
        tasks[task_id]["message"] = "评测完成"

        overall = config_result.config_similarity * 0.35 + handler_result.handler_similarity * 0.65

        tasks[task_id]["result"] = {
            "topology_similarity": round(config_result.topology_similarity.score, 2),
            "cmd_list_similarity": round(config_result.cmd_list_similarity.score, 2),
            "tele_handler_similarity": round(config_result.tele_handler_similarity.score, 2),
            "internal_handler_similarity": round(config_result.internal_handler_similarity.score, 2),
            "config_similarity": round(config_result.config_similarity, 2),
            "handler_similarity": round(handler_result.handler_similarity, 2),
            "overall_similarity": round(overall, 2),
            "gen_function_count": len(gen_functions),
            "gt_function_count": len(gt_functions),
            "matched_count": len([m for m in handler_result.matches if m.is_matched]),
            "unmatched_gt": handler_result.unmatched_gt_functions,
            "function_matches": [m.to_dict() for m in handler_result.matches],
        }

    except Exception as e:
        logger.error(f"评测对比失败: {e}")
        import traceback
        traceback.print_exc()
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["message"] = f"评测失败: {e}"


@router.get("/result/{target}")
async def get_compare_result(target: str):
    """获取评测结果"""
    target = validate_name(target, "target")
    metrics_file = project_relative(f"output/reports/{target}/metrics.json")
    if not metrics_file.exists():
        return {"exists": False}

    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # 适配前端期望的 details 结构
    if "details" not in metrics:
        metrics["details"] = {
            "function_matches": metrics.get("function_match_results", []),
            "unmatched_gt_functions": metrics.get("unmatched_gt_functions", []),
            "unmatched_gen_functions": metrics.get("unmatched_gen_functions", []),
        }

    return {"exists": True, "metrics": metrics}


@router.get("/report/{target}")
async def get_compare_report(target: str):
    """获取 Markdown 报告"""
    target = validate_name(target, "target")
    report_file = project_relative(f"output/reports/{target}/comparison_report.md")
    if not report_file.exists():
        return {"exists": False}

    with open(report_file, "r", encoding="utf-8") as f:
        content = f.read()

    return {"exists": True, "content": content}


@router.get("/gt-config/{target}")
async def get_gt_config(target: str):
    """获取真实配置文件"""
    target = validate_name(target, "target")
    config_file = project_relative(f"data/target_proto/{target}/{target}_config.json")
    if not config_file.exists():
        return {"exists": False}

    with open(config_file, "r", encoding="utf-8") as f:
        return {"exists": True, "config": json.load(f)}


@router.get("/gt-handler/{target}")
async def get_gt_handler(target: str):
    """获取真实 Handler 源码"""
    target = validate_name(target, "target")
    handler_file = project_relative(f"data/target_proto/{target}/{target}_handler.c")
    if not handler_file.exists():
        return {"exists": False}

    with open(handler_file, "r", encoding="utf-8") as f:
        content = f.read()

    return {"exists": True, "source": content, "lines": len(content.splitlines())}
