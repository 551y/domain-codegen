"""
代码生成路由
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


class GenerateConfigRequest(BaseModel):
    target: str
    source_kb: str = ""
    top_k: int = 5


class GenerateHandlerRequest(BaseModel):
    target: str
    source_kb: str = ""
    top_k: int = 8


@router.get("/targets")
async def list_targets():
    """列出可用的目标协议"""
    target_dir = project_relative("data/target_proto")
    if not target_dir.exists():
        return {"targets": []}

    result = []
    for d in sorted(target_dir.iterdir()):
        if d.is_dir():
            has_docx = any(d.glob("*.docx"))
            has_gt_config = (d / f"{d.name}_config.json").exists()
            has_gt_handler = (d / f"{d.name}_handler.c").exists()
            result.append({
                "name": d.name,
                "has_docx": has_docx,
                "has_gt_config": has_gt_config,
                "has_gt_handler": has_gt_handler,
            })

    return {"targets": result}


@router.post("/config")
async def generate_config(req: GenerateConfigRequest):
    """生成 config.json（后台任务）"""
    task_id = tasks.create("generate")

    asyncio.get_event_loop().run_in_executor(
        None, _run_generate_config, task_id, req
    )

    return {"task_id": task_id, "status": "pending"}


@router.post("/handler")
async def generate_handler(req: GenerateHandlerRequest):
    """生成 handler.c（后台任务）"""
    task_id = tasks.create("generate")

    asyncio.get_event_loop().run_in_executor(
        None, _run_generate_handler, task_id, req
    )

    return {"task_id": task_id, "status": "pending"}


def _run_generate_config(task_id: str, req: GenerateConfigRequest):
    """线程池中执行配置生成"""
    try:
        tasks[task_id]["status"] = "running"
        tasks[task_id]["message"] = "正在检索知识库..."
        tasks[task_id]["progress"] = 20

        from herta.generator.config_generator import ConfigGenerator

        kb_dir = project_relative(f"data/kb/{req.source_kb}") if req.source_kb else None
        target_doc = project_relative(f"data/target_proto/{req.target}/{req.target}.docx")

        output_dir = project_relative(f"output/generated/{req.target}")
        output_dir.mkdir(parents=True, exist_ok=True)

        generator = ConfigGenerator()

        tasks[task_id]["message"] = "正在调用 LLM 生成配置..."
        tasks[task_id]["progress"] = 50

        result = generator.generate(
            target_doc_path=target_doc,
            source_kb_dir=kb_dir,
            target_name=req.target,
            top_k=req.top_k,
            output_dir=output_dir,
        )

        # 生成器内部会把阶段失败吞进 result（error_message/is_valid），需据此判定任务成败
        if getattr(result, "error_message", None) or getattr(result, "is_valid", True) is False:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = getattr(result, "error_message", None) or "生成结果未通过校验"
            tasks[task_id]["message"] = f"生成失败: {tasks[task_id]['error']}"
            return

        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = 100
        tasks[task_id]["message"] = "配置生成完成"
        tasks[task_id]["result"] = {
            "config_path": result.output_path,
            "retrieval_count": result.retrieved_chunk_count,
            "model": result.model,
        }

    except Exception as e:
        logger.error(f"配置生成失败: {e}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["message"] = f"生成失败: {e}"


def _run_generate_handler(task_id: str, req: GenerateHandlerRequest):
    """线程池中执行 Handler 生成"""
    try:
        tasks[task_id]["status"] = "running"
        tasks[task_id]["message"] = "正在检索知识库..."
        tasks[task_id]["progress"] = 10

        from herta.generator.handler_generator import HandlerGenerator

        kb_dir = project_relative(f"data/kb/{req.source_kb}") if req.source_kb else None
        target_doc = project_relative(f"data/target_proto/{req.target}/{req.target}.docx")

        output_dir = project_relative(f"output/generated/{req.target}")
        output_dir.mkdir(parents=True, exist_ok=True)

        generator = HandlerGenerator()

        tasks[task_id]["message"] = "阶段A: 生成设计草图..."
        tasks[task_id]["progress"] = 30

        result = generator.generate(
            target_doc_path=target_doc,
            source_kb_dir=kb_dir,
            target_name=req.target,
            top_k=req.top_k,
            output_dir=output_dir,
        )

        if getattr(result, "error_message", None) or getattr(result, "is_valid", True) is False:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = getattr(result, "error_message", None) or "生成结果未通过校验"
            tasks[task_id]["message"] = f"生成失败: {tasks[task_id]['error']}"
            return

        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = 100
        tasks[task_id]["message"] = "Handler 代码生成完成"
        tasks[task_id]["result"] = {
            "handler_path": result.code_output_path,
            "plan_path": result.plan_output_path,
            "retrieval_count": result.retrieved_chunk_count,
            "model": result.model,
        }

    except Exception as e:
        logger.error(f"Handler 生成失败: {e}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["message"] = f"生成失败: {e}"


@router.get("/result/{target}")
async def get_generated_result(target: str):
    """获取已生成的结果"""
    target = validate_name(target, "target")
    gen_dir = project_relative(f"output/generated/{target}")
    if not gen_dir.exists():
        return {"exists": False}

    result = {"exists": True, "target": target}

    config_file = gen_dir / f"{target}_config.json"
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            result["config"] = json.load(f)

    handler_file = gen_dir / f"{target}_handler.c"
    if handler_file.exists():
        with open(handler_file, "r", encoding="utf-8") as f:
            result["handler_source"] = f.read()
            result["handler_lines"] = len(result["handler_source"].splitlines())

    return result


# ---------------------------------------------------------------------------
# 生成输入预览（演示用：解析 → 检索 → 提示词组装，全程不调用 LLM，离线可用）
# ---------------------------------------------------------------------------

@router.get("/preview")
async def preview_generation_input(target: str, kind: str = "config", source_kb: str = ""):
    """组装一次生成请求的完整输入（查询、知识块、提示词），但不调用 LLM。

    direct 策略的知识库走 retrieve_all() 全量注入，整条链路零网络调用；
    其他策略离线时检索可能失败，降级为空检索结果并标注。
    """
    target = validate_name(target, "target")
    kind = kind if kind in ("config", "handler") else "config"

    # 目标协议文档
    tdir = project_relative(f"data/target_proto/{target}")
    docs = sorted(tdir.glob("*.docx")) if tdir.exists() else []
    if not docs:
        return {"exists": False, "error": f"目标协议 {target} 下未找到 docx 文档"}

    # 知识库：缺省取第一个
    kb_root = project_relative("data/kb")
    if source_kb:
        source_kb = validate_name(source_kb, "source_kb")
        kb_dir = kb_root / source_kb
    else:
        subs = [d for d in sorted(kb_root.iterdir()) if d.is_dir()] if kb_root.exists() else []
        kb_dir = subs[0] if subs else None
    if kb_dir is None or not kb_dir.exists():
        return {"exists": False, "error": "知识库不存在"}

    from herta.generator.target_parser import TargetParser
    from herta.generator.retriever import KnowledgeRetriever
    from herta.generator.prompt_loader import PromptLoader

    # 1. 解析目标协议
    spec = TargetParser().parse(str(docs[0]), target)

    # 2. 检索（direct 全量注入；其他策略尝试多查询检索，离线失败则降级）
    retriever = KnowledgeRetriever(kb_dir)
    retriever.load()
    strategy = (retriever.meta or {}).get("chunk_strategy", "")
    retrieval_note = ""

    # 构造多查询（展示框架的查询设计；direct 模式运行时实际跳过检索）
    queries: list[str] = []
    try:
        if kind == "config":
            from herta.generator.config_generator import ConfigGenerator
            queries = ConfigGenerator()._build_multi_queries(spec)
        else:
            from herta.generator.handler_generator import HandlerGenerator
            queries = HandlerGenerator()._build_handler_multi_queries(spec)
    except Exception as e:
        logger.warning(f"构造多查询失败（不影响预览）: {e}")

    if strategy == "direct":
        chunks = retriever.retrieve_all()
        retrieval_note = f"direct 策略：全量注入 {len(chunks)} 个知识块（不走向量检索）"
    else:
        try:
            chunks = retriever.retrieve_multi(queries=queries, top_k_per_query=10, max_total=30)
            retrieval_note = f"多查询检索：{len(queries)} 条查询召回 {len(chunks)} 个知识块"
        except Exception as e:
            chunks = []
            retrieval_note = f"离线环境向量检索不可用（{e}），检索结果为空"

    # 3. 组装提示词（与真实生成完全一致的模板与变量）
    prompt_loader = PromptLoader()
    prompt_stage2 = None
    if kind == "config":
        from herta.generator.schemas import GenerationContext
        context = GenerationContext(
            target=spec, retrieved_chunks=chunks,
            source_kb=str(kb_dir), top_k=30, model="", temperature=0.1,
        )
        target_summary = context.format_target_summary()
        retrieved_context = context.format_retrieved_context()
        user_prompt = prompt_loader.load_and_render(
            "generate_config.md",
            {"target_summary": target_summary, "retrieved_context": retrieved_context,
             "output_schema": context.output_schema},
        )
        system_prompt = "你是一个专业的协议配置生成助手。请严格按照指定的 JSON 结构输出配置文件，不要增减任何顶层字段。"
        template = "generate_config.md"
        stage_label = "Config 生成（单阶段）"
    else:
        from herta.generator.handler_generator import HandlerGenerator
        gen = HandlerGenerator()
        target_summary = gen._format_target_summary(spec)
        retrieved_context = gen._format_retrieved_context(chunks)
        user_prompt = prompt_loader.load_and_render(
            "generate_handler_plan.md",
            {"target_summary": target_summary, "retrieved_context": retrieved_context},
        )
        system_prompt = "你是一位资深的嵌入式系统架构师，专注于协议处理模块的设计。"
        template = "generate_handler_plan.md"
        stage_label = "Handler 生成 · 阶段A（设计草图）"

    # 两阶段草图（若已通过 capture_trace 或 --save-plan 预生成）
    plan_file = project_relative(f"output/intermediate/{target}/handler_plan.md")
    plan: dict = {"exists": plan_file.exists()}
    if plan_file.exists():
        plan["content"] = plan_file.read_text(encoding="utf-8", errors="replace")
        if kind == "handler":
            from herta.generator.handler_generator import HandlerGenerator
            gen = HandlerGenerator()
            stage2_prompt = gen.prompt_loader.load_and_render(
                "generate_handler_code.md",
                {"target_summary": target_summary, "handler_plan": plan["content"],
                 "retrieved_context": retrieved_context},
            )
            prompt_stage2 = {
                "template": "generate_handler_code.md",
                "stage_label": "Handler 生成 · 阶段B（完整代码）",
                "system": "你是一位资深的嵌入式 C 语言开发者，擅长编写高效、健壮的协议处理代码。",
                "user": stage2_prompt,
                "user_chars": len(stage2_prompt),
            }

    return {
        "exists": True,
        "target": target,
        "kind": kind,
        "doc": {
            "file": docs[0].name,
            "raw_chars": len(spec.raw_text),
            "sections": [str(s.get("title") or s.get("name") or "") for s in (spec.sections or [])][:20],
        },
        "kb": {"source": kb_dir.name, "strategy": strategy, "chunk_total": len(chunks)},
        "retrieval_note": retrieval_note,
        "queries": queries,
        "chunks": [
            {
                "chunk_id": c.chunk_id, "kind": c.kind, "file": c.file,
                "section": c.section, "score": round(c.score, 3),
                "token_count": c.token_count, "text_preview": (c.text or "")[:300],
            }
            for c in chunks
        ],
        "target_summary": target_summary,
        "retrieved_context_chars": len(retrieved_context),
        "prompt": {
            "template": template, "stage_label": stage_label,
            "system": system_prompt, "user": user_prompt, "user_chars": len(user_prompt),
        },
        "prompt_stage2": prompt_stage2,
        "plan": plan,
    }


@router.get("/plan/{target}")
async def get_handler_plan(target: str):
    """获取已保存的 Handler 生成草图（阶段A 产物，若存在）"""
    target = validate_name(target, "target")
    plan_file = project_relative(f"output/intermediate/{target}/handler_plan.md")
    if not plan_file.exists():
        return {"exists": False}
    return {"exists": True, "content": plan_file.read_text(encoding="utf-8", errors="replace")}
