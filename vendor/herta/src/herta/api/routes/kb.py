"""
知识库管理路由
"""

import asyncio
import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

from herta.api.task_store import tasks
from herta.utils.paths import project_relative
from herta.logging import get_logger
from herta.api.guards import validate_name

logger = get_logger(__name__)
router = APIRouter()


class BuildKBRequest(BaseModel):
    source: str
    chunk_strategy: str = "simple"
    chunk_size: int = 500
    chunk_overlap: int = 50
    force: bool = False


@router.get("/list")
async def list_knowledge_bases():
    """列出所有知识库"""
    kb_dir = project_relative("data/kb")
    if not kb_dir.exists():
        return {"knowledge_bases": []}

    result = []
    for d in sorted(kb_dir.iterdir()):
        if d.is_dir():
            meta_file = d / "meta.json"
            meta = {}
            if meta_file.exists():
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            result.append({
                "source": d.name,
                "path": str(d),
                "chunk_count": meta.get("chunk_count", 0),
                "embedding_model": meta.get("embedding_model", ""),
                "chunk_strategy": meta.get("chunk_strategy", ""),
                "has_index": (d / "vector.index").exists(),
                "meta": meta,
            })

    return {"knowledge_bases": result}


@router.get("/status/{source}")
async def get_kb_status(source: str):
    """获取知识库状态"""
    source = validate_name(source, "source")
    kb_dir = project_relative(f"data/kb/{source}")
    if not kb_dir.exists():
        return {"exists": False}

    meta_file = kb_dir / "meta.json"
    meta = {}
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

    build_report_file = kb_dir / "build_report.json"
    build_report = {}
    if build_report_file.exists():
        with open(build_report_file, "r", encoding="utf-8") as f:
            build_report = json.load(f)

    return {
        "exists": True,
        "source": source,
        "path": str(kb_dir),
        "files": [f.name for f in kb_dir.iterdir()],
        "meta": meta,
        "build_report": build_report,
    }


@router.post("/build")
async def build_knowledge_base(req: BuildKBRequest):
    """构建知识库（后台任务）"""
    task_id = tasks.create("buildkb")

    asyncio.get_event_loop().run_in_executor(
        None, _run_buildkb, task_id, req
    )

    return {"task_id": task_id, "status": "pending"}


def _run_buildkb(task_id: str, req: BuildKBRequest):
    """在线程池中执行知识库构建"""
    try:
        tasks[task_id]["status"] = "running"
        tasks[task_id]["message"] = "正在解析文档..."
        tasks[task_id]["progress"] = 10

        from herta.kb.builder import build_kb
        from herta.kb.chunker import ChunkConfig
        from herta.utils.paths import project_relative

        source_dir = project_relative(f"data/source_proto/{req.source}")
        input_files = []

        for suffix in [".docx", ".pdf", ".json", ".c", ".h"]:
            for f in source_dir.glob(f"*{suffix}"):
                input_files.append(f)

        if not input_files:
            raise FileNotFoundError(f"未找到源文件: {source_dir}")

        tasks[task_id]["message"] = f"找到 {len(input_files)} 个文件，开始处理..."
        tasks[task_id]["progress"] = 20

        output_dir = project_relative(f"data/kb/{req.source}")

        # force 时先删除旧知识库
        if req.force and output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)
            logger.info(f"强制重建: 已删除旧知识库 {output_dir}")

        chunk_config = ChunkConfig(
            strategy=req.chunk_strategy,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
        )

        tasks[task_id]["message"] = "构建知识库中（可能需要几分钟）..."
        tasks[task_id]["progress"] = 30

        meta = build_kb(
            source=req.source,
            input_files=input_files,
            output_dir=output_dir,
            chunk_config=chunk_config,
        )

        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = 100
        tasks[task_id]["message"] = f"构建完成: {meta.chunk_count} 个 chunks"
        tasks[task_id]["result"] = {
            "chunk_count": meta.chunk_count,
            "embedding_model": meta.embedding_model,
            "source": meta.source,
        }

    except Exception as e:
        logger.error(f"构建知识库失败: {e}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["message"] = f"构建失败: {e}"


@router.get("/visualization/{source}")
async def get_kb_visualization(source: str):
    """获取知识库 Embedding 2D 可视化数据"""
    source = validate_name(source, "source")
    vis_dir = project_relative(f"output/visualization/{source}")
    proj_file = vis_dir / "projection_2d.json"

    if not proj_file.exists():
        return {"points": [], "meta": {"method": "none", "dimensions": 0}}

    with open(proj_file, "r", encoding="utf-8") as f:
        proj_data = json.load(f)

    coordinates = proj_data.get("coordinates", [])
    record_ids = proj_data.get("record_ids", [])
    method = proj_data.get("method", "unknown")

    # 加载 chunks 元数据以获取 kind/file
    chunks_file = project_relative(f"data/kb/{source}/chunks.jsonl")
    chunk_meta = {}
    if chunks_file.exists():
        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    c = json.loads(line)
                    chunk_meta[c.get("chunk_id", "")] = {
                        "kind": c.get("kind", "unknown"),
                        "file": c.get("file", ""),
                        "section": c.get("section", ""),
                    }

    def _infer_kind_from_id(record_id: str) -> str:
        """从 record_id 模式推断 kind（兼容旧版 projection 数据）"""
        rid = record_id.lower()
        if "_doc_" in rid or rid.startswith("doc_"):
            return "doc"
        if "_code_json_" in rid or "_json_" in rid:
            return "code_json"
        if "_code_c_" in rid or "_c_" in rid:
            return "code_c"
        if "_code_h_" in rid or "_h_" in rid:
            return "code_h"
        return "unknown"

    points = []
    for i, coord in enumerate(coordinates):
        cid = record_ids[i] if i < len(record_ids) else f"chunk_{i}"
        meta = chunk_meta.get(cid, {})
        kind = meta.get("kind") or _infer_kind_from_id(cid)
        points.append({
            "x": coord[0] if len(coord) > 0 else 0,
            "y": coord[1] if len(coord) > 1 else 0,
            "chunk_id": cid,
            "kind": kind,
            "file": meta.get("file", ""),
            "label": meta.get("section", cid),
        })

    return {
        "points": points,
        "meta": {
            "method": method,
            "dimensions": len(coordinates),
        },
    }


@router.get("/chunks/{source}")
async def get_kb_chunks(source: str, limit: int = Query(default=20, ge=1, le=100)):
    """获取知识库 chunks 样本"""
    source = validate_name(source, "source")
    chunks_file = project_relative(f"data/kb/{source}/chunks.jsonl")
    if not chunks_file.exists():
        return {"chunks": [], "total": 0}

    chunks = []
    total = 0
    # 单次遍历：统计非空行数并取前 limit 条样本
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            if len(chunks) < limit:
                chunk = json.loads(line)
                chunk["text_preview"] = chunk.get("text", "")[:200]
                chunk.pop("text", None)  # 统一 schema：一律不返回全文
                chunks.append(chunk)

    return {"chunks": chunks, "total": total}
