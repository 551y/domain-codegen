"""
基于RAG的代码生成基本框架 — 集成演示编排后端（domain-codegen）

一个后端统一编排两套知识库的代码生成流水线：
  - 入口一（协议知识库）：协议文档 → 检索增强 → LLM 生成 config/handler → 本地评测
  - 入口二（算子映射知识库）：PyTorch 源码 → 命中映射规则 → AST 转换 → 产物评测

引擎复用：herta（LLM 生成与评测）、opalign/CMC（确定性转换）。本文件只做编排与演示适配。
"""

import ast
import hashlib
import json
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# 路径与常量
# ---------------------------------------------------------------------------

PLATFORM_ROOT = Path(__file__).resolve().parent.parent
# 自包含部署：引擎与数据全部内置在 vendor/ 下，迁移只需拷贝整个目录
VENDOR_DIR = PLATFORM_ROOT / "vendor"
HERTA_ROOT = VENDOR_DIR / "herta"
OPALIGN_ROOT = VENDOR_DIR / "opalign"
WORKSPACE = PLATFORM_ROOT / "server" / "workspace"
UPLOAD_DIR = WORKSPACE / "uploads"
GEN_DIR = WORKSPACE / "generated"

HERTA_KB_DIR = HERTA_ROOT / "data" / "kb"
HERTA_TARGET_DIR = HERTA_ROOT / "data" / "target_proto"
HERTA_PRE_GEN = HERTA_ROOT / "output" / "generated"
OPALIGN_MODELS = OPALIGN_ROOT / "EVALUATION" / "datasets" / "models"
ONNX_DATASET = VENDOR_DIR / "onnx" / "dataset.standardized.jsonl"
ONNX_PRESET_DIR = PLATFORM_ROOT / "server" / "preset_onnx"

for d in (UPLOAD_DIR, GEN_DIR):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="RAG 代码生成基本框架 · 演示编排", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# 任务存储与工具
# ---------------------------------------------------------------------------

tasks: dict = {}
_tasks_lock = threading.Lock()


def _task_create(kind: str) -> str:
    task_id = uuid.uuid4().hex[:12]
    with _tasks_lock:
        tasks[task_id] = {
            "kind": kind, "status": "pending", "progress": 0,
            "message": "排队中", "result": None, "error": None,
            "intermediate": {},
        }
    return task_id


def _task_update(task_id: str, **kw):
    with _tasks_lock:
        if task_id in tasks:
            tasks[task_id].update(kw)


def _task_intermediate(task_id: str, **kw):
    """追加任务的中间过程数据（解析/检索等阶段的产出，供前端在执行过程中实时展示）"""
    with _tasks_lock:
        t = tasks.get(task_id)
        if t is not None:
            t.setdefault("intermediate", {}).update(kw)


def _llm_available() -> bool:
    try:
        from herta.config import get_settings
        s = get_settings()
        key = s.effective_chat_key() or ""
        return bool(key) and key != "offline-demo-placeholder"
    except Exception:
        key = os.environ.get("ZAI_API_KEY", "")
        return bool(key) and key != "offline-demo-placeholder"


def _safe_name(name: str) -> str:
    name = re.sub(r"[^\w.-]", "_", name)
    if not name or name.startswith("."):
        raise HTTPException(status_code=400, detail="非法文件名")
    return name


# ---------------------------------------------------------------------------
# ONNX 知识库（入口二）：加载 + 离线 BM25 检索
# ---------------------------------------------------------------------------

ONNX_CATEGORY_LABELS = {
    "operator_mapping": "算子映射",
    "operator_composite": "复合算子实现",
    "operator_testcase": "算子测试用例",
    "benchmark_case": "模型基准测试",
    "document": "技术文档",
    "fix_case": "问题修复案例",
    "version_matrix_row": "版本兼容矩阵",
    "error_pattern": "错误模式",
    "example": "转换示例",
}

_onnx_kb_cache: Optional[list] = None
_onnx_index_cache: Optional[dict] = None


def _onnx_kb() -> list:
    """加载 ONNX 知识库（1814 条标准化记录）"""
    global _onnx_kb_cache
    if _onnx_kb_cache is None:
        records = []
        if ONNX_DATASET.exists():
            with open(ONNX_DATASET, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    records.append({
                        "id": d.get("id", ""),
                        "title": d.get("title", ""),
                        "category": d.get("category", ""),
                        "category_label": ONNX_CATEGORY_LABELS.get(d.get("category", ""), d.get("category", "")),
                        "source": d.get("source", ""),
                        "framework_version": d.get("framework_version", ""),
                        "content": d.get("content", "") if isinstance(d.get("content"), str) else json.dumps(d.get("content"), ensure_ascii=False),
                        "verification_status": d.get("verification_status", ""),
                    })
        _onnx_kb_cache = records
    return _onnx_kb_cache


def _tokenize(text: str) -> list:
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+", text.lower())


def _char_bigrams(text: str) -> set:
    t = text.lower()
    return {t[i:i + 2] for i in range(len(t) - 1)}


def _onnx_index() -> dict:
    """构建 BM25 检索索引（内存计算，毫秒级）"""
    global _onnx_index_cache
    if _onnx_index_cache is not None:
        return _onnx_index_cache
    import math
    docs = []
    df: dict = {}
    total_len = 0
    for rec in _onnx_kb():
        text = f"{rec['title']} {rec['category_label']} {rec['source']}\n{rec['content']}"
        tokens = _tokenize(text)
        tf: dict = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        docs.append({"rec": rec, "tf": tf, "len": len(tokens), "bigrams": _char_bigrams(rec["title"] + " " + rec["source"])})
        total_len += len(tokens)
        for t in tf:
            df[t] = df.get(t, 0) + 1
    n = max(len(docs), 1)
    idf = {t: math.log((n - c + 0.5) / (c + 0.5) + 1) for t, c in df.items()}
    _onnx_index_cache = {"docs": docs, "idf": idf, "avg_len": total_len / n if n else 1}
    return _onnx_index_cache


def _onnx_retrieve(query: str, top_k: int = 8) -> list:
    """BM25 + bigram 混合检索（完全离线）"""
    idx = _onnx_index()
    q_tokens = _tokenize(query)
    q_bigrams = _char_bigrams(query)
    k1, b = 1.5, 0.75
    scored = []
    for doc in idx["docs"]:
        score = 0.0
        for t in q_tokens:
            if t not in doc["tf"]:
                continue
            tf = doc["tf"][t]
            denom = tf + k1 * (1 - b + b * doc["len"] / idx["avg_len"])
            score += idx["idf"].get(t, 0) * tf * (k1 + 1) / denom
        # bigram 字面重合加成（算子名等精确匹配）
        if q_bigrams:
            overlap = len(q_bigrams & doc["bigrams"]) / len(q_bigrams)
            score += overlap * 2.0
        if score > 0:
            scored.append((score, doc["rec"]))
    scored.sort(key=lambda x: -x[0])
    return [
        {**rec, "score": round(score, 2), "preview": rec["content"][:400]}
        for score, rec in scored[:top_k]
    ]


def _analyze_source(source: str) -> dict:
    """提取源码中的 torch 算子调用（用于检索查询构造与评测统计）"""
    calls = _collect_torch_calls(source)
    return {"total_calls": len(calls), "calls": calls}


def _collect_torch_calls(source: str) -> list:
    """AST 提取 torch.* 调用全名（处理 import 别名），去重保序。"""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    alias = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "torch" or a.name.startswith("torch."):
                    alias[a.asname or a.name.split(".")[-1]] = a.name
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").startswith("torch"):
                for a in node.names:
                    alias[a.asname or a.name] = f"{node.module}.{a.name}"

    def _fullname(expr):
        parts = []
        while isinstance(expr, ast.Attribute):
            parts.append(expr.attr)
            expr = expr.value
        if isinstance(expr, ast.Name):
            parts.append(expr.id)
        parts.reverse()
        return ".".join(parts)

    seen = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _fullname(node.func)
        if not name:
            continue
        head, _, rest = name.partition(".")
        if head == "torch":
            full = name
        elif head in alias:
            full = alias[head] + ("." + rest if rest else "")
        else:
            continue
        if full.startswith("torch"):
            seen.setdefault(full)
    return list(seen)


def _line_similarity(a: str, b: str) -> float:
    import difflib
    return round(difflib.SequenceMatcher(None, a.splitlines(), b.splitlines()).ratio() * 100, 1)


EVAL_CACHE = WORKSPACE / "eval_cache"


def _cached_protocol_eval(gen_config: Path, gt_config: Path, gen_handler: Path, gt_handler: Path) -> dict:
    """协议生成结果评测（AST 树编辑距离计算量大，按四文件内容指纹缓存）"""
    EVAL_CACHE.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    for p in (gen_config, gt_config, gen_handler, gt_handler):
        h.update(p.read_bytes())
    cache_file = EVAL_CACHE / f"{h.hexdigest()[:24]}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    from herta.compare import compare_configs, parse_c_functions, match_functions
    config_result = compare_configs(str(gen_config), str(gt_config))
    gen_funcs = parse_c_functions(str(gen_handler))
    gt_funcs = parse_c_functions(str(gt_handler))
    handler_result = match_functions(gen_funcs, gt_funcs)
    overall = config_result.config_similarity * 0.35 + handler_result.handler_similarity * 0.65
    evaluation = {
        "kind": "full",
        "scores": {
            "拓扑结构": round(config_result.topology_similarity.score, 2),
            "命令表": round(config_result.cmd_list_similarity.score, 2),
            "遥测处理": round(config_result.tele_handler_similarity.score, 2),
            "内部处理": round(config_result.internal_handler_similarity.score, 2),
            "Config 综合": round(config_result.config_similarity, 2),
            "Handler 综合": round(handler_result.handler_similarity, 2),
            "整体加权": round(overall, 2),
        },
        "functions": {
            "真实函数": len(gt_funcs), "生成函数": len(gen_funcs),
            "成功匹配": len([m for m in handler_result.matches if m.is_matched]),
        },
    }
    cache_file.write_text(json.dumps(evaluation, ensure_ascii=False), encoding="utf-8")
    return evaluation


def _norm_cmd_token(v: str):
    """命令码归一化：可解析为整数（0x 十六进制/十进制）按整数比较，否则按归一化字符串"""
    s = str(v).strip().upper().replace("0X", "0x")
    try:
        return ("i", int(s, 0))
    except (ValueError, TypeError):
        return ("s", re.sub(r"[^0-9A-Z]", "", s))


def _norm_ident(v: str) -> str:
    """标识符归一化（函数名/信号名模糊匹配用）"""
    return re.sub(r"[\s_\-]+", "", str(v).strip().lower())


def _doc_protocol_eval(doc_path: Path, config_obj: dict, gen_handler: Path) -> Optional[dict]:
    """上传协议（无人工标准答案）的参照评测：
    从上传文档的表格中确定性提取 命令码 / 遥测信号名 / 处理函数名 作为参照，
    对生成产物做覆盖度与结构完整性评测。
    文档中未提取到结构化表格时返回 None（调用方回退为基础结构检查）。"""
    from docx import Document

    doc_cmds, doc_signals, doc_funcs = set(), set(), set()
    try:
        document = Document(str(doc_path))
        for table in document.tables:
            if not table.rows:
                continue
            header = [c.text.strip() for c in table.rows[0].cells]
            htext = " ".join(header)

            def col(exact=(), loose=()):
                for i, h in enumerate(header):
                    if h.lower() in [k.lower() for k in exact]:
                        return i
                for i, h in enumerate(header):
                    if any(k in h for k in loose):
                        return i
                return None

            is_cmd_table = any(k in htext for k in ("命令", "指令", "遥控"))
            is_tele_table = any(k in htext for k in ("位段", "字节", "换算", "遥测", "信号"))
            cmd_col = col(("cmd", "命令码", "指令码", "命令字"), ("命令码", "指令码")) if is_cmd_table else None
            sig_col = col(("信号名称", "参数名称", "信号"), ("信号名称", "参数名称")) if is_tele_table else None
            func_col = col(("处理函数名", "函数名", "handler"), ("函数名", "handler"))

            for row in table.rows[1:]:
                cells = [c.text.strip() for c in row.cells]
                if cmd_col is not None and cmd_col < len(cells) and cells[cmd_col]:
                    doc_cmds.add(_norm_cmd_token(cells[cmd_col]))
                if sig_col is not None and sig_col < len(cells) and cells[sig_col]:
                    doc_signals.add(cells[sig_col].strip())
                if func_col is not None and func_col < len(cells) and cells[func_col]:
                    doc_funcs.add(_norm_ident(cells[func_col]))
    except Exception:
        pass

    if not (doc_cmds or doc_signals or doc_funcs):
        return None

    gen_cmds = {_norm_cmd_token(e.get("cmd")) for e in config_obj.get("cmd_list", [])
                if isinstance(e, dict) and e.get("cmd") is not None}
    gen_tele_entries = [e for e in config_obj.get("tele_handler", []) if isinstance(e, dict)]
    gen_tele = []
    for e in gen_tele_entries:
        gen_tele.extend(str(e.get(k) or "") for k in ("desc", "name", "handler"))
    gen_funcs = set()
    syntax_ok = False
    if gen_handler.exists():
        try:
            from herta.compare import parse_c_functions
            gen_funcs = {_norm_ident(f.name) for f in parse_c_functions(str(gen_handler))}
            syntax_ok = len(gen_funcs) > 0
        except Exception:
            syntax_ok = False

    from rapidfuzz import fuzz

    def _fuzzy_hits(doc_names, gen_names):
        gen_norm = [_norm_ident(g) for g in gen_names if g]
        hits = 0
        for d in doc_names:
            dn = _norm_ident(d)
            if dn and any(dn == gn or dn in gn or gn in dn or fuzz.ratio(dn, gn) >= 88 for gn in gen_norm):
                hits += 1
        return hits

    cmd_hits = len(doc_cmds & gen_cmds)
    tele_hits = _fuzzy_hits(doc_signals, gen_tele)
    func_hits = _fuzzy_hits(doc_funcs, gen_funcs)

    def _f1(hits, doc_n, gen_n):
        """覆盖度 F1：既看文档条目被覆盖的比例，也防止『多生成凑命中』的水分"""
        if not doc_n:
            return None
        p = hits / max(gen_n, 1)
        r = hits / doc_n
        return round(2 * p * r / (p + r) * 100, 1) if (p + r) else 0.0

    def _pct(a, b):
        return round(a / b * 100, 1) if b else None

    scores = {}
    if doc_cmds:
        scores["命令覆盖率"] = _f1(cmd_hits, len(doc_cmds), len(gen_cmds))
    if doc_signals:
        scores["遥测覆盖率"] = _f1(tele_hits, len(doc_signals), len(gen_tele_entries))
    if doc_funcs:
        scores["处理函数覆盖率"] = _f1(func_hits, len(doc_funcs), len(gen_funcs))
    entries = list(config_obj.get("cmd_list", [])) + list(config_obj.get("tele_handler", []))
    if entries:
        ok = sum(1 for e in entries
                 if isinstance(e, dict) and e.get("handler") and (e.get("desc") or e.get("cmd") is not None))
        scores["字段完整性"] = _pct(ok, len(entries))
    if gen_handler.exists():
        scores["Handler 语法可解析"] = 100.0 if syntax_ok else 0.0

    weights = {"命令覆盖率": 0.3, "遥测覆盖率": 0.25, "处理函数覆盖率": 0.3, "字段完整性": 0.15}
    wsum = sum(w for k, w in weights.items() if k in scores)
    if wsum:
        scores["综合（参照文档）"] = round(
            sum(scores[k] * weights[k] for k in weights if k in scores) / wsum, 1)

    return {
        "kind": "document",
        "scores": scores,
        "stats": {
            "文档命令/生成/命中": f"{len(doc_cmds)}/{len(gen_cmds)}/{cmd_hits}",
            "文档遥测/生成/命中": f"{len(doc_signals)}/{len(gen_tele_entries)}/{tele_hits}",
            "文档函数/生成/命中": f"{len(doc_funcs)}/{len(gen_funcs)}/{func_hits}",
        },
        "functions": {"真实函数": len(doc_funcs), "生成函数": len(gen_funcs), "成功匹配": func_hits},
        "note": "无人工标准答案：以「上传文档表格中提取的命令码/遥测信号/处理函数名」为参照的覆盖度评测，"
                "口径不同于预置样例的 ground truth AST 比对，仅供参考",
    }


# ---------------------------------------------------------------------------
# API：基础信息
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/kbs")
def list_kbs():
    """两套知识库卡片（真实扫描数据）"""
    # 协议知识库
    proto = {"id": "protocol", "available": False}
    if HERTA_KB_DIR.exists():
        kbs = []
        total = 0
        for d in sorted(HERTA_KB_DIR.iterdir()):
            meta_file = d / "meta.json"
            if d.is_dir() and meta_file.exists():
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                kbs.append({"source": d.name, "chunks": meta.get("chunk_count", 0)})
                total += meta.get("chunk_count", 0)
        proto.update({
            "available": total > 0, "bases": kbs, "total_chunks": total,
            "form": "协议条目 + 命名规范 + 代码模式 · FAISS 向量索引",
        })
    # ONNX 知识库（PyTorch→ONNX）
    op = {"id": "operator", "available": False}
    kb = _onnx_kb()
    if kb:
        counts = {}
        for rec in kb:
            counts[rec["category_label"]] = counts.get(rec["category_label"], 0) + 1
        op.update({
            "available": True, "total_rules": len(kb), "category_counts": counts,
            "form": "算子映射 / 复合实现 / 测试用例 / 技术文档等 9 类知识",
        })
    return {"kbs": [proto, op]}


@app.get("/api/samples")
def list_samples(kb: str):
    """各入口的预置样例"""
    if kb == "protocol":
        out = []
        if HERTA_TARGET_DIR.exists():
            for d in sorted(HERTA_TARGET_DIR.iterdir()):
                if d.is_dir():
                    out.append({
                        "name": d.name,
                        "label": f"{d.name} 协议（卫星星务 CAN 总线）",
                        "has_gt": (d / f"{d.name}_config.json").exists(),
                        "pre_generated": (HERTA_PRE_GEN / d.name / f"{d.name}_handler.c").exists(),
                    })
        return {"samples": out}
    if kb == "operator":
        out = []
        for fw_dir in [OPALIGN_MODELS / "pytorch"]:
            if not fw_dir.exists():
                continue
            for domain_dir in sorted(fw_dir.iterdir()):
                # 仅展示视觉骨干样例（贴合无人机视觉感知部署场景；语音/文本模型保留在数据集中但不进演示列表）
                if domain_dir.name != "vision":
                    continue
                src_models = domain_dir / "sourceModels"
                if not src_models.exists():
                    continue
                seen = set()
                for entry in sorted(src_models.iterdir()):
                    name = entry.stem if entry.is_file() else entry.name
                    if name == "__init__" or name in seen:
                        continue
                    seen.add(name)
                    pre = (ONNX_PRESET_DIR / f"{name}_to_onnx.py").exists()
                    out.append({
                        "name": name, "domain": domain_dir.name,
                        "id": f"pytorch/{domain_dir.name}/{name}",
                        "label": f"{name}（{domain_dir.name}）",
                        "pre_generated": pre,
                    })
        return {"samples": out}
    raise HTTPException(status_code=400, detail="未知知识库")


# ---------------------------------------------------------------------------
# API：上传
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_file(kb: str, file: UploadFile = File(...)):
    """上传待处理文件：protocol → .docx；operator → .py"""
    ext = Path(file.filename or "").suffix.lower()
    if kb == "protocol" and ext != ".docx":
        raise HTTPException(status_code=400, detail="协议代码生成入口仅支持 .docx 协议文档")
    if kb == "operator" and ext != ".py":
        raise HTTPException(status_code=400, detail="模型部署适配入口仅支持 .py 模型源码")
    if kb not in ("protocol", "operator"):
        raise HTTPException(status_code=400, detail="未知知识库")

    safe = _safe_name(Path(file.filename).stem)
    upload_id = f"{safe}_{uuid.uuid4().hex[:6]}"
    dest_dir = UPLOAD_DIR / kb / upload_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{safe}{ext}"
    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="文件超过 20MB 限制")
    dest.write_bytes(content)
    return {"upload_id": upload_id, "filename": f"{safe}{ext}", "size": len(content)}


# ---------------------------------------------------------------------------
# API：生成（后台任务）
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    kb: str                       # protocol | operator
    source: str                   # 样例名 或 upload_id
    mode: str = "sample"          # sample | upload
    top_k: int = 30


@app.post("/api/generate")
def generate(req: GenerateRequest):
    task_id = _task_create(req.kb)
    if req.kb == "protocol":
        target = _run_protocol
    elif req.kb == "operator":
        target = _run_operator
    else:
        raise HTTPException(status_code=400, detail="未知知识库")
    threading.Thread(target=target, args=(task_id, req), daemon=True).start()
    return {"task_id": task_id, "status": "pending"}


@app.get("/api/task/{task_id}")
def task_status(task_id: str):
    t = tasks.get(task_id)
    if not t:
        return {"exists": False}
    return {"exists": True, **t}


@app.get("/api/result/{task_id}")
def task_result(task_id: str):
    t = tasks.get(task_id)
    if not t:
        raise HTTPException(status_code=404, detail="任务不存在")
    if t["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"任务未完成: {t['status']}")
    return t["result"]


# ---------------------------------------------------------------------------
# 入口一流水线：协议代码生成（herta 引擎）
# ---------------------------------------------------------------------------

def _run_protocol(task_id: str, req: GenerateRequest):
    try:
        _task_update(task_id, status="running", progress=5, message="解析目标协议文档…")

        # 输入定位
        if req.mode == "upload":
            up = UPLOAD_DIR / "protocol" / req.source
            docs = list(up.glob("*.docx"))
            if not docs:
                raise RuntimeError("上传文件不存在")
            doc_path = docs[0]
            target_name = req.source
            gt_config = None
            gt_handler = None
        else:
            name = _safe_name(req.source)
            doc_path = HERTA_TARGET_DIR / name / f"{name}.docx"
            if not doc_path.exists():
                docs = sorted((HERTA_TARGET_DIR / name).glob("*.docx"))
                doc_path = docs[0] if docs else None
            if doc_path is None:
                raise RuntimeError(f"样例 {name} 的协议文档不存在")
            target_name = name
            gt_config = HERTA_TARGET_DIR / name / f"{name}_config.json"
            gt_handler = HERTA_TARGET_DIR / name / f"{name}_handler.c"

        kb_dirs = [d for d in sorted(HERTA_KB_DIR.iterdir()) if d.is_dir()] if HERTA_KB_DIR.exists() else []
        if not kb_dirs:
            raise RuntimeError("协议知识库不存在")
        kb_dir = kb_dirs[0]

        _task_intermediate(task_id, parsed={
            "目标协议": target_name,
            "输入方式": "上传文档" if req.mode == "upload" else "预置样例",
            "标准答案": "有（AST 比对评测）" if (gt_config and gt_config.exists()) else "无（结构完整性检查）",
        })

        out_dir = GEN_DIR / task_id
        out_dir.mkdir(parents=True, exist_ok=True)
        pre_generated = False

        pre = HERTA_PRE_GEN / req.source
        if req.mode == "sample" and pre.exists():
            # 预置样例：永远走预生成回放（快且稳），不调 LLM
            pre_generated = True
            _task_update(task_id, progress=30, message="载入预生成结果（真实流水线历史产物）…")
            _task_intermediate(task_id, gen_mode="预生成回放（真实流水线历史产物）")
            shutil.copytree(pre, out_dir, dirs_exist_ok=True)
            time.sleep(0.5)
            _task_update(task_id, progress=60, message="载入预生成结果…")
            time.sleep(0.5)
        elif _llm_available():
            # 上传的自有 case：真实生成（解析 → 检索 → LLM 两阶段生成）
            from herta.generator.config_generator import ConfigGenerator
            from herta.generator.handler_generator import HandlerGenerator

            _task_update(task_id, progress=15, message="检索协议知识库…")
            _task_update(task_id, progress=30, message="LLM 生成配置（真实调用，约 2–3 分钟）…")
            cfg_result = ConfigGenerator().generate(
                target_doc_path=doc_path, source_kb_dir=kb_dir,
                target_name=target_name, top_k=req.top_k, output_dir=out_dir,
            )
            if cfg_result.error_message and not cfg_result.config:
                raise RuntimeError(f"Config 生成失败: {cfg_result.error_message}")
            cfg_stats = {}
            if cfg_result.config:
                cfg_stats = {
                    "命令条目数": len(cfg_result.config.get("cmd_list", []) or []),
                    "遥测条目数": len(cfg_result.config.get("tele_handler", []) or []),
                }
            _task_intermediate(
                task_id, gen_mode="LLM 在线生成",
                retrieval={"协议知识库注入知识块": cfg_result.retrieved_chunk_count},
                config_stats=cfg_stats,
            )

            _task_update(task_id, progress=60, message="LLM 生成处理代码（真实调用，两阶段约 5–6 分钟）…")
            HandlerGenerator().generate(
                target_doc_path=doc_path, source_kb_dir=kb_dir,
                target_name=target_name, top_k=req.top_k, output_dir=out_dir,
                save_plan=True, intermediate_dir=out_dir / "intermediate",
            )
        else:
            raise RuntimeError("上传的自有文件需要配置 LLM API Key（顶栏齿轮设置）；预置样例可直接演示")

        # 收集产物
        gen_config = out_dir / f"{target_name}_config.json"
        gen_handler = out_dir / f"{target_name}_handler.c"
        files = []
        config_obj = None
        if gen_config.exists():
            config_obj = json.loads(gen_config.read_text(encoding="utf-8"))
            files.append({"name": f"{target_name}_config.json", "language": "json",
                          "content": json.dumps(config_obj, ensure_ascii=False, indent=2)})
        if gen_handler.exists():
            files.append({"name": f"{target_name}_handler.c", "language": "c",
                          "content": gen_handler.read_text(encoding="utf-8", errors="replace")})
        plan_file = out_dir / "intermediate" / "handler_plan.md"
        if not plan_file.exists():
            plan_file = out_dir / "handler_plan.md"
        if plan_file.exists():
            files.insert(0, {"name": "handler_plan.md（设计草图）", "language": "markdown",
                             "content": plan_file.read_text(encoding="utf-8", errors="replace")})

        # 评测（纯本地计算，始终真实执行；TED 计算量大，按内容指纹缓存）
        evaluation = None
        _task_update(task_id, progress=85, message="评测生成结果（本地计算）…")
        if gt_config and gt_config.exists() and gen_config.exists() and gt_handler.exists() and gen_handler.exists():
            evaluation = _cached_protocol_eval(gen_config, gt_config, gen_handler, gt_handler)
        elif config_obj:
            # 无 ground truth：优先以「上传文档表格提取」为参照做覆盖度评测，提取不到再回退结构检查
            evaluation = _doc_protocol_eval(doc_path, config_obj, gen_handler) or {
                "kind": "structural",
                "scores": {
                    "JSON 结构合法": 100.0,
                    "顶层字段齐备": 100.0 if all(
                        k in config_obj for k in ("cmd_list", "tele_handler")
                    ) else 0.0,
                },
                "stats": {
                    "命令条目数": len(config_obj.get("cmd_list", [])),
                    "遥测条目数": len(config_obj.get("tele_handler", [])),
                },
                "note": "上传的新协议无 ground truth，仅做结构完整性检查",
            }

        _task_update(task_id, status="completed", progress=100, message="完成", result={
            "kb": "protocol", "target": target_name, "files": files,
            "evaluation": evaluation, "pre_generated": pre_generated,
        })
    except Exception as e:
        _task_update(task_id, status="failed", error=str(e), message=f"失败: {e}")


# ---------------------------------------------------------------------------
# 入口二流水线：模型部署适配（算子映射知识库检索 + LLM 生成）
# ---------------------------------------------------------------------------

def _build_onnx_prompt(source_text: str, analysis: dict, retrieved: list) -> list:
    """组装 PyTorch→ONNX 导出代码的 LLM 请求（检索到的 ONNX 知识注入）"""
    ops_text = "、".join(f"`{c}`" for c in analysis["calls"][:20]) or "（未检出）"
    knowledge_blocks = []
    for i, rec in enumerate(retrieved, 1):
        knowledge_blocks.append(
            f"【知识 {i}】{rec['category_label']} · {rec['title']}（来源: {rec['source']}）\n"
            f"{rec['content'][:1200]}"
        )
    knowledge_text = "\n\n".join(knowledge_blocks) if knowledge_blocks else "（未检索到相关知识）"

    system = (
        "你是一位资深的模型部署工程师，精通 PyTorch 模型导出 ONNX 的完整链路"
        "（torch.onnx.export、ATen 算子到 ONNX 算子的映射、自定义 symbolic、opset 版本选择、导出校验）。"
        "你的任务是为给定的 PyTorch 模型编写完整可用的 ONNX 导出代码。"
    )
    user = f"""# PyTorch → ONNX 导出代码生成

## 模型源码中检出的算子
{ops_text}

## 从 ONNX 知识库检索到的相关知识（请参考其映射写法与处理方式）
{knowledge_text}

## 编写要求
1. 使用 `torch.onnx.export`（或 `torch.onnx.dynamo_export`）导出模型，权重随模型加载
2. 模型类定义直接从源文件导入（假设导出脚本与源文件同目录运行），构造合理的示例输入
3. 标注清晰的 input_names / output_names / dynamic_axes（支持动态 batch）
4. 对知识库中标明需要特殊处理的算子，给出自定义 symbolic 或替代写法，并附注释说明
5. 导出后用 `onnx.checker.check_model` 校验，并打印导出信息
6. 代码可直接运行，含必要的注释

## 源模型代码（PyTorch）
```python
{source_text}
```

## 输出要求
仅输出完整的 Python 导出脚本，不要输出解释文字或 markdown 围栏。
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _clean_code(text: str) -> str:
    """剥离 LLM 输出可能的 markdown 围栏"""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t.rstrip())
    return t


def _locate_operator_source(req: GenerateRequest):
    """返回 (源文件/目录 Path, 显示名, 预置导出脚本 或 None)"""
    if req.mode == "upload":
        up = UPLOAD_DIR / "operator" / req.source
        pys = list(up.glob("*.py"))
        if not pys:
            raise RuntimeError("上传文件不存在")
        return pys[0], pys[0].stem, None
    parts = req.source.split("/")
    if len(parts) != 3:
        raise RuntimeError("样例 id 格式错误")
    _, domain, name = parts
    src_dir = OPALIGN_MODELS / "pytorch" / domain / "sourceModels" / name
    src = src_dir if src_dir.is_dir() else src_dir.with_suffix(".py")
    if not src.exists():
        raise RuntimeError("样例源码不存在")
    preset = ONNX_PRESET_DIR / f"{name}_to_onnx.py"
    return src, name, (preset if preset.exists() else None)


def _eval_onnx(gen_code: str, analysis: dict, retrieved: list) -> dict:
    """评测生成的 ONNX 导出代码：语法、导出调用完备性、知识覆盖"""
    try:
        ast.parse(gen_code)
        syntax_valid = True
    except SyntaxError:
        syntax_valid = False
    has_export = ("torch.onnx.export" in gen_code) or ("dynamo_export" in gen_code) or ("onnxscript" in gen_code)
    has_checker = "check_model" in gen_code or "onnx.checker" in gen_code
    custom_symbolics = len(re.findall(r"custom_symbolic|onnx_symbolic|def .*symbolic", gen_code))
    dynamic_axes = "dynamic_axes" in gen_code

    return {
        "kind": "operator",
        "scores": {
            "语法合法": 100.0 if syntax_valid else 0.0,
            "含导出调用": 100.0 if has_export else 0.0,
            "导出后自检": 100.0 if has_checker else 0.0,
        },
        "syntax_valid": syntax_valid,
        "has_export_call": has_export,
        "has_checker": has_checker,
        "dynamic_axes": dynamic_axes,
        "custom_symbolics": custom_symbolics,
        "total_calls": analysis["total_calls"],
        "retrieved_count": len(retrieved),
        "retrieved": [
            {"title": r["title"], "category_label": r["category_label"], "score": r["score"], "source": r["source"]}
            for r in retrieved
        ],
    }


def _run_operator(task_id: str, req: GenerateRequest):
    try:
        _task_update(task_id, status="running", progress=5, message="解析模型源码…")
        src, name, preset_file = _locate_operator_source(req)

        if src.is_dir():
            texts = [p.read_text(encoding="utf-8", errors="replace") for p in sorted(src.rglob("*.py"))]
            source_text = "\n\n".join(texts)
        else:
            source_text = src.read_text(encoding="utf-8", errors="replace")

        _task_update(task_id, progress=20, message="检索 ONNX 知识库…")
        analysis = _analyze_source(source_text)
        # 以模型名 + 算子叶子名为查询，本地 BM25 检索（离线）
        leaf_ops = sorted({c.split(".")[-1] for c in analysis["calls"]})
        query = f"{name} PyTorch ONNX export " + " ".join(leaf_ops[:12])
        retrieved = _onnx_retrieve(query, top_k=8)
        _task_intermediate(
            task_id,
            operators={"total_calls": analysis["total_calls"], "leaf_ops": leaf_ops},
            retrieved=[{"title": r["title"], "category_label": r["category_label"],
                        "score": r["score"], "source": r["source"]} for r in retrieved],
        )

        pre_generated = False
        if req.mode == "sample" and preset_file:
            # 预置样例：永远走预生成回放（真实流水线的历史产物），不调 LLM
            pre_generated = True
            _task_update(task_id, progress=45, message="载入预生成导出脚本（真实流水线历史产物）…")
            _task_intermediate(task_id, gen_mode="预生成回放（真实流水线历史产物）")
            gen_code = preset_file.read_text(encoding="utf-8", errors="replace")
            time.sleep(0.5)
        elif _llm_available():
            # 上传的自有 case：检索 → LLM 真实生成
            _task_intermediate(task_id, gen_mode="LLM 在线生成")
            _task_update(task_id, progress=45, message=f"LLM 生成 ONNX 导出代码（真实调用，约 1–2 分钟，注入 {len(retrieved)} 条知识）…")
            from herta.config import get_settings
            from herta.llm.client import BigModelClient
            messages = _build_onnx_prompt(source_text, analysis, retrieved)
            raw = BigModelClient().chat(
                messages=messages, model=get_settings().chat_model, temperature=0.1,
            )
            gen_code = _clean_code(raw)
        else:
            raise RuntimeError("上传的自有文件需要配置 LLM API Key（顶栏齿轮设置）；预置样例可直接演示")

        _task_update(task_id, progress=85, message="评测生成代码…")
        evaluation = _eval_onnx(gen_code, analysis, retrieved)
        if not pre_generated and preset_file:
            evaluation["line_similarity_vs_reference"] = _line_similarity(
                gen_code, preset_file.read_text(encoding="utf-8", errors="replace")
            )

        _task_update(task_id, status="completed", progress=100, message="完成", result={
            "kb": "operator", "target": name,
            "files": [{"name": f"{name}_to_onnx.py", "language": "python", "content": gen_code}],
            "evaluation": evaluation, "pre_generated": pre_generated,
        })
    except Exception as e:
        _task_update(task_id, status="failed", error=str(e), message=f"失败: {e}")


# ---------------------------------------------------------------------------
# API：知识库浏览（两套知识库内容查看）
# ---------------------------------------------------------------------------

@app.get("/api/kb/protocol/chunks")
def protocol_chunks(query: str = "", limit: int = 100):
    """协议知识库知识块浏览（读 chunks.jsonl，本地过滤）"""
    chunks_file = None
    if HERTA_KB_DIR.exists():
        for d in sorted(HERTA_KB_DIR.iterdir()):
            if d.is_dir() and (d / "chunks.jsonl").exists():
                chunks_file = d / "chunks.jsonl"
                source = d.name
                break
    if chunks_file is None:
        return {"chunks": [], "total": 0, "source": ""}
    q = query.strip().lower()
    items, total = [], 0
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            total += 1
            text = c.get("text", "")
            if q and q not in text.lower() and q not in str(c.get("chunk_id", "")).lower():
                continue
            if len(items) < limit:
                items.append({
                    "chunk_id": c.get("chunk_id", ""),
                    "kind": c.get("kind", ""),
                    "file": str(c.get("file", "")).replace("\\", "/").split("/")[-1],
                    "token_count": c.get("token_count", 0),
                    "text_preview": text[:400],
                })
    return {"chunks": items, "total": total, "source": source}


@app.get("/api/kb/onnx/records")
def onnx_records(query: str = "", category: str = "", limit: int = 100):
    """ONNX 知识库记录浏览（可按类别与关键词过滤）"""
    q = query.strip().lower()
    items = []
    total = 0
    for rec in _onnx_kb():
        if category and rec["category"] != category:
            continue
        total += 1
        if q and q not in rec["title"].lower() and q not in rec["content"].lower() and q not in rec["source"].lower():
            continue
        if len(items) < limit:
            items.append({
                "id": rec["id"],
                "title": rec["title"],
                "category": rec["category"],
                "category_label": rec["category_label"],
                "source": rec["source"],
                "preview": rec["content"][:400],
            })
    return {"records": items, "total": total}


# ---------------------------------------------------------------------------
# API：设置（进程内生效）
# ---------------------------------------------------------------------------

class SettingsRequest(BaseModel):
    api_key: str = ""
    chat_model: str = ""
    chat_provider: str = ""
    chat_base_url: str = ""
    chat_api_key: str = ""


@app.get("/api/settings")
def get_settings_api():
    return {
        "llm_available": _llm_available(),
        "chat_model": os.environ.get("HERTA_CHAT_MODEL", "glm-5.1"),
    }


@app.post("/api/settings")
def update_settings(req: SettingsRequest):
    mapping = {
        "ZAI_API_KEY": req.api_key,
        "HERTA_CHAT_MODEL": req.chat_model,
        "HERTA_CHAT_PROVIDER": req.chat_provider,
        "HERTA_CHAT_BASE_URL": req.chat_base_url,
        "HERTA_CHAT_API_KEY": req.chat_api_key,
    }
    for k, v in mapping.items():
        if v:
            os.environ[k] = v
    try:
        from herta.config import reload_settings
        reload_settings()
    except Exception:
        pass
    return {"status": "ok", "llm_available": _llm_available()}
