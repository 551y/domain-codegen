"""
基于RAG的代码生成基本框架 — 集成演示编排后端（domain-codegen）

一个后端统一编排两套知识库的代码生成流水线：
  - 入口一（星务协议知识库）：协议文档 → 检索增强 → LLM 生成 config/handler → 本地评测
  - 入口二（模型部署适配，ONNX 知识库与 ModelX Paddle 映射规则独立维护）：
      PyTorch→ONNX：BM25 检索 → LLM 生成导出代码 → 子进程真实执行导出（产出可下载的单文件 .onnx）
      PyTorch→Paddle：ModelX CMC AST/规则转换 → 迁移彻底性/结构评测 → Paddle 冒烟检查

引擎复用：herta（LLM 生成与评测）、opalign（样例模型）。本文件只做编排与演示适配。
"""

import ast
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import tokenize
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

try:
    from paddle_converter import ConversionResult, convert_pytorch_source, paddle_demo_cases, run_paddle_smoke_test
except ModuleNotFoundError:  # importing as ``server.app`` in tests/tools
    from server.paddle_converter import ConversionResult, convert_pytorch_source, paddle_demo_cases, run_paddle_smoke_test

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
ONNX_DATASET = VENDOR_DIR / "onnx" / "dataset.expanded.jsonl"
ONNX_PRESET_DIR = PLATFORM_ROOT / "server" / "preset_onnx"
PADDLE_PRESET_DIR = PLATFORM_ROOT / "server" / "preset_paddle"
PADDLE_DATASET = VENDOR_DIR / "modelx" / "paddle_dataset_view.json"

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
            "kind": kind, "status": "pending", "progress": 0, "stage": 0,
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
    if "evaluation_steps" in kw:
        # Publish snapshots; the worker mutates its local step list later.
        kw["evaluation_steps"] = [dict(step) for step in kw["evaluation_steps"]]
    with _tasks_lock:
        t = tasks.get(task_id)
        if t is not None:
            t.setdefault("intermediate", {}).update(kw)


def _live_done(task_id: str, key: str):
    """标记一路实时输出完成（停止前端的转动/闪烁，保留已生成文本）"""
    with _tasks_lock:
        t = tasks.get(task_id)
        live = (t or {}).get("intermediate", {}).get(key)
        if isinstance(live, dict):
            live["done"] = True


def _gen_failure(result) -> Optional[str]:
    """识别生成器「吞错误」的返回：herta 生成器把解析/检索/LLM 异常包装进结果对象
    （error_message 非空、is_valid_json/is_valid=False），不抛异常——编排层必须显式检查，
    否则错误结果会被当作成功继续交付（产物文件缺失、任务却标记完成）。"""
    if result is None:
        return "生成器未返回结果"
    msg = getattr(result, "error_message", None)
    if msg:
        return str(msg)
    valid = getattr(result, "is_valid_json", getattr(result, "is_valid", True))
    if valid is False:
        return "生成结果无效（is_valid=False）"
    return None


def _live_sink(task_id: str, keys: list, stage_names: list, on_begin=None):
    """生成线程内的流式输出回调：聚合增量文本，节流后写入任务 intermediate。

    每次 LLM 调用（kind="begin"）占用 keys/stage_names 中对应的一路，独立成框展示；
    新调用开始时先把上一路**完整落盘**并标记完成再切换，避免后一次调用的输出覆盖前一次
    （handler 链有 设计草图→完整代码 两次调用，完整代码不再覆盖设计草图）。
    kind="flush"：一次调用结束，把尚未到节流窗口的尾部增量冲刷落盘（否则结尾显示不完整）。
    on_begin(call_no)：可选，每次 LLM 调用真正开始时回调（首次 begin 即代表
    生成器内部的检索阶段已结束，可用于推进任务阶段显示）。
    """
    MAX_CHARS = 300_000  # 安全上限，正常产物（数百行代码/配置）远低于此
    state = {"call": -1, "buf": "", "last": 0.0}

    def _idx() -> int:
        return min(state["call"], len(keys) - 1)

    def _flush():
        if state["call"] >= 0:
            _task_intermediate(task_id, **{
                keys[_idx()]: {"stage": stage_names[min(_idx(), len(stage_names) - 1)],
                               "text": state["buf"][-MAX_CHARS:]},
            })

    def sink(kind: str, text: str):
        if kind == "begin":
            if state["call"] >= 0:
                _flush()
                _live_done(task_id, keys[_idx()])
            state["call"] += 1
            state["buf"] = ""
            _task_intermediate(task_id, **{
                keys[_idx()]: {"stage": stage_names[min(_idx(), len(stage_names) - 1)], "text": ""},
            })
            if on_begin is not None:
                try:
                    on_begin(state["call"])
                except Exception:
                    pass
            return
        if kind == "flush":
            _flush()
            return
        if state["call"] < 0:
            return
        state["buf"] += text
        now = time.monotonic()
        if now - state["last"] >= 0.4:
            state["last"] = now
            _flush()

    return sink


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
    """加载 ONNX 知识库（1349 条，pytorch-onnx-dataset2 交付版 expanded 数据集）"""
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
    """协议生成结果评测（AST 树编辑距离计算量大，按四文件内容指纹缓存；v2 含维度问题明细与函数匹配明细）"""
    EVAL_CACHE.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    for p in (gen_config, gt_config, gen_handler, gt_handler):
        h.update(p.read_bytes())
    cache_file = EVAL_CACHE / f"{h.hexdigest()[:24]}.v2.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    from herta.compare import compare_configs, parse_c_functions, match_functions
    config_result = compare_configs(str(gen_config), str(gt_config))
    gen_funcs = parse_c_functions(str(gen_handler))
    gt_funcs = parse_c_functions(str(gt_handler))
    handler_result = match_functions(gen_funcs, gt_funcs)
    overall = config_result.config_similarity * 0.35 + handler_result.handler_similarity * 0.65

    matched = [m for m in handler_result.matches if m.is_matched]
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
        # 各维度的问题明细（缺失字段/空列表等，可直接指导修正）
        "issues": {
            "拓扑结构": config_result.topology_similarity.issues,
            "命令表": config_result.cmd_list_similarity.issues,
            "遥测处理": config_result.tele_handler_similarity.issues,
            "内部处理": config_result.internal_handler_similarity.issues,
        },
        "functions": {
            "真实函数": len(gt_funcs), "生成函数": len(gen_funcs),
            "成功匹配": len(matched),
            "其中同名": len([m for m in matched if m.is_name_match]),
            "未覆盖真实函数": len(handler_result.unmatched_gt_functions),
            "多生成函数": len(handler_result.unmatched_gen_functions),
        },
        # 逐函数匹配明细（按 AST 相似度降序）
        "function_matches": [
            {
                "gt": m.gt_func_name, "gen": m.gen_func_name,
                "matched": m.is_matched, "name_match": m.is_name_match,
                "ast": round(m.ast_similarity * 100, 1),
                "tree_edit": round(m.tree_edit_similarity * 100, 1),
                "semantic": round(m.semantic_feature_similarity * 100, 1),
            }
            for m in sorted(handler_result.matches, key=lambda m: -m.ast_similarity)
        ],
        "unmatched_gt_functions": handler_result.unmatched_gt_functions,
        "unmatched_gen_functions": handler_result.unmatched_gen_functions,
        "match_note": "函数匹配：以 AST 相似度（树编辑距离 + 语义特征加权）构建真实函数 × 生成函数的相似度矩阵，"
                      "用匈牙利算法求一对一最优分配；相似度 ≥ 30% 判为成功匹配",
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

    checks = {}
    if gen_handler.exists():
        checks["Handler 语法可解析"] = syntax_ok

    weights = {"命令覆盖率": 0.3, "遥测覆盖率": 0.25, "处理函数覆盖率": 0.3, "字段完整性": 0.15}
    wsum = sum(w for k, w in weights.items() if k in scores)
    if wsum:
        scores["综合（参照文档）"] = round(
            sum(scores[k] * weights[k] for k in weights if k in scores) / wsum, 1)

    return {
        "kind": "document",
        "scores": scores,
        "checks": checks,
        "stats": {
            "文档命令/生成/命中": f"{len(doc_cmds)}/{len(gen_cmds)}/{cmd_hits}",
            "文档遥测/生成/命中": f"{len(doc_signals)}/{len(gen_tele_entries)}/{tele_hits}",
            "文档函数/生成/命中": f"{len(doc_funcs)}/{len(gen_funcs)}/{func_hits}",
        },
        "functions": {"真实函数": len(doc_funcs), "生成函数": len(gen_funcs), "成功匹配": func_hits},
        "note": "无人工标准答案：以「上传文档表格中提取的命令码/遥测信号/处理函数名」为参照的覆盖度评测，"
                "口径不同于预置样例的 ground truth AST 比对，仅供参考",
    }


def _match_ground_truth(doc_path: Path):
    """上传文档匹配目标协议库的标准答案：先按 docx 内容指纹，再按文件名。
    匹配成功则上传路径与离线预置走同一套 ground truth AST 比对评测。"""
    if not HERTA_TARGET_DIR.exists():
        return None, None
    doc_hash = hashlib.sha256(doc_path.read_bytes()).hexdigest()
    stem = doc_path.stem.lower()
    for d in sorted(HERTA_TARGET_DIR.iterdir()):
        if not d.is_dir():
            continue
        gt_c = d / f"{d.name}_config.json"
        gt_h = d / f"{d.name}_handler.c"
        if not (gt_c.exists() and gt_h.exists()):
            continue
        for ref in d.glob("*.docx"):
            if hashlib.sha256(ref.read_bytes()).hexdigest() == doc_hash:
                return gt_c, gt_h
        if stem == d.name.lower():
            return gt_c, gt_h
    return None, None


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


def _vision_samples(preset_dir: Path, preset_suffix: str) -> list:
    """视觉骨干样例列表（PyTorch 源码 + 预置产物标记）"""
    paddle_cases = paddle_demo_cases() if preset_suffix == "_to_paddle.py" else {}
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
                item = {
                    "name": name, "domain": domain_dir.name,
                    "id": f"pytorch/{domain_dir.name}/{name}",
                    "label": f"{name}（{domain_dir.name}）",
                    "pre_generated": (preset_dir / f"{name}{preset_suffix}").exists(),
                }
                if name in paddle_cases:
                    item["demo_label"] = paddle_cases[name]["label"]
                    item["demo_available"] = True
                    if paddle_cases[name].get("modelx_variant"):
                        item["modelx_variant"] = paddle_cases[name]["modelx_variant"]
                out.append(item)
    return out


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
        return {"samples": _vision_samples(ONNX_PRESET_DIR, "_to_onnx.py")}
    if kb == "paddle":
        return {"samples": _vision_samples(PADDLE_PRESET_DIR, "_to_paddle.py")}
    raise HTTPException(status_code=400, detail="未知知识库")


# ---------------------------------------------------------------------------
# API：上传
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_file(kb: str, file: UploadFile = File(...)):
    """上传待处理文件：protocol → .docx；operator / paddle → .py"""
    ext = Path(file.filename or "").suffix.lower()
    if kb == "protocol" and ext != ".docx":
        raise HTTPException(status_code=400, detail="协议代码生成入口仅支持 .docx 协议文档")
    if kb in ("operator", "paddle") and ext != ".py":
        raise HTTPException(status_code=400, detail="模型类入口仅支持 .py 模型源码（PyTorch）")
    if kb not in ("protocol", "operator", "paddle"):
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
    kb: str                       # protocol | operator | paddle
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
    elif req.kb == "paddle":
        target = _run_paddle
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


@app.get("/api/result/{task_id}/onnx")
def download_onnx(task_id: str):
    """下载任务产出的 .onnx 模型文件（单文件，权重已内嵌）"""
    t = tasks.get(task_id)
    if not t or t["status"] != "completed":
        raise HTTPException(status_code=404, detail="结果不存在")
    om = (t["result"] or {}).get("onnx_model")
    if not om:
        raise HTTPException(status_code=404, detail="该结果无 ONNX 模型文件")
    path = GEN_DIR / task_id / om["name"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="模型文件已被清理")
    return FileResponse(path, media_type="application/octet-stream", filename=om["name"])


# ---------------------------------------------------------------------------
# 入口一流水线：协议代码生成（herta 引擎）
# ---------------------------------------------------------------------------

def _run_protocol(task_id: str, req: GenerateRequest):
    try:
        _task_update(task_id, status="running", progress=5, stage=0, message="解析目标协议文档…")

        # 输入定位
        if req.mode == "upload":
            up = UPLOAD_DIR / "protocol" / req.source
            docs = list(up.glob("*.docx"))
            if not docs:
                raise RuntimeError("上传文件不存在")
            doc_path = docs[0]
            target_name = req.source
            # 上传文档若与协议库中已有目标一致（如 dllg），挂上其标准答案，统一评测体系
            gt_config, gt_handler = _match_ground_truth(doc_path)
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
            "标准答案": "有（AST 比对评测）" if (gt_config and gt_config.exists()) else "无（参照文档覆盖度评测）",
        })

        out_dir = GEN_DIR / task_id
        out_dir.mkdir(parents=True, exist_ok=True)
        pre_generated = False

        pre = HERTA_PRE_GEN / req.source
        if req.mode == "sample" and pre.exists():
            # 预置样例：永远走预生成回放（快且稳），不调 LLM
            pre_generated = True
            _task_update(task_id, progress=30, stage=2, message="载入预生成结果（真实流水线历史产物）…")
            _task_intermediate(task_id, gen_mode="预生成回放（真实流水线历史产物）")
            shutil.copytree(pre, out_dir, dirs_exist_ok=True)
            time.sleep(0.5)
            _task_update(task_id, progress=60, stage=2, message="载入预生成结果…")
            time.sleep(0.5)
        elif _llm_available():
            # 上传的自有 case：真实生成（解析 → 检索 → LLM 生成）
            # config 与 handler 两条生成链相互独立（均只依赖文档+知识库，互不读取对方产物），
            # 并行执行可将 wall-clock 从两者之和降到两者之较长者
            from herta.generator.config_generator import ConfigGenerator
            from herta.generator.handler_generator import HandlerGenerator
            from herta.llm.client import set_stream_sink

            _task_update(task_id, progress=15, stage=1, message="检索协议知识库…")
            _task_intermediate(task_id, gen_mode="LLM 在线生成")

            cfg_holder: dict = {}
            hdl_holder: dict = {}
            err_holder: dict = {}

            def _on_llm_begin(_call_no: int):
                # 任一条链的首次 LLM 调用开始，说明其内部的知识库检索（RAG 注入前的
                # 向量检索）已真实完成，全局阶段此时才推进到「生成目标代码」；
                # 带 stage 守卫，后续 begin（如 handler 的第二次调用）不会回退进度
                with _tasks_lock:
                    cur = tasks.get(task_id, {}).get("stage", 0)
                if cur < 2:
                    _task_update(task_id, progress=30, stage=2,
                                 message="LLM 并行生成配置与处理代码（真实调用，通常数分钟）…")

            def _gen_config():
                sink = _live_sink(task_id, ["live_config"], ["配置生成 config.json"],
                                  on_begin=_on_llm_begin)
                set_stream_sink(sink)
                try:
                    cfg_holder["r"] = ConfigGenerator().generate(
                        target_doc_path=doc_path, source_kb_dir=kb_dir,
                        target_name=target_name, top_k=req.top_k, output_dir=out_dir,
                    )
                    sink("flush", "")
                except Exception as e:
                    err_holder["config"] = e

            def _gen_handler():
                sink = _live_sink(task_id, ["live_handler_plan", "live_handler"],
                                  ["处理代码 · 设计草图", "处理代码 · 完整代码"], on_begin=_on_llm_begin)
                set_stream_sink(sink)
                try:
                    hdl_holder["r"] = HandlerGenerator().generate(
                        target_doc_path=doc_path, source_kb_dir=kb_dir,
                        target_name=target_name, top_k=req.top_k, output_dir=out_dir,
                        save_plan=True, intermediate_dir=out_dir / "intermediate",
                    )
                    sink("flush", "")
                except Exception as e:
                    err_holder["handler"] = e

            # 两条链各自先检索知识库（生成器内部），阶段保持在「检索知识库」，
            # 直到任一链首次 begin 回调（_on_llm_begin）才推进到生成阶段
            t_cfg = threading.Thread(target=_gen_config, daemon=True)
            t_hdl = threading.Thread(target=_gen_handler, daemon=True)
            t_cfg.start()
            t_hdl.start()

            # config 通常先完成：先展示其中间产物，handler 继续跑
            t_cfg.join()
            cfg_result = cfg_holder.get("r")
            if "config" not in err_holder and _gen_failure(cfg_result):
                # 生成器把失败吞进了返回值（模型格式 glitch / 检索或调用失败等）：自动重试一次
                _task_update(task_id, progress=32, stage=2,
                             message=f"配置生成失败（{_gen_failure(cfg_result)}），自动重试一次…")
                cfg_holder.clear()
                t_cfg = threading.Thread(target=_gen_config, daemon=True)
                t_cfg.start()
                t_cfg.join()
            if "config" in err_holder:
                raise RuntimeError(f"Config 生成失败: {err_holder['config']}")
            cfg_err = _gen_failure(cfg_holder.get("r"))
            if cfg_err:
                raise RuntimeError(f"Config 生成失败: {cfg_err}")
            cfg_result = cfg_holder["r"]
            _live_done(task_id, "live_config")
            cfg_stats = {}
            if cfg_result.config:
                cfg_stats = {
                    "命令条目数": len(cfg_result.config.get("cmd_list", []) or []),
                    "遥测条目数": len(cfg_result.config.get("tele_handler", []) or []),
                }
            retrieval_info: dict = {"协议知识库注入知识块": cfg_result.retrieved_chunk_count}
            if not cfg_result.retrieved_chunk_count:
                retrieval_info["提示"] = "检索注入为 0：embedding 调用可能失败（检查 Key 权限/网络），生成质量会受影响"
            _task_intermediate(task_id, retrieval=retrieval_info, config_stats=cfg_stats)
            _task_update(task_id, progress=55, stage=2, message="配置已生成，处理代码生成中…")

            t_hdl.join()
            _live_done(task_id, "live_handler_plan")
            _live_done(task_id, "live_handler")
            if "handler" in err_holder:
                raise RuntimeError(f"Handler 生成失败: {err_holder['handler']}")
            hdl_err = _gen_failure(hdl_holder.get("r"))
            if hdl_err:
                raise RuntimeError(f"Handler 生成失败: {hdl_err}")
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

        # 兜底：产物为空绝不能标记完成（空结果会让前端结果区一片空白且无从排查）
        if not files:
            raise RuntimeError("生成未产出任何文件（config.json 与 handler.c 均缺失），请重试")

        # 评测（纯本地计算，始终真实执行；TED 计算量大，按内容指纹缓存）
        evaluation = None
        _task_update(task_id, progress=85, stage=3, message="评测生成结果（本地计算）…")
        if gt_config and gt_config.exists() and gen_config.exists() and gt_handler.exists() and gen_handler.exists():
            evaluation = _cached_protocol_eval(gen_config, gt_config, gen_handler, gt_handler)
        elif config_obj:
            # 无 ground truth：优先以「上传文档表格提取」为参照做覆盖度评测，提取不到再回退结构检查
            evaluation = _doc_protocol_eval(doc_path, config_obj, gen_handler) or {
                "kind": "structural",
                "scores": {},
                "checks": {
                    "JSON 结构合法": True,
                    "顶层字段齐备": all(
                        k in config_obj for k in ("cmd_list", "tele_handler")
                    ),
                },
                "stats": {
                    "命令条目数": len(config_obj.get("cmd_list", [])),
                    "遥测条目数": len(config_obj.get("tele_handler", [])),
                },
                "note": "上传的新协议无 ground truth，仅做结构完整性检查",
            }

        _task_update(task_id, status="completed", progress=100, stage=3, message="完成", result={
            "kb": "protocol", "target": target_name, "files": files,
            "evaluation": evaluation, "pre_generated": pre_generated,
        })
    except Exception as e:
        _live_done(task_id, "live_config")
        _live_done(task_id, "live_handler_plan")
        _live_done(task_id, "live_handler")
        _task_update(task_id, status="failed", error=str(e), message=f"失败: {e}")


# ---------------------------------------------------------------------------
# 入口二流水线：模型部署适配（算子映射知识库检索 + LLM 生成）
# ---------------------------------------------------------------------------

def _build_onnx_prompt(source_text: str, analysis: dict, retrieved: list, module_name: str) -> list:
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
2. 模型类定义从源文件导入：源文件已在本目录提供为 `{module_name}.py`，请用 `from {module_name} import …` 导入（模块名必须完全一致），构造合理的示例输入
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


def _locate_torch_source(req: GenerateRequest, preset_dir: Path, preset_suffix: str):
    """返回 (源文件/目录 Path, 显示名, 预置产物 或 None)；operator/paddle 两目标共用同一批 PyTorch 样例"""
    if req.mode == "upload":
        # 模型类两目标共用上传产物：按 upload_id 跨目录定位（上传时记录的是当时所选目标的目录）
        for up_kb in ("operator", "paddle"):
            up = UPLOAD_DIR / up_kb / req.source
            if up.exists():
                pys = list(up.glob("*.py"))
                if pys:
                    return pys[0], pys[0].stem, None
        raise RuntimeError("上传文件不存在")
    parts = req.source.split("/")
    if len(parts) != 3:
        raise RuntimeError("样例 id 格式错误")
    _, domain, name = parts
    src_dir = OPALIGN_MODELS / "pytorch" / domain / "sourceModels" / name
    src = src_dir if src_dir.is_dir() else src_dir.with_suffix(".py")
    if not src.exists():
        raise RuntimeError("样例源码不存在")
    preset = preset_dir / f"{name}{preset_suffix}"
    return src, name, (preset if preset.exists() else None)


def _onnx_runtime_available() -> bool:
    """导出执行环境探测（find_spec 不真正 import，torch 不进入服务进程）。
    onnxscript 也要算：torch 2.9+ 的 torch.onnx 任意导出路径都会 import 它，缺失时
    导出必然 ModuleNotFoundError——探测为不可用即可优雅降级为仅交付导出脚本。"""
    import importlib.util
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "onnx", "onnxscript"))


# 推理冒烟测试脚本（子进程运行，坏模型崩溃不影响服务进程）
_ORT_SMOKE_SCRIPT = r"""
import json, sys
import numpy as np
import onnxruntime as ort

s = ort.InferenceSession(sys.argv[1], providers=["CPUExecutionProvider"])
inp, out = s.get_inputs()[0], s.get_outputs()[0]

def shape_for(batch):
    # 动态维（符号/None）按 batch 处理；静态维保持
    return [batch if i == 0 and not isinstance(d, int) else (d if isinstance(d, int) and d > 0 else 1)
            for i, d in enumerate(inp.shape)]

def run(batch):
    x = np.random.rand(*shape_for(batch)).astype(np.float32)
    return list(s.run(None, {inp.name: x})[0].shape)

r1 = run(1)
res = {"ok": True, "input": inp.name, "output": out.name, "output_shape": r1,
       "dynamic_batch_dim": not isinstance(inp.shape[0], int)}
if res["dynamic_batch_dim"]:
    r3 = run(3)
    res["dynamic_batch_ok"] = len(r3) == len(r1) and r3[0] == 3 and r3[1:] == r1[1:]
print("__ORT_RESULT__" + json.dumps(res))
"""


def _ort_smoke_test(model_path: Path) -> dict:
    """onnxruntime 真实加载 .onnx 并跑 dummy 推理：验证模型可执行；batch 维为符号时实测动态 batch"""
    try:
        proc = subprocess.run([sys.executable, "-c", _ORT_SMOKE_SCRIPT, str(model_path)],
                              capture_output=True, text=True, encoding="utf-8", errors="replace",
                              env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
                              timeout=180, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "推理冒烟测试超时（180s）"}
    for line in (proc.stdout or "").splitlines():
        if line.startswith("__ORT_RESULT__"):
            return json.loads(line[len("__ORT_RESULT__"):])
    return {"ok": False, "error": f"冒烟测试失败（exit {proc.returncode}）: {(proc.stderr or '')[-300:]}"}


# ONNX Runtime C++ 推理源码模板（无人机机载侧调用 .onnx 产物）
# IO 名称/形状在运行时从模型文件动态读取（换模型不用改代码）；
# 若导出阶段已拿到模型元数据，则以注释形式嵌入供人工核对
_CPP_INFERENCE_TEMPLATE = r'''// @NAME@_inference.cpp —— 无人机机载侧 ONNX 推理调用源码（ONNX Runtime C++ API）
// 由 domain-codegen 平台在导出流程中自动实例化，与 @MODEL_FILE@ 配套
//
// 依赖：ONNX Runtime >= 1.16（https://onnxruntime.ai/，机载端取 CPU 版即可）
// 编译：g++ -std=c++17 -O2 @NAME@_inference.cpp -o @NAME@_inference \
//         -I/path/to/onnxruntime/include -L/path/to/onnxruntime/lib -lonnxruntime
// 运行：将 @MODEL_FILE@ 与可执行文件放在同一目录，执行 ./@NAME@_inference

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

// 生成时读取的模型元数据（供核对；运行时以下方动态查询为准）：
@META@

int main() {
    // ---- 会话初始化：机载 CPU 推理；如有算力模组可追加 CUDA/TensorRT Provider ----
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "uav_infer");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);  // 机载 CPU 线程数按平台算力调整
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session session(env, "@MODEL_FILE@", session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    // ---- 动态读取输入/输出信息（与模型文件自描述一致，更换模型无需改代码）----
    std::vector<Ort::AllocatedStringPtr> in_ptrs, out_ptrs;
    std::vector<const char*> input_names, output_names;
    for (size_t i = 0; i < session.GetInputCount(); i++) {
        in_ptrs.push_back(session.GetInputNameAllocated(i, allocator));
        input_names.push_back(in_ptrs.back().get());
    }
    for (size_t i = 0; i < session.GetOutputCount(); i++) {
        out_ptrs.push_back(session.GetOutputNameAllocated(i, allocator));
        output_names.push_back(out_ptrs.back().get());
    }

    auto input_info = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    std::vector<int64_t> input_shape = input_info.GetShape();
    std::printf("输入 %s [", input_names[0]);
    for (size_t i = 0; i < input_shape.size(); i++) {
        if (input_shape[i] < 0) input_shape[i] = 1;  // 动态 batch 维按 1 处理
        std::printf("%s%lld", i ? ", " : "", (long long)input_shape[i]);
    }
    std::printf("]\n输出 %s\n", output_names[0]);

    // ---- 构造输入张量 ----
    // 实际场景替换为机载相机帧的预处理结果：解码 → resize 到模型输入尺寸 → 归一化 → NCHW 排布
    size_t input_count = 1;
    for (int64_t d : input_shape) input_count *= (size_t)d;
    std::vector<float> input_data(input_count, 0.5f);  // 演示用占位数据

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_count, input_shape.data(), input_shape.size());

    // ---- 推理 ----
    auto outputs = session.Run(Ort::RunOptions{nullptr},
                               input_names.data(), &input_tensor, input_names.size(),
                               output_names.data(), output_names.size());

    // ---- 输出解析：分类模型取 Top-5（分割/检测模型在此替换为对应后处理）----
    float* logits = outputs[0].GetTensorMutableData<float>();
    size_t out_count = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<size_t> order(out_count);
    std::iota(order.begin(), order.end(), 0);
    size_t top_k = out_count < 5 ? out_count : 5;
    std::partial_sort(order.begin(), order.begin() + top_k, order.end(),
                      [&](size_t a, size_t b) { return logits[a] > logits[b]; });
    std::printf("Top-%zu:\n", top_k);
    for (size_t i = 0; i < top_k; i++)
        std::printf("  类别 %4zu  置信度 %.4f\n", order[i], logits[order[i]]);
    return 0;
}
'''


def _gen_cpp_inference(name: str, onnx_model: Optional[dict]) -> str:
    """实例化 ONNX Runtime C++ 推理源码模板；有导出元数据时嵌入真实 IO 信息供核对"""
    if onnx_model:
        lines = []
        for io in onnx_model["inputs"]:
            lines.append(f"//   输入 {io['name']}: [{', '.join(map(str, io['shape']))}]")
        for io in onnx_model["outputs"]:
            lines.append(f"//   输出 {io['name']}: [{', '.join(map(str, io['shape']))}]")
        opset = next((o["version"] for o in onnx_model["opset"] if o["domain"] == "ai.onnx"), "—")
        lines.append(f"//   opset {opset} · 算子节点 {onnx_model['nodes']} · 模型大小 {onnx_model['size_mb']} MB")
        meta = "\n".join(lines)
    else:
        meta = "//（导出未执行，无元数据；IO 信息将在运行时从 .onnx 动态读取）"
    return (_CPP_INFERENCE_TEMPLATE
            .replace("@NAME@", name)
            .replace("@MODEL_FILE@", f"{name}.onnx")
            .replace("@META@", meta))


def _module_name_for(src: Path) -> str:
    """源文件在导出执行目录中的可导入模块名（目录型源码用包名；文件名清洗为合法 Python 标识符——
    上传文件名可能含短横线等非法字符，如 模型源码示例-dronenet.py → 模型源码示例_dronenet）"""
    if src.is_dir():
        return src.name
    mod = re.sub(r"\W", "_", src.stem)
    if not mod:
        return "model_source"
    return f"m_{mod}" if mod[0].isdigit() else mod


def _execute_onnx_export(out_dir: Path, script_path: Path, src: Path, name: str, module_name: str) -> dict:
    """子进程真实执行导出脚本，产出单文件 .onnx（外部权重自动重打包内嵌）并收集模型信息。

    产物交付物是 .onnx 模型文件；脚本本身仍作为代码产物保留展示。
    """
    # 源码以可导入的模块名拷入工作目录，保证脚本的 `from <module_name> import ...` 可用
    # （目录型源码整包拷贝；单文件按清洗后的模块名落地）
    if src.is_dir():
        shutil.copytree(src, out_dir / src.name, dirs_exist_ok=True)
    else:
        shutil.copy2(src, out_dir / f"{module_name}.py")

    try:
        # 子进程强制 UTF-8：Windows 中文环境默认 GBK，torch 导出器进度打印含 ✅ 等字符会
        # UnicodeEncodeError 崩掉导出；父进程按 UTF-8 读回日志
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        proc = subprocess.run(
            [sys.executable, script_path.name],
            cwd=out_dir, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            env=env, timeout=300,
        )
    except subprocess.TimeoutExpired:
        return {"attempted": True, "ok": False, "error": "导出脚本执行超时（300s 上限）"}

    log_tail = ((proc.stdout or "") + (proc.stderr or ""))[-800:]
    produced = sorted(out_dir.rglob("*.onnx"), key=lambda p: p.stat().st_mtime)  # 脚本可能写入子目录，递归扫描
    if proc.returncode != 0 or not produced:
        return {"attempted": True, "ok": False, "log": log_tail,
                "error": f"导出脚本执行失败（exit {proc.returncode}），日志尾部见评测备注"}

    raw_onnx = produced[-1]
    final_path = out_dir / f"{name}.onnx"
    try:
        import onnx
        m = onnx.load(str(raw_onnx))  # 外部权重（.onnx.data）按文件位置自动解析载入
        # torch 2.x 新导出器默认权重外部存储 → 重打包为单文件 .onnx，便于下载分发
        onnx.save_model(m, str(final_path), save_as_external_data=False)
        try:
            onnx.checker.check_model(m)
            checker_ok = True
        except Exception:
            checker_ok = False
        init_names = {i.name for i in m.graph.initializer}

        def _dims(t):
            return [d.dim_param or d.dim_value for d in t.type.tensor_type.shape.dim]

        inputs = [{"name": i.name, "shape": _dims(i)} for i in m.graph.input if i.name not in init_names]
        outputs = [{"name": o.name, "shape": _dims(o)} for o in m.graph.output]
        opset = [{"domain": o.domain or "ai.onnx", "version": o.version} for o in m.opset_import]
        nodes = len(m.graph.node)
        # 结构统计：参数量（initializer 维度积求和）与算子类型分布
        import math as _math
        from collections import Counter as _Counter
        params = sum(int(_math.prod(i.dims)) for i in m.graph.initializer if i.dims)
        op_counts = _Counter(n.op_type for n in m.graph.node).most_common(8)
    except Exception as e:
        return {"attempted": True, "ok": False, "log": log_tail,
                "error": f"ONNX 产物校验/重打包失败: {e}"}

    # 清理中间产物（外部权重 sidecar 与不同名的原始 .onnx）
    if raw_onnx != final_path:
        raw_onnx.unlink(missing_ok=True)
    for sidecar in out_dir.glob("*.onnx.data"):
        sidecar.unlink(missing_ok=True)

    size = final_path.stat().st_size
    # 推理冒烟测试（装了 onnxruntime 才做）：真实加载并跑 dummy 推理
    smoke = None
    import importlib.util as _ilu
    if _ilu.find_spec("onnxruntime") is not None:
        smoke = _ort_smoke_test(final_path)
    return {
        "attempted": True, "ok": True, "log": log_tail, "checker": checker_ok, "smoke": smoke,
        "onnx_model": {
            "name": final_path.name, "size_bytes": size,
            "size_mb": round(size / 1024 / 1024, 1),
            "nodes": nodes, "opset": opset, "inputs": inputs, "outputs": outputs,
            "checker": checker_ok,
            "params": params, "params_m": round(params / 1e6, 2),
            "op_counts": [[op, c] for op, c in op_counts],
        },
    }


def _eval_onnx(gen_code: str, analysis: dict, retrieved: list, exec_info: dict) -> dict:
    """评测：脚本静态检查 + 真实执行结果（导出成功 / ONNX 校验）；无 torch 环境时回退为纯静态检查"""
    try:
        ast.parse(gen_code)
        syntax_valid = True
    except SyntaxError:
        syntax_valid = False
    has_export = ("torch.onnx.export" in gen_code) or ("dynamo_export" in gen_code) or ("onnxscript" in gen_code)
    has_checker = "check_model" in gen_code or "onnx.checker" in gen_code
    custom_symbolics = len(re.findall(r"custom_symbolic|onnx_symbolic|def .*symbolic", gen_code))
    dynamic_axes = "dynamic_axes" in gen_code

    # 通过/失败型检查项（不打百分比）；连续程度分留在 scores
    checks = {
        "语法合法（导出脚本）": syntax_valid,
        "含导出调用（torch.onnx.export）": has_export,
    }
    if exec_info.get("attempted"):
        checks["导出执行成功"] = bool(exec_info.get("ok"))
        if exec_info.get("ok"):
            checks["ONNX 校验通过"] = bool(exec_info.get("checker"))
    else:
        checks["导出后自检（脚本含 check_model）"] = has_checker

    # 推理冒烟测试（onnxruntime 实跑）：模型不止结构合法，还要可执行
    smoke = exec_info.get("smoke")
    if smoke is not None:
        checks["推理冒烟测试（onnxruntime 实跑）"] = bool(smoke.get("ok"))
        if smoke.get("ok") and smoke.get("dynamic_batch_dim"):
            checks["动态 batch 实测"] = bool(smoke.get("dynamic_batch_ok"))

    evaluation = {
        "kind": "operator",
        "scores": {},
        "checks": checks,
        "syntax_valid": syntax_valid,
        "has_export_call": has_export,
        "has_checker": has_checker,
        "dynamic_axes": dynamic_axes,
        "custom_symbolics": custom_symbolics,
        "export_attempted": bool(exec_info.get("attempted")),
        "export_executed": bool(exec_info.get("ok")),
        "onnx_checker": bool(exec_info.get("checker")),
        "smoke_test": smoke,
        "total_calls": analysis["total_calls"],
        "retrieved_count": len(retrieved),
        "retrieved": [
            {"title": r["title"], "category_label": r["category_label"], "score": r["score"], "source": r["source"]}
            for r in retrieved
        ],
    }
    om = exec_info.get("onnx_model")
    if om:
        # 模型结构统计（从导出的 .onnx 实测）
        evaluation["stats"] = {
            "参数量": f"{om['params_m']:,.2f} M",
            "算子节点": om["nodes"],
            "算子种类": len(om["op_counts"]),
            "模型大小": f"{om['size_mb']} MB",
        }
        if smoke and smoke.get("ok"):
            evaluation["stats"]["冒烟输出形状"] = str(smoke["output_shape"])
    if exec_info.get("error"):
        evaluation["note"] = f"{exec_info['error']}\n{exec_info.get('log', '')[-400:]}"
    elif exec_info.get("skip_reason"):
        evaluation["note"] = exec_info["skip_reason"]
    return evaluation


def _run_operator(task_id: str, req: GenerateRequest):
    try:
        _task_update(task_id, status="running", progress=5, stage=0, message="解析模型源码…")
        src, name, preset_file = _locate_torch_source(req, ONNX_PRESET_DIR, "_to_onnx.py")
        module_name = _module_name_for(src)  # 源文件在导出执行目录中的可导入模块名

        if src.is_dir():
            texts = [p.read_text(encoding="utf-8", errors="replace") for p in sorted(src.rglob("*.py"))]
            source_text = "\n\n".join(texts)
        else:
            source_text = src.read_text(encoding="utf-8", errors="replace")

        _task_update(task_id, progress=20, stage=1, message="检索模型部署适配知识库…")
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
            _task_update(task_id, progress=45, stage=2, message="载入预生成导出脚本（真实流水线历史产物）…")
            _task_intermediate(task_id, gen_mode="预生成回放（真实流水线历史产物）")
            gen_code = preset_file.read_text(encoding="utf-8", errors="replace")
            time.sleep(0.5)
        elif _llm_available():
            # 上传的自有 case：检索 → LLM 真实生成
            _task_intermediate(task_id, gen_mode="LLM 在线生成")
            _task_update(task_id, progress=45, stage=2, message=f"LLM 生成 ONNX 导出代码（真实调用，约 1–2 分钟，注入 {len(retrieved)} 条知识）…")
            from herta.config import get_settings
            from herta.llm.client import BigModelClient, set_stream_sink
            sink = _live_sink(task_id, ["live"], ["ONNX 导出代码"])
            set_stream_sink(sink)
            messages = _build_onnx_prompt(source_text, analysis, retrieved, module_name)
            raw = BigModelClient().chat(
                messages=messages, model=get_settings().chat_model, temperature=0.1,
            )
            sink("flush", "")
            gen_code = _clean_code(raw)
            if not gen_code.strip():
                raise RuntimeError("模型返回内容为空或仅含格式围栏，请重试或更换模型")
            _live_done(task_id, "live")
        else:
            raise RuntimeError("上传的自有文件需要配置 LLM API Key（顶栏齿轮设置）；预置样例可直接演示")

        # 真实执行导出脚本，产出可下载的 .onnx 模型文件（torch/onnx 可用时；子进程隔离运行）
        out_dir = GEN_DIR / task_id
        out_dir.mkdir(parents=True, exist_ok=True)
        script_path = out_dir / f"{name}_to_onnx.py"
        script_path.write_text(gen_code, encoding="utf-8")
        onnx_model = None
        exec_info: dict = {"attempted": False}
        if _onnx_runtime_available():
            _task_update(task_id, progress=65, stage=2,
                         message="执行导出脚本，产出 ONNX 模型文件（真实运行 torch.onnx.export）…")
            exec_info = _execute_onnx_export(out_dir, script_path, src, name, module_name)
            onnx_model = exec_info.get("onnx_model")
        else:
            exec_info["skip_reason"] = "当前环境未安装 torch/onnx，仅交付导出脚本（部署环境安装后可现场产出 .onnx）"

        # 生成配套的机载侧 C++ 推理源码（ONNX Runtime 模板实例化；有导出元数据时嵌入真实 IO）
        _task_update(task_id, progress=78, stage=2, message="生成 C++ 推理调用源码（模板实例化 + 模型 IO 元数据）…")
        cpp_code = _gen_cpp_inference(name, onnx_model)
        (out_dir / f"{name}_inference.cpp").write_text(cpp_code, encoding="utf-8")

        _task_update(task_id, progress=85, stage=3, message="评测生成代码…")
        evaluation = _eval_onnx(gen_code, analysis, retrieved, exec_info)
        evaluation["checks"]["C++ 推理源码生成"] = True
        evaluation["cpp_inference"] = True
        if not pre_generated and preset_file:
            evaluation["line_similarity_vs_reference"] = _line_similarity(
                gen_code, preset_file.read_text(encoding="utf-8", errors="replace")
            )

        _task_update(task_id, status="completed", progress=100, stage=3, message="完成", result={
            "kb": "operator", "target": name,
            "files": [{"name": f"{name}_to_onnx.py", "language": "python", "content": gen_code},
                      {"name": f"{name}_inference.cpp", "language": "cpp", "content": cpp_code}],
            "onnx_model": onnx_model,
            "evaluation": evaluation, "pre_generated": pre_generated,
        })
    except Exception as e:
        _live_done(task_id, "live")
        _task_update(task_id, status="failed", error=str(e), message=f"失败: {e}")


# ---------------------------------------------------------------------------
# 入口二 · Paddle 目标：PyTorch → PaddlePaddle 跨框架转换
# （转换使用 ModelX 自带映射规则；ONNX 知识检索只作参考展示）
# ---------------------------------------------------------------------------

def _code_without_comments(src: str) -> str:
    """剥离注释（评测 torch 残留用，避免「torch.cat → paddle.concat」这类对齐注释误判）"""
    try:
        toks = (t for t in tokenize.generate_tokens(io.StringIO(src).readline)
                if t.type != tokenize.COMMENT)
        return tokenize.untokenize(toks)
    except Exception:
        return src


def _eval_paddle(
    gen_code: str,
    analysis: dict,
    retrieved: list,
    source_text: str = "",
    conversion: Optional[ConversionResult] = None,
    runtime: Optional[dict] = None,
    companion_files: Optional[dict] = None,
) -> dict:
    """评测 Paddle 产物。

    评测口径与 ONNX 路线保持一致：静态检查项单独列出，连续程度指标
    放在 scores 中；如果 Paddle 安装可用，再执行一次导入和 dummy 前向冒烟。
    """
    try:
        ast.parse(gen_code)
        syntax_valid = True
    except SyntaxError:
        syntax_valid = False

    code_only = _code_without_comments(gen_code)
    uses_paddle = bool(re.search(r"(?m)^\s*(import|from)\s+paddle\b", code_only))
    torch_residue = bool(
        re.search(r"(?m)^\s*(import|from)\s+torch\b", code_only)
        or re.search(r"(?<![\w.])torch\.", code_only)
    )
    has_layer = bool(re.search(r"(?:paddle\.)?nn\.Layer\b", code_only))
    has_forward = "def forward" in code_only
    wrappers = len(re.findall(r"class\s+Modified_\w+", code_only))

    scores = {}
    companion_files = conversion.extra_files if conversion is not None else companion_files or {}
    # 通过/失败型检查项
    checks = {
        "语法合法": syntax_valid,
        "引入 paddle": uses_paddle,
        "无 torch 残留": not torch_residue,
        "网络结构完整（nn.Layer + forward）": has_layer and has_forward,
    }

    runtime = runtime or {"attempted": False}
    if runtime.get("attempted"):
        checks["Paddle 模块导入成功"] = bool(runtime.get("import_ok"))
        if runtime.get("construction_attempted"):
            checks["Paddle 模型构造成功"] = bool(runtime.get("construction_ok"))
        if runtime.get("forward_attempted"):
            checks["Paddle dummy 前向成功"] = bool(runtime.get("forward_ok"))
        if runtime.get("error"):
            checks["Paddle 运行时检查无异常"] = False

    if conversion is not None:
        # ModelX 的 API 迁移统计是转换工具实际返回的计数，不把它伪装成
        # 代码相似度；与 ONNX 的导出/校验检查一样，在界面上单独展示。
        checks["ModelX 转换器产出代码"] = bool(gen_code.strip())
        checks["ModelX API 全部转换"] = conversion.failed_api_count == 0
        scores["ModelX API 转换成功率"] = conversion.api_success_rate
    if companion_files:
        companion_syntax_ok = True
        for filename, content in companion_files.items():
            try:
                ast.parse(content, filename=filename)
            except SyntaxError:
                companion_syntax_ok = False
        checks["伴随 Python 模块语法合法"] = companion_syntax_ok
        wrappers += sum(len(re.findall(r"class\s+Modified_\w+", text))
                        for text in companion_files.values())

    # 结构对齐（静态指标，无需 paddle 运行时）：
    # 主类名一致 / 层构造数量对齐 / forward 中 self.* 调用序列相似度
    if source_text:
        src_classes = re.findall(r"class\s+(\w+)\s*\(\s*(?:(?:torch\.)?nn)\.Module\s*\)", source_text)
        gen_classes = set(re.findall(r"class\s+(\w+)\s*\(\s*(?:(?:paddle\.)?nn\.)?Layer\s*\)", code_only))
        if src_classes:
            preserved = sum(1 for c in src_classes if c in gen_classes)
            scores["类结构对齐"] = round(preserved / len(src_classes) * 100, 1)

        src_layers = re.findall(r"(?:(?:torch\.)?nn)\.([A-Z]\w*)\s*\(", source_text)
        gen_layers = re.findall(r"(?:(?:paddle\.)?nn)\.([A-Z]\w*)\s*\(", code_only)
        if src_layers and gen_layers:
            scores["层数量对齐"] = round(min(len(src_layers), len(gen_layers))
                                       / max(len(src_layers), len(gen_layers)) * 100, 1)

        def _forward_calls(code: str) -> list:
            try:
                tree = ast.parse(code)
            except SyntaxError:
                return []
            calls = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef,)) and item.name == "forward":
                            for sub in ast.walk(item):
                                if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                                        and isinstance(sub.func.value, ast.Name) and sub.func.value.id == "self"):
                                    calls.append(sub.func.attr)
            return calls

        src_seq, gen_seq = _forward_calls(source_text), _forward_calls(code_only)
        if src_seq and gen_seq:
            import difflib
            scores["forward 调用对齐"] = round(
                difflib.SequenceMatcher(None, src_seq, gen_seq).ratio() * 100, 1)

    evaluation = {
        "kind": "paddle",
        "scores": scores,
        "checks": checks,
        "syntax_valid": syntax_valid,
        "uses_paddle": uses_paddle,
        "torch_residue": torch_residue,
        "custom_wrappers": wrappers,
        "total_calls": analysis["total_calls"],
        "retrieved_count": len(retrieved),
        "retrieved": [
            {"title": r["title"], "category_label": r["category_label"], "score": r["score"], "source": r["source"]}
            for r in retrieved
        ],
    }
    if conversion is not None:
        evaluation["stats"] = {
            "ModelX 转换器": conversion.converter,
            "源文件": Path(conversion.source_file).name,
            "成功转换 API": conversion.success_api_count,
            "未支持 API": conversion.failed_api_count,
        }
        if conversion.unsupported:
            evaluation["unsupported_apis"] = conversion.unsupported
        evaluation["compatibility_fixes"] = conversion.compatibility_fixes
    evaluation["library_modifications"] = [
        {"file": filename, "level": "Python underlying library",
         "classes": re.findall(r"class\s+(Modified_\w+)", content)}
        for filename, content in companion_files.items()
        if Path(filename).name.startswith("modified_")
    ]
    evaluation.setdefault("stats", {})["伴随模块"] = len(companion_files)
    if runtime.get("class_name"):
        evaluation.setdefault("stats", {})["主模型类"] = runtime["class_name"]
    if runtime.get("params") is not None:
        evaluation.setdefault("stats", {})["参数量"] = f"{runtime['params'] / 1e6:.2f} M"
    evaluation["paddle_runtime"] = runtime
    if runtime.get("attempted"):
        if runtime.get("output_shape_ok") is not None:
            checks["输出形状符合样例约定"] = bool(runtime["output_shape_ok"])
        if runtime.get("forward_attempted") and runtime.get("finite") is not None:
            checks["输出不含 NaN/Inf"] = bool(runtime["finite"])
        if runtime.get("error"):
            evaluation["note"] = runtime["error"]
        elif runtime.get("skip_reason"):
            evaluation["note"] = runtime["skip_reason"]
    else:
        evaluation["note"] = runtime.get("skip_reason") or "未执行 Paddle 运行时检查；已完成静态迁移评测。"
    return evaluation


def _run_paddle(task_id: str, req: GenerateRequest):
    try:
        _task_update(task_id, status="running", progress=5, stage=0, message="解析模型源码…")
        src, name, preset_file = _locate_torch_source(req, PADDLE_PRESET_DIR, "_to_paddle.py")
        profile = paddle_demo_cases().get(name) if req.mode == "sample" else None

        if src.is_dir():
            texts = [p.read_text(encoding="utf-8", errors="replace") for p in sorted(src.rglob("*.py"))]
            source_text = "\n\n".join(texts)
        else:
            source_text = src.read_text(encoding="utf-8", errors="replace")

        _task_update(task_id, progress=20, stage=1, message="检索模型部署适配知识库…")
        analysis = _analyze_source(source_text)
        leaf_ops = sorted({c.split(".")[-1] for c in analysis["calls"]})
        # 与 ONNX 目标共用同一数据集，仅用于工作台展示相关算子知识；
        # ModelX CMC 本身使用内置 AST/规则映射，不读取 LLM 或该检索结果。
        query = f"{name} PyTorch Paddle 转换 算子映射 " + " ".join(leaf_ops[:12])
        retrieved = _onnx_retrieve(query, top_k=8)
        _task_intermediate(
            task_id,
            operators={"total_calls": analysis["total_calls"], "leaf_ops": leaf_ops},
            retrieved=[{"title": r["title"], "category_label": r["category_label"],
                        "score": r["score"], "source": r["source"]} for r in retrieved],
        )

        out_dir = GEN_DIR / task_id
        out_dir.mkdir(parents=True, exist_ok=True)
        conversion: Optional[ConversionResult] = None
        extra_files = {}
        modelx_fallback_reason: Optional[str] = None
        pre_generated = False
        modelx_dir = out_dir / "modelx"
        _task_update(task_id, progress=45, stage=2, message="运行 ModelX CMC 源码级转换…")
        # ModelX CMC 是确定性的源码级 AST/规则转换器，不依赖 LLM。
        try:
            conversion = convert_pytorch_source(src, modelx_dir)
            gen_code = conversion.code
            extra_files = conversion.extra_files
            _task_intermediate(
                task_id,
                gen_mode="ModelX CMC 源码级转换（确定性）",
                conversion_stats={
                    "success": conversion.success_api_count,
                    "failed": conversion.failed_api_count,
                    "rate": conversion.api_success_rate,
                },
            )
        except (ModuleNotFoundError, ImportError) as modelx_error:
            if req.mode == "sample" and preset_file:
                # 预置样例保留离线回放兜底；在安装 ModelX 依赖的环境中，
                # 同一个样例会优先展示真实 CMC 转换产物。
                pre_generated = True
                modelx_fallback_reason = str(modelx_error)
                _task_update(task_id, progress=45, stage=2, message="ModelX 不可用，载入预生成 Paddle 转换产物…")
                _task_intermediate(task_id, gen_mode="预生成回放（ModelX 依赖缺失时的离线兜底）")
                gen_code = preset_file.read_text(encoding="utf-8", errors="replace")
                companion_dir = PADDLE_PRESET_DIR / name
                extra_files = {
                    p.relative_to(companion_dir).as_posix(): p.read_text(encoding="utf-8")
                    for p in sorted(companion_dir.rglob("*.py"))
                } if companion_dir.exists() else {}
                time.sleep(0.5)
            else:
                raise RuntimeError(
                    "ModelX CMC 转换失败：%s。请安装 vendor/modelx/requirements.txt。" % modelx_error
                ) from modelx_error

        evaluation_steps = [
            {
                "name": "ModelX CMC 源码转换",
                "status": ("failed" if conversion.failed_api_count else "passed") if conversion is not None else "skipped",
                "detail": (
                    f"{conversion.success_api_count} 个 API 成功，{conversion.failed_api_count} 个未支持"
                    if conversion is not None else "使用预生成离线产物"
                ),
            },
            {"name": "Paddle 导入与 dummy 前向", "status": "pending"},
            {"name": "语法/迁移/结构静态评测", "status": "pending"},
        ]
        _task_intermediate(task_id, evaluation_steps=evaluation_steps)
        script_path = out_dir / f"{name}_to_paddle.py"
        # Execute the same multi-file artifact shown in the UI, preserving the
        # relative imports emitted by ModelX. The smoke runner loads a package.
        script_path.write_text(gen_code, encoding="utf-8")
        for relative_name, content in extra_files.items():
            companion = out_dir / relative_name
            companion.parent.mkdir(parents=True, exist_ok=True)
            companion.write_text(content, encoding="utf-8")
        _task_update(task_id, progress=75, stage=2, message="执行 Paddle 产物导入与 dummy 前向检查…")
        evaluation_steps[1] = {"name": "Paddle 导入与 dummy 前向", "status": "running"}
        _task_intermediate(task_id, evaluation_steps=evaluation_steps)
        runtime = run_paddle_smoke_test(script_path, profile=profile)

        if runtime.get("attempted") and runtime.get("error"):
            runtime_status = "failed"
            runtime_detail = runtime["error"]
        elif runtime.get("attempted") and runtime.get("import_ok"):
            if runtime.get("forward_attempted"):
                runtime_status = "passed" if runtime.get("forward_ok") else "failed"
                runtime_detail = "前向输出 " + str(runtime.get("output_shape")) if runtime.get("forward_ok") else runtime.get("error", "前向失败")
            else:
                runtime_status = "skipped"
                runtime_detail = runtime.get("skip_reason", "未找到可默认构造的 Layer")
        elif runtime.get("attempted"):
            runtime_status = "failed"
            runtime_detail = runtime.get("error", "Paddle 导入失败")
        else:
            runtime_status = "skipped"
            runtime_detail = runtime.get("skip_reason", "未执行 Paddle 运行时检查")
        evaluation_steps[1] = {
            "name": "Paddle 导入与 dummy 前向",
            "status": runtime_status,
            "detail": runtime_detail,
        }
        evaluation_steps[2] = {"name": "语法/迁移/结构静态评测", "status": "running"}
        _task_intermediate(task_id, evaluation_steps=evaluation_steps)
        _task_update(task_id, progress=85, stage=3, message="评测生成代码…")
        evaluation = _eval_paddle(gen_code, analysis, retrieved, source_text, conversion, runtime,
                                  companion_files=extra_files)
        static_names = {"语法合法", "引入 paddle", "无 torch 残留", "网络结构完整（nn.Layer + forward）"}
        if extra_files:
            static_names.add("伴随 Python 模块语法合法")
        if conversion is not None:
            static_names.add("ModelX 转换器产出代码")
        static_checks = {k: v for k, v in evaluation.get("checks", {}).items() if k in static_names}
        static_ok = bool(static_checks) and all(static_checks.values())
        evaluation_steps[2] = {
            "name": "语法/迁移/结构静态评测",
            "status": "passed" if static_ok else "failed",
            "detail": f"{sum(static_checks.values())}/{len(static_checks)} 项通过",
        }
        _task_intermediate(task_id, evaluation_steps=evaluation_steps)
        if modelx_fallback_reason:
            evaluation["note"] = (
                "ModelX CMC 依赖不可用，当前展示预生成 Paddle 产物；原因：%s。 %s"
                % (modelx_fallback_reason, evaluation.get("note", ""))
            ).strip()
        result_files = [{"name": f"{name}_paddle.py", "language": "python", "content": gen_code}]
        result_files.extend(
            {"name": relative_name, "language": "python", "content": content}
            for relative_name, content in extra_files.items()
        )
        _task_update(task_id, status="completed", progress=100, stage=3, message="完成", result={
            "kb": "paddle", "target": name,
            "files": result_files,
            "evaluation": evaluation, "pre_generated": pre_generated,
        })
    except Exception as e:
        _live_done(task_id, "live")
        _task_update(task_id, status="failed", error=str(e), message=f"失败: {e}")


# ---------------------------------------------------------------------------
# API：知识库浏览（两套知识库内容查看）
# ---------------------------------------------------------------------------

BASELINE_DATASET = HERTA_KB_DIR / "baseline_dataset.json"


@app.get("/api/kb/protocol/dataset")
def protocol_dataset():
    """星务协议知识库的基本数据集可视化载荷（356 条原始基线 + 增强 3090 对照统计）。

    由 server/tools/build_baseline_dataset.py 从 协议代码数据集2 打包生成。
    """
    if not BASELINE_DATASET.exists():
        raise HTTPException(status_code=404, detail="基本数据集载荷不存在（先运行 server/tools/build_baseline_dataset.py）")
    return json.loads(BASELINE_DATASET.read_text(encoding="utf-8"))


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


@app.get("/api/kb/onnx/dataset")
def onnx_dataset():
    """模型部署适配知识库的数据集可视化载荷（1,349 条交付记录 + 基线/增量/筛选对照）。

    由 server/tools/build_onnx_dataset.py 从 pytorch-onnx-dataset2 打包生成。
    """
    view = ONNX_DATASET.parent / "dataset_view.json"
    if not view.exists():
        raise HTTPException(status_code=404, detail="数据集载荷不存在（先运行 server/tools/build_onnx_dataset.py）")
    return json.loads(view.read_text(encoding="utf-8"))


@app.get("/api/kb/paddle/dataset")
def paddle_dataset():
    """ModelX PyTorch→Paddle 映射规则数据集可视化载荷。"""
    if not PADDLE_DATASET.exists():
        raise HTTPException(status_code=404, detail="Paddle 映射规则载荷不存在（先运行 server/tools/build_paddle_dataset.py）")
    return json.loads(PADDLE_DATASET.read_text(encoding="utf-8"))


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
