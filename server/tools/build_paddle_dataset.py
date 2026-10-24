"""Build the independent PyTorch→Paddle rule dataset view.

This payload is intentionally separate from ``build_onnx_dataset.py``.  ONNX
records come from the ONNX knowledge dataset; Paddle records come from the
mapping table and describe the source API, target API, matcher and
parameter/semantic rewrite used by the source converter.

Usage::

    python server/tools/build_paddle_dataset.py
    python server/tools/build_paddle_dataset.py <converter directory>
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path


PLATFORM = Path(__file__).resolve().parents[2]
CMC_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else PLATFORM / "vendor" / "modelx" / "CMC"
MAPPING = CMC_DIR / "pytorch_to_paddlepaddle.json"
OUT = PLATFORM / "vendor" / "modelx" / "paddle_dataset_view.json"


def raw_lines(path: Path) -> int:
    data = path.read_bytes()
    return data.count(b"\n") + (0 if data.endswith(b"\n") else 1)


def category_for(entry: dict) -> tuple[str, str]:
    matcher = str(entry.get("Matcher") or "")
    classification = entry.get("Classification")
    if matcher == "Unsupported" or classification == -1:
        return "unsupported", "不支持算子"
    if entry.get("params_change"):
        return "parameter_rewrite", "参数语义改写"
    if entry.get("target_api") and matcher in {"DirectMatcher", "InitializerMatcher"}:
        return "direct_mapping", "直接 API 映射"
    if matcher:
        return "semantic_rewrite", "语义转换/复合实现"
    return "unclassified", "待确认规则"


def main() -> None:
    if not MAPPING.exists():
        raise SystemExit(f"找不到映射表：{MAPPING}")

    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))
    records = []
    for source_api, entry in mapping.items():
        if source_api == "_comment" or not isinstance(entry, dict):
            continue
        category, category_label = category_for(entry)
        target_api = entry.get("target_api")
        if isinstance(target_api, (list, dict)):
            target_api = json.dumps(target_api, ensure_ascii=False)
        params_change = entry.get("params_change") or {}
        text = json.dumps({
            "source_api": source_api,
            "target_api": target_api or "",
            "matcher": entry.get("Matcher", ""),
            "classification": entry.get("Classification"),
            "params_change": params_change,
            "custom_mapping": entry.get("custom_mapping", ""),
        }, ensure_ascii=False, indent=2)
        records.append({
            "id": source_api,
            "title": source_api,
            "category": category,
            "category_label": category_label,
            "source": "pytorch_to_paddlepaddle.json",
            "text": text,
            "source_api": source_api,
            "target_api": target_api or "",
            "matcher": entry.get("Matcher", ""),
            "classification": entry.get("Classification"),
            "params_change": params_change,
            "framework_version": "AST/映射规则",
            "verification_status": "mapping rule",
        })

    records.sort(key=lambda r: (r["category"], r["title"].lower()))
    counts = collections.Counter(r["category"] for r in records)
    labels = {}
    for r in records:
        labels[r["category"]] = r["category_label"]
    categories = [
        {"key": key, "label": labels[key], "count": counts[key]}
        for key in ("direct_mapping", "parameter_rewrite", "semantic_rewrite", "unsupported", "unclassified")
        if counts[key]
    ]

    files = []
    for path in (MAPPING, CMC_DIR / "special_mapping.py", CMC_DIR / "nodeBase.py", CMC_DIR / "astAnalysis.py"):
        if path.exists():
            files.append({
                "path": str(path.relative_to(CMC_DIR)).replace("\\", "/"),
                "lines": raw_lines(path),
                "note": "转换规则/实现文件",
            })

    payload = {
        "total": len(records),
        "categories": categories,
        "composition": {
            "direct": counts["direct_mapping"],
            "parameter_rewrites": counts["parameter_rewrite"],
            "semantic_rewrites": counts["semantic_rewrite"],
            "unsupported": counts["unsupported"],
        },
        "files": files,
        "records": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"records": len(records), "categories": dict(counts), "out": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
