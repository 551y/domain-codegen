"""把 pytorch-onnx-dataset2（模型部署适配知识库数据源）打包为前端可视化载荷。

输出：vendor/onnx/dataset_view.json（知识库页「模型部署适配知识库」tab 的数据集可视化数据源）
口径：当前交付 1,349 条（基线 1,268 + 实测验证增量 81）。
      记录取自 vendor 内的 expanded.jsonl（即在线知识库本体）。

用法：python server/tools/build_onnx_dataset.py
"""
import collections
import json
from pathlib import Path

import sys
PLATFORM = Path(__file__).resolve().parents[2]
# 数据集源目录由命令行参数传入（本仓库已内置打包产物，换机器无需重建）
if len(sys.argv) < 2:
    sys.exit("用法：python server/tools/build_onnx_dataset.py <pytorch-onnx-dataset2 目录>")
DS = Path(sys.argv[1])
VENDOR = PLATFORM / "vendor" / "onnx"
OUT = VENDOR / "dataset_view.json"

# 数据集独立页（pytorch-onnx-dataset2/index.html）的类别顺序与调色板，保持一致
CATEGORY_ORDER = ["operator_testcase", "operator_composite", "benchmark_case", "operator_mapping",
                  "version_matrix_row", "document", "fix_case", "error_pattern", "example"]


def raw_lines(path: Path) -> int:
    data = path.read_bytes()
    return data.count(b"\n") + (0 if data.endswith(b"\n") else 1)


manifest = json.load(open(VENDOR / "manifest.json", encoding="utf-8"))
labels = manifest["category_labels"]
additions_count = sum(1 for line in open(DS / "data" / "expansion" / "additions.jsonl", encoding="utf-8") if line.strip())

records = []
for line in open(VENDOR / "dataset.expanded.jsonl", encoding="utf-8"):
    if not line.strip():
        continue
    d = json.loads(line)
    content = d.get("content", "")
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    text = content if len(content) <= 1500 else content[:1500] + "\n…（截断，完整内容见数据集文件）"
    records.append({
        "id": d.get("id", ""),
        "title": d.get("title", ""),
        "category": d.get("category", ""),
        "category_label": labels.get(d.get("category", ""), d.get("category", "")),
        "source": d.get("source", ""),
        "text": text,
        "framework_version": d.get("framework_version", ""),
        "verification_status": d.get("verification_status", ""),
    })

counts = collections.Counter(r["category"] for r in records)
categories = [{"key": k, "label": labels[k], "count": counts[k]} for k in CATEGORY_ORDER if counts.get(k)]

payload = {
    "total": len(records),
    "categories": categories,
    "composition": {
        "baseline": manifest.get("baseline_count", 1268),
        "additions": additions_count,
    },
    "files": [
        {"path": "data/dataset.expanded.jsonl",
         "lines": raw_lines(DS / "data" / "dataset.expanded.jsonl"),
         "note": "当前交付全量（基线 + 验证增量），每行一条记录"},
        {"path": "data/dataset.standardized.jsonl",
         "lines": raw_lines(DS / "data" / "dataset.standardized.jsonl"),
         "note": "整理后基线记录"},
        {"path": "data/expansion/additions.jsonl",
         "lines": raw_lines(DS / "data" / "expansion" / "additions.jsonl"),
         "note": "实测验证增量（49 个完整示例 + 32 个错误复现/修复对）"},
        {"path": "data/dataset.markdown.jsonl",
         "lines": raw_lines(DS / "data" / "dataset.markdown.jsonl"),
         "note": "同一份数据的 Markdown 展示形式"},
    ],
    "records": records,
}

OUT.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
print(json.dumps({"records": len(records), "additions": additions_count,
                  "out": str(OUT), "size_kb": round(OUT.stat().st_size / 1024)}, ensure_ascii=False))
