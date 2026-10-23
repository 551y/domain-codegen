"""把 协议代码数据集2 的基本数据集（original 356 条）+ 构建对照统计打包为前端可视化载荷。

输出：vendor/herta/data/kb/baseline_dataset.json（知识库页「星务协议知识库」tab 的数据集可视化数据源）
口径：主数字为原始基线 356 条；增强后 3090 条仅作对照。行数统计按原始字节行数（同 wc -l）。

用法：python server/tools/build_baseline_dataset.py
"""
import collections
import json
from pathlib import Path

import sys
PLATFORM = Path(__file__).resolve().parents[2]
# 数据集源目录由命令行参数传入（本仓库已内置打包产物，换机器无需重建）
if len(sys.argv) < 2:
    sys.exit("用法：python server/tools/build_baseline_dataset.py <协议代码数据集2/dataset 目录>")
DS = Path(sys.argv[1])
OUT = PLATFORM / "vendor" / "herta" / "data" / "kb" / "baseline_dataset.json"


def raw_lines(path: Path) -> int:
    data = path.read_bytes()
    return data.count(b"\n") + (0 if data.endswith(b"\n") else 1)


def load_jsonl(path: Path) -> list:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


original = load_jsonl(DS / "original" / "herta_kb_original.jsonl")
original.sort(key=lambda r: int(r["id"]))

# chunk_id 消歧（与知识库构建口径一致：C 代码块 id 跨文件重复，追加文件主干名）
seen = {}
for r in original:
    cid = r["chunk_id"]
    if cid in seen:
        cid = f"{cid}__{Path(r['file_name']).stem}"
        assert cid not in seen
        r["chunk_id"] = cid
    seen[cid] = True

KIND_LABELS = {"doc": "协议文档", "code_c": "C 代码", "code_json": "配置 JSON"}

records = []
for r in original:
    is_doc = r["kind"] == "doc"
    title = r["file_name"] + (f" · 片段 {int(r.get('chunk_index', 0)) + 1}" if is_doc else "")
    records.append({
        "id": r["chunk_id"],
        "title": title,
        "category": r["kind"],
        "category_label": KIND_LABELS[r["kind"]],
        "source": r["source_protocol"],
        "file": r["file_name"],
        "text": r["text"],
        "char_count": int(r.get("char_count") or len(r["text"])),
        "token_count": int(r.get("token_count") or 0),
        "chunk_index": int(r.get("chunk_index") or 0),
    })

aug_meta = json.load(open(DS / "augmented" / "dataset_meta.json", encoding="utf-8"))
# dataset_meta 的 kind_distribution 键是中文标签，转回 kind key
label_to_kind = {v: k for k, v in KIND_LABELS.items()}
aug_kinds = {label_to_kind[k]: v for k, v in aug_meta["kind_distribution"].items()}

payload = {
    "baseline": {
        "total": len(original),
        "protocols": dict(collections.Counter(r["source_protocol"] for r in original)),
        "kinds": dict(collections.Counter(r["kind"] for r in original)),
        "mean_chars": round(sum(r["char_count"] for r in records) / len(records)),
        "chunk": "粗分块 500 / 50",
        "vectorized": "未向量化",
    },
    "augmented": {
        "total": aug_meta["total_records"],
        "protocols": aug_meta["source_protocols"],
        "kinds": aug_kinds,
        "mean_chars": round(aug_meta["text_length"]["mean"]),
        "chunk": "细分块 150 / 30",
        "vectorized": "embedding-3 · 1024 维",
        "ratio": round(aug_meta["total_records"] / len(original), 1),
    },
    "files": [
        {"path": "original/herta_kb_original.jsonl",
         "lines": raw_lines(DS / "original" / "herta_kb_original.jsonl"),
         "note": "每行一条 JSON 记录，即基本数据集 356 条"},
        {"path": "original/herta_kb_original.csv",
         "lines": raw_lines(DS / "original" / "herta_kb_original.csv"),
         "note": "同一份数据的表格形式；行数多是因为文本字段内含换行符"},
        {"path": "augmented/herta_kb_dataset.jsonl",
         "lines": raw_lines(DS / "augmented" / "herta_kb_dataset.jsonl"),
         "note": "每行一条 JSON 记录，即增强后 3,090 条"},
        {"path": "augmented/herta_kb_dataset.csv",
         "lines": raw_lines(DS / "augmented" / "herta_kb_dataset.csv"),
         "note": "表格形式；同样因文本字段内换行符多于记录数"},
    ],
    "records": records,
}

OUT.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
print(json.dumps({"records": len(records), "out": str(OUT),
                  "size_kb": round(OUT.stat().st_size / 1024)}, ensure_ascii=False))
