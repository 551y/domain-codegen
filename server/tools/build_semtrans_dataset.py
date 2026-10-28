"""Build the standalone SemTrans semantic-definition catalog used by the UI."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor" / "semtrans"
SEMTRANS = VENDOR / "SemTrans"
OUTPUT = VENDOR / "paddle_dataset_view.json"
PINNED_REVISION = "431bd23c9773ec4a3c2d9d924e433e93d5a5577b"
ORIGINAL_ROOT = os.environ.get("SEMTRANS_ORIGINAL_ROOT")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_definitions(framework: str) -> tuple[dict, dict]:
    root = SEMTRANS / "math_symbol_base" / "symbolic_definition" / framework
    files = {name: root / name for name in ("nn_ops.yaml", "tensors.yaml")}
    entries: dict = {}
    for filename, path in files.items():
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for key, value in payload.items():
            if key in entries and entries[key] != value:
                raise ValueError(f"Conflicting {framework} definition for {key}")
            entries[key] = value
    return entries, files


def main() -> None:
    torch, torch_files = load_definitions("pytorch")
    paddle, paddle_files = load_definitions("paddle")
    paddle_by_name = {name.casefold(): (name, item) for name, item in paddle.items()}
    records = []
    matched = set()
    for source_name, source_def in sorted(torch.items(), key=lambda pair: pair[0].casefold()):
        target = paddle_by_name.get(source_name.casefold())
        target_name, target_def = target if target else ("", {})
        if target:
            matched.add(target_name)
        record = {
            "id": f"pytorch:{source_name}",
            "title": source_name,
            "source_api": source_name,
            "target_api": target_name,
            "category": "paired" if target else "source_only",
            "category_label": "PyTorch/Paddle 均有定义" if target else "仅有 PyTorch 定义",
            "source": "SemTrans-JIT / PyTorch 与 Paddle 数学定义",
            "framework_version": "SemTrans",
            "verification_status": "definition_present" if target else "target_definition_missing",
            "source_definition": source_def,
            "target_definition": target_def,
            "text": json.dumps({
                "pytorch": source_def,
                "paddle": target_def or None,
            }, ensure_ascii=False, indent=2),
        }
        records.append(record)
    for target_name, target_def in sorted(paddle.items(), key=lambda pair: pair[0].casefold()):
        if target_name in matched:
            continue
        records.append({
            "id": f"paddle:{target_name}",
            "title": target_name,
            "source_api": "",
            "target_api": target_name,
            "category": "target_only",
            "category_label": "仅有 Paddle 定义",
            "source": "SemTrans-JIT / Paddle 数学定义",
            "framework_version": "SemTrans",
            "verification_status": "source_definition_missing",
            "source_definition": {},
            "target_definition": target_def,
            "text": json.dumps({"pytorch": None, "paddle": target_def}, ensure_ascii=False, indent=2),
        })

    categories = []
    for key, label in (("paired", "PyTorch/Paddle 均有定义"),
                       ("source_only", "仅有 PyTorch 定义"),
                       ("target_only", "仅有 Paddle 定义")):
        count = sum(record["category"] == key for record in records)
        if count:
            categories.append({"key": key, "label": label, "count": count})
    file_rows = []
    for framework, source_files in (("pytorch", torch_files), ("paddle", paddle_files)):
        for name, path in source_files.items():
            file_rows.append({
                "path": f"SemTrans/math_symbol_base/symbolic_definition/{framework}/{name}",
                "lines": path.read_bytes().count(b"\n"),
                "note": f"SemTrans {framework} 原始数学语义定义 · SHA-256 {digest(path)}",
            })
    counts = {category["key"]: category["count"] for category in categories}
    payload = {
        "total": len(records),
        "source_revision": PINNED_REVISION,
        "categories": categories,
        "composition": {
            "paired": counts.get("paired", 0),
            "source_only": counts.get("source_only", 0),
            "target_only": counts.get("target_only", 0),
        },
        "files": file_rows,
        "records": records,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    provenance_path = VENDOR / "provenance.json"
    previous = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else {}
    prior_originals = previous.get("file_manifest", {})
    file_manifest = {}
    for path in sorted(SEMTRANS.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(SEMTRANS).as_posix()
        original_path = Path(ORIGINAL_ROOT) / Path(*Path(relative).parts) if ORIGINAL_ROOT else None
        original_hash = digest(original_path) if original_path and original_path.is_file() else prior_originals.get(relative, {}).get("original_sha256")
        vendored_hash = digest(path)
        file_manifest[relative] = {
            "original_sha256": original_hash,
            "vendored_sha256": vendored_hash,
            "matches_original": (original_hash == vendored_hash) if original_hash else None,
        }
    provenance = {
        "source_repository": "SemTrans local source tree provided for this migration",
        "source_revision": PINNED_REVISION,
        "scope": "SemTrans package files copied into vendor/semtrans/SemTrans, excluding tests and experiment artifacts",
        "file_manifest": file_manifest,
        "vendored_core_files": {
            path.relative_to(VENDOR).as_posix(): digest(path)
            for path in sorted(SEMTRANS.rglob("*.py"))
        },
        "known_adapter_changes": [
            "enhanced_conversion_agent accepts task_guidance for the two explicit demo profiles",
            "SemTrans LLM response keyword mismatch fixed to model_type",
            "bounded demo input-search configuration and empty-input fail-closed status added",
            "capture actual SemanticTracer region and first-mismatch evidence for the UI",
            "record every output numerical error and validate each output shape, dtype, finiteness, and tolerance",
            "use explicit client transport capability and response shape instead of model-name streaming heuristics",
            "compact demo generation prompt suppresses chain-of-thought and long reference scaffolding for real model latency",
            "parameter injection transposes 2D source Linear weights when target Paddle shape is [in, out]",
            "Paddle Linear bias_attr alias maps the stored bias tensor to SemTrans symbol B",
        ],
        "third_party_notice": "vendor/semtrans/THIRD_PARTY_NOTICES.md",
        "license_copy": "vendor/semtrans/LICENSES/Apache-2.0.txt",
        "original_semtrans_license": "No public license selected in source README; review before public redistribution.",
    }
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "composition": payload["composition"], "source_revision": PINNED_REVISION}, ensure_ascii=False))


if __name__ == "__main__":
    main()
