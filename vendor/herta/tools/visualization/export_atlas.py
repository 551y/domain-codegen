"""
Atlas 数据导出脚本

将知识库数据导出为 Nomic Atlas 支持的格式。
此脚本仅做数据导出，默认不执行联网上传。
需要人工手动执行，独立于 Herta 主流程。
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from common import (
    get_kb_dir,
    ensure_output_dir,
    load_kb_records,
    write_json_file,
    build_visualization_meta,
)


def build_atlas_payload(records: list[dict]) -> list[dict]:
    """
    构建 Atlas 所需的 payload 数据

    Args:
        records: 标准化记录列表

    Returns:
        Atlas payload 列表，每条记录包含:
            - id: 唯一标识
            - text: 完整文本
            - text_preview: 文本预览
            - file: 来源文件
            - section: 章节/段落
            - kind: 内容类型
            - chunk_strategy: 分块策略
            - chunk_index: 分块索引
    """
    payload = []
    for record in records:
        item = {
            "id": record["chunk_id"],
            "text": record["text"],
            "text_preview": record["text_preview"],
            "file": record["file"],
            "section": record["section"],
            "kind": record["kind"],
            "chunk_strategy": record["chunk_strategy"],
            "chunk_index": record["chunk_index"],
        }
        payload.append(item)
    return payload


def build_atlas_metadata_df(records: list[dict]) -> pd.DataFrame:
    """
    构建 Atlas 所需的 metadata DataFrame

    Args:
        records: 标准化记录列表

    Returns:
        metadata DataFrame，包含:
            - id
            - file
            - section
            - kind
            - chunk_strategy
            - chunk_index
            - text_preview
    """
    data = []
    for record in records:
        item = {
            "id": record["chunk_id"],
            "file": record["file"],
            "section": record["section"],
            "kind": record["kind"],
            "chunk_strategy": record["chunk_strategy"],
            "chunk_index": record["chunk_index"],
            "text_preview": record["text_preview"],
        }
        data.append(item)

    return pd.DataFrame(data)


def export_atlas_embeddings(
    records: list[dict],
    output_path: Path,
) -> Path:
    """
    导出 embeddings 为单独的 numpy 文件

    Args:
        records: 标准化记录列表
        output_path: 输出目录路径

    Returns:
        导出的文件路径
    """
    embeddings = np.array([r["embedding"] for r in records])
    embeddings_file = output_path / "atlas_embeddings.npy"
    np.save(embeddings_file, embeddings)
    return embeddings_file


def export_atlas_data(
    kb_name: str,
    kb_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    with_embeddings: bool = True,
) -> list[str]:
    """
    导出 Atlas 格式的数据文件

    Args:
        kb_name: 知识库名称
        kb_dir: 可选，手动指定 KB 目录
        out_dir: 可选，手动指定输出目录
        with_embeddings: 是否额外导出 embeddings.npy 文件

    Returns:
        生成的文件列表
    """
    # 确定知识库目录
    kb_path = get_kb_dir(kb_name, kb_dir)
    print(f"加载知识库: {kb_path}")

    # 加载数据
    records = load_kb_records(kb_path, kb_name)
    print(f"加载记录数: {len(records)}")

    if len(records) == 0:
        raise ValueError("没有可用的记录，无法导出")

    # 提取 embeddings 维度信息
    embeddings = np.array([r["embedding"] for r in records])
    embedding_dim = embeddings.shape[1]
    print(f"Embedding 维度: {embedding_dim}")

    # 准备输出目录
    output_path = ensure_output_dir(kb_name, out_dir)
    print(f"输出目录: {output_path}")

    generated_files = []

    # 1. 导出 payload (JSONL)
    print("导出 payload...")
    payload = build_atlas_payload(records)
    payload_file = output_path / "atlas_payload.jsonl"
    with open(payload_file, "w", encoding="utf-8") as f:
        for item in payload:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    generated_files.append("atlas_payload.jsonl")
    print(f"已生成: {payload_file}")

    # 2. 导出 metadata (CSV)
    print("导出 metadata...")
    metadata_df = build_atlas_metadata_df(records)
    metadata_file = output_path / "atlas_metadata.csv"
    metadata_df.to_csv(metadata_file, index=False, encoding="utf-8")
    generated_files.append("atlas_metadata.csv")
    print(f"已生成: {metadata_file}")

    # 3. 可选：导出 embeddings
    if with_embeddings:
        print("导出 embeddings...")
        embeddings_file = export_atlas_embeddings(records, output_path)
        generated_files.append("atlas_embeddings.npy")
        print(f"已生成: {embeddings_file}")

    # 4. 导出/更新可视化元信息
    meta_file = output_path / "visualization_meta.json"

    # 如果已有 meta 文件，尝试合并
    existing_meta = {}
    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                existing_meta = json.load(f)
        except json.JSONDecodeError:
            pass

    new_meta = build_visualization_meta(
        kb_name=kb_name,
        kb_dir=kb_path,
        record_count=len(records),
        embedding_dim=embedding_dim,
        projection_method="none",
        color_by="none",
        generated_files=list(set(existing_meta.get("generated_files", []) + generated_files)),
    )

    # 合并时保留最早的生成时间
    if "generated_at" in existing_meta:
        new_meta["generated_at"] = existing_meta["generated_at"]
        new_meta["updated_at"] = pd.Timestamp.now().isoformat()

    write_json_file(new_meta, meta_file)
    if "visualization_meta.json" not in generated_files:
        generated_files.append("visualization_meta.json")
    print(f"已生成: {meta_file}")

    return generated_files


def try_atlas_upload(
    payload_file: Path,
    embeddings_file: Path,
    kb_name: str,
) -> bool:
    """
    尝试上传到 Nomic Atlas（可选功能）

    Args:
        payload_file: payload JSONL 文件路径
        embeddings_file: embeddings numpy 文件路径
        kb_name: 知识库名称

    Returns:
        是否上传成功
    """
    try:
        from nomic import atlas
    except ImportError:
        print("\n提示: 要启用 Atlas 上传功能，请安装 nomic 包:")
        print("  pip install nomic")
        return False

    # 检查 API key
    import os
    if not os.environ.get("NOMIC_API_KEY"):
        print("\n提示: 要上传 Atlas，请先设置 API key:")
        print("  export NOMIC_API_KEY='your-api-key'")
        print("  或运行: nomic login")
        return False

    try:
        print(f"\n正在上传到 Nomic Atlas (项目: {kb_name})...")

        # 读取数据
        df = pd.read_json(payload_file, lines=True)
        embeddings = np.load(embeddings_file)

        # 上传
        dataset = atlas.map_data(
            data=df,
            embeddings=embeddings.tolist(),
            identifier=f"herta-{kb_name}",
            description=f"Herta 知识库 '{kb_name}' 的向量可视化",
        )

        print(f"上传成功！数据集 ID: {dataset.id}")
        return True

    except Exception as e:
        print(f"上传失败: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="导出知识库的 Atlas 格式数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 仅导出数据（默认）
  python export_atlas.py --kb gdwx

  # 导出并尝试上传 Atlas（需安装 nomic 并设置 API key）
  python export_atlas.py --kb gdwx --with-upload

  # 手动指定输入输出路径
  python export_atlas.py --kb gdwx --kb-dir /path/to/kb --out-dir /path/to/output

说明:
  此脚本默认只做数据导出，不强制联网上传。
  如需上传，请使用 --with-upload 参数，并确保已安装 nomic 包并设置 API key。
        """,
    )

    parser.add_argument(
        "--kb",
        type=str,
        required=True,
        help="知识库名称，例如 'gdwx'",
    )
    parser.add_argument(
        "--kb-dir",
        type=Path,
        default=None,
        help="知识库目录路径，默认根据 data/kb/<kb_name>/ 推导",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="输出目录路径，默认写入 output/visualization/<kb_name>/",
    )
    parser.add_argument(
        "--with-embeddings",
        action="store_true",
        default=True,
        dest="with_embeddings",
        help="导出 embeddings.npy 文件（默认开启）",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_false",
        dest="with_embeddings",
        help="不导出 embeddings.npy 文件",
    )
    parser.add_argument(
        "--with-upload",
        action="store_true",
        default=False,
        help="导出后尝试上传到 Nomic Atlas（需安装 nomic 并设置 API key）",
    )

    args = parser.parse_args()

    try:
        generated_files = export_atlas_data(
            kb_name=args.kb,
            kb_dir=args.kb_dir,
            out_dir=args.out_dir,
            with_embeddings=args.with_embeddings,
        )

        print("\nAtlas 数据导出完成！")
        print(f"生成文件: {', '.join(generated_files)}")

        # 打印文件说明
        out_path = ensure_output_dir(args.kb, args.out_dir)
        print(f"\n导出文件说明:")
        print(f"  - atlas_payload.jsonl : 文本数据（包含 id, text, metadata）")
        print(f"  - atlas_metadata.csv  : 元数据表格")
        if args.with_embeddings:
            print(f"  - atlas_embeddings.npy  : embedding 向量（来自原始数据）")
        print(f"  - visualization_meta.json: 导出元信息")

        # 打印 Atlas 上传提示
        print(f"\n如需上传到 Nomic Atlas:")
        print(f"  1. 安装依赖: pip install nomic")
        print(f"  2. 登录获取 API key: nomic login")
        print(f"  3. 运行: python export_atlas.py --kb {args.kb} --with-upload")

        # 可选：执行上传
        if args.with_upload:
            print()
            embeddings_file = out_path / "atlas_embeddings.npy"
            payload_file = out_path / "atlas_payload.jsonl"
            try_atlas_upload(payload_file, embeddings_file, args.kb)

    except FileNotFoundError as e:
        print(f"错误: 文件未找到 - {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"错误: 数据校验失败 - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
