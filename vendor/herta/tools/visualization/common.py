"""
可视化工具集的公共模块

提供知识库数据加载、校验和预处理功能。
此模块独立于 Herta 主流程，仅供人工可视化分析使用。
"""

import json
import sys
from pathlib import Path
from typing import Union, Optional
from datetime import datetime

import numpy as np


def get_project_root() -> Path:
    """
    获取项目根目录

    通过向上查找 pyproject.toml 或 .git 来确定项目根目录

    Returns:
        项目根目录的 Path 对象
    """
    # 从当前文件开始向上查找
    current = Path(__file__).resolve().parent

    while True:
        # 检查标记文件
        if (current / "pyproject.toml").exists() or (current / ".git").exists():
            return current

        # 到达根目录，停止搜索
        parent = current.parent
        if parent == current:
            # 找不到标记文件，使用当前工作目录
            return Path.cwd()
        current = parent


def get_kb_dir(kb_name: str, kb_dir: Optional[Path] = None) -> Path:
    """
    获取知识库目录路径

    Args:
        kb_name: 知识库名称
        kb_dir: 可选，手动指定的知识库目录路径

    Returns:
        知识库目录的 Path 对象

    Raises:
        FileNotFoundError: 当知识库目录不存在时
    """
    if kb_dir is not None:
        path = Path(kb_dir)
    else:
        project_root = get_project_root()
        path = project_root / "data" / "kb" / kb_name

    if not path.exists():
        raise FileNotFoundError(f"知识库目录不存在: {path}")

    if not path.is_dir():
        raise NotADirectoryError(f"路径不是目录: {path}")

    return path


def ensure_output_dir(kb_name: str, out_dir: Optional[Path] = None) -> Path:
    """
    确保输出目录存在，如果不存在则创建

    Args:
        kb_name: 知识库名称，用于构建默认输出路径
        out_dir: 可选，手动指定的输出目录路径

    Returns:
        输出目录的 Path 对象
    """
    if out_dir is not None:
        path = Path(out_dir)
    else:
        project_root = get_project_root()
        path = project_root / "output" / "visualization" / kb_name

    path.mkdir(parents=True, exist_ok=True)
    return path


def build_text_preview(text: str, max_len: int = 120) -> str:
    """
    构建文本预览，截断过长文本

    Args:
        text: 原始文本
        max_len: 最大预览长度，默认 120

    Returns:
        截断后的文本预览
    """
    if not text:
        return ""

    # 清理文本中的换行符和多余空格
    cleaned = " ".join(text.split())

    if len(cleaned) <= max_len:
        return cleaned

    # 截断并添加省略号
    return cleaned[:max_len].rstrip() + "..."


def load_chunks_jsonl(kb_dir: Path) -> list[dict]:
    """
    加载 chunks.jsonl 文件

    Args:
        kb_dir: 知识库目录路径

    Returns:
        chunk 记录列表

    Raises:
        FileNotFoundError: 当文件不存在时
        json.JSONDecodeError: 当 JSON 解析失败时
    """
    chunks_file = kb_dir / "chunks.jsonl"

    if not chunks_file.exists():
        raise FileNotFoundError(f"chunks.jsonl 不存在: {chunks_file}")

    chunks = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
                chunks.append(chunk)
            except json.JSONDecodeError as e:
                print(f"警告: 第 {line_num} 行 JSON 解析失败: {e}", file=sys.stderr)
                continue

    return chunks


def load_meta_json(kb_dir: Path) -> dict:
    """
    加载 meta.json 文件

    Args:
        kb_dir: 知识库目录路径

    Returns:
        meta 数据字典

    Raises:
        FileNotFoundError: 当文件不存在时
    """
    meta_file = kb_dir / "meta.json"

    if not meta_file.exists():
        raise FileNotFoundError(f"meta.json 不存在: {meta_file}")

    with open(meta_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_embeddings_npy(kb_dir: Path) -> np.ndarray:
    """
    加载 embeddings.npy 文件

    Args:
        kb_dir: 知识库目录路径

    Returns:
        embeddings 数组，shape 为 (N, D)

    Raises:
        FileNotFoundError: 当文件不存在时
    """
    embeddings_file = kb_dir / "embeddings.npy"

    if not embeddings_file.exists():
        raise FileNotFoundError(f"embeddings.npy 不存在: {embeddings_file}")

    return np.load(embeddings_file)


def validate_kb_data(
    chunks: list[dict],
    embeddings: np.ndarray,
    kb_name: str,
) -> tuple[list[dict], np.ndarray]:
    """
    校验 chunks 和 embeddings 的一致性

    Args:
        chunks: chunk 记录列表
        embeddings: embeddings 数组
        kb_name: 知识库名称，用于生成 chunk_id

    Returns:
        校验通过的 (chunks, embeddings) 元组

    Raises:
        ValueError: 当数据校验失败时
    """
    chunk_count = len(chunks)
    embedding_count = len(embeddings)

    if chunk_count == 0:
        raise ValueError("chunks 数量为空，无法继续")

    if embedding_count == 0:
        raise ValueError("embeddings 数量为空，无法继续")

    if chunk_count != embedding_count:
        # 如果数量不一致，以较小的为准，打印警告
        min_count = min(chunk_count, embedding_count)
        print(
            f"警告: chunks ({chunk_count}) 和 embeddings ({embedding_count}) 数量不一致，"
            f"将使用前 {min_count} 条数据",
            file=sys.stderr,
        )
        chunks = chunks[:min_count]
        embeddings = embeddings[:min_count]

    return chunks, embeddings


def normalize_chunk_record(
    chunk: dict,
    embedding: np.ndarray,
    index: int,
    kb_name: str,
) -> dict:
    """
    将 chunk 记录标准化为统一结构

    Args:
        chunk: 原始 chunk 字典
        embedding: 对应的 embedding 向量
        index: 数据索引位置
        kb_name: 知识库名称

    Returns:
        标准化的记录字典，包含以下字段:
            - chunk_id: 唯一标识
            - text: 原始文本
            - text_preview: 文本预览
            - file: 来源文件
            - section: 章节/段落标识
            - kind: 内容类型
            - chunk_strategy: 分块策略
            - chunk_index: 分块索引
            - metadata: 原始元数据
            - embedding: embedding 向量
    """
    # 提取或生成 chunk_id
    chunk_id = chunk.get("chunk_id")
    if not chunk_id:
        chunk_id = f"{kb_name}_{index:06d}"

    # 提取文本
    text = chunk.get("text", "")

    # 构建文本预览
    text_preview = build_text_preview(text)

    # 提取文件信息（兼容多种字段位置）
    file_path = chunk.get("file", "")
    if not file_path:
        # 尝试从 metadata 中提取
        metadata = chunk.get("metadata", {})
        if isinstance(metadata, dict):
            file_path = metadata.get("file", "")
    if not file_path:
        file_path = "unknown"

    # 提取章节信息
    section = chunk.get("section", "")
    if not section:
        metadata = chunk.get("metadata", {})
        if isinstance(metadata, dict):
            section = metadata.get("section", "")
    if not section:
        section = "unknown"

    # 提取类型信息
    kind = chunk.get("kind", "")
    if not kind:
        metadata = chunk.get("metadata", {})
        if isinstance(metadata, dict):
            kind = metadata.get("kind", "")
    if not kind:
        kind = "unknown"

    # 提取分块策略
    chunk_strategy = chunk.get("chunk_strategy", "")
    if not chunk_strategy:
        metadata = chunk.get("metadata", {})
        if isinstance(metadata, dict):
            chunk_strategy = metadata.get("chunk_strategy", "")
    if not chunk_strategy:
        chunk_strategy = "unknown"

    # 提取分块索引
    chunk_index = chunk.get("chunk_index", index)

    # 保留原始元数据
    metadata = chunk.get("metadata", {})

    return {
        "chunk_id": chunk_id,
        "text": text,
        "text_preview": text_preview,
        "file": file_path,
        "section": section,
        "kind": kind,
        "chunk_strategy": chunk_strategy,
        "chunk_index": chunk_index,
        "metadata": metadata,
        "embedding": embedding,
    }


def load_kb_records(
    kb_dir: Path,
    kb_name: str,
) -> list[dict]:
    """
    加载知识库的所有记录，包含校验和标准化

    Args:
        kb_dir: 知识库目录路径
        kb_name: 知识库名称

    Returns:
        标准化的记录列表，每个记录包含 chunk 信息和 embedding

    Raises:
        FileNotFoundError: 当必要文件不存在时
        ValueError: 当数据校验失败时
    """
    # 加载数据
    chunks = load_chunks_jsonl(kb_dir)
    embeddings = load_embeddings_npy(kb_dir)

    # 校验数据一致性
    chunks, embeddings = validate_kb_data(chunks, embeddings, kb_name)

    # 标准化记录
    records = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        record = normalize_chunk_record(chunk, embedding, i, kb_name)
        records.append(record)

    return records


def write_json_file(data: Union[dict, list], filepath: Path, indent: int = 2) -> None:
    """
    将数据写入 JSON 文件

    Args:
        data: 要写入的数据
        filepath: 目标文件路径
        indent: JSON 缩进，默认 2
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def write_jsonl_file(records: list[dict], filepath: Path) -> None:
    """
    将记录写入 JSONL 文件

    Args:
        records: 记录列表
        filepath: 目标文件路径
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for record in records:
            # 过滤掉 numpy array 等非 JSON 可序列化的数据
            serializable = {
                k: v for k, v in record.items() if not isinstance(v, np.ndarray)
            }
            f.write(json.dumps(serializable, ensure_ascii=False) + "\n")


def build_visualization_meta(
    kb_name: str,
    kb_dir: Path,
    record_count: int,
    embedding_dim: int,
    projection_method: str,
    color_by: str,
    generated_files: list[str],
) -> dict:
    """
    构建可视化元信息

    Args:
        kb_name: 知识库名称
        kb_dir: 知识库目录路径
        record_count: 记录数量
        embedding_dim: embedding 维度
        projection_method: 降维方法
        color_by: 着色字段
        generated_files: 生成的文件列表

    Returns:
        可视化元信息字典
    """
    return {
        "kb_name": kb_name,
        "kb_dir": str(kb_dir),
        "record_count": record_count,
        "embedding_dim": embedding_dim,
        "projection_method": projection_method,
        "color_by": color_by,
        "generated_files": generated_files,
        "generated_at": datetime.now().isoformat(),
    }
