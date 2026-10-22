"""
Plotly 可视化导出脚本

对知识库 embedding 进行 2D 降维投影，生成交互式散点图 HTML 文件。
此脚本独立于 Herta 主流程，需人工手动执行。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from common import (
    get_kb_dir,
    ensure_output_dir,
    load_kb_records,
    write_json_file,
    build_visualization_meta,
)


def project_embeddings(
    embeddings: np.ndarray,
    method: str = "umap",
    random_state: int = 42,
) -> np.ndarray:
    """
    对 embeddings 进行 2D 降维投影

    Args:
        embeddings: 原始 embeddings 数组，shape 为 (N, D)
        method: 降维方法，支持 "umap" 或 "pca"
        random_state: 随机种子，用于结果可复现

    Returns:
        2D 投影后的数组，shape 为 (N, 2)

    Raises:
        ValueError: 当指定的方法不支持时
        ImportError: 当使用 umap 但未安装 umap-learn 时
    """
    method = method.lower()

    if method == "umap":
        try:
            import umap

            reducer = umap.UMAP(
                n_components=2,
                random_state=random_state,
                min_dist=0.1,
                n_neighbors=15,
                metric="cosine",
            )
            return reducer.fit_transform(embeddings)
        except ImportError:
            print(
                "警告: umap-learn 未安装，将自动回退到 PCA",
                file=sys.stderr,
            )
            method = "pca"

    if method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
        return reducer.fit_transform(embeddings)

    raise ValueError(f"不支持的降维方法: {method}，仅支持 'umap' 或 'pca'")


def get_color_values(records: list[dict], color_by: str) -> list:
    """
    从记录中提取用于着色的值

    Args:
        records: 标准化记录列表
        color_by: 着色字段名称

    Returns:
        着色值列表
    """
    values = []
    for record in records:
        value = record.get(color_by, "unknown")
        if not value:
            value = "unknown"
        values.append(value)
    return values


def extract_filename(file_path: str) -> str:
    """
    从完整路径中提取文件名

    Args:
        file_path: 文件路径

    Returns:
        文件名
    """
    if not file_path or file_path == "unknown":
        return "unknown"
    return Path(file_path).name


def limit_categories_for_color(
    color_values: list,
    max_categories: int = 20,
) -> list:
    """
    限制着色类别数量，避免类别过多导致可视化混乱

    Args:
        color_values: 原始着色值列表
        max_categories: 最大类别数

    Returns:
        处理后的着色值列表
    """
    # 统计各值出现次数
    value_counts = {}
    for v in color_values:
        value_counts[v] = value_counts.get(v, 0) + 1

    # 如果类别数量在限制内，直接返回
    if len(value_counts) <= max_categories:
        return color_values

    # 按出现次数排序，保留前 N-1 个，其余归为 "其他"
    sorted_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
    top_values = set(v for v, _ in sorted_values[: max_categories - 1])

    result = []
    for v in color_values:
        if v in top_values:
            result.append(v)
        else:
            result.append("其他")

    return result


def create_plotly_scatter(
    records: list[dict],
    projection_2d: np.ndarray,
    color_values: list,
    title: str = "Knowledge Base Embedding Visualization",
) -> go.Figure:
    """
    创建 Plotly 散点图

    Args:
        records: 标准化记录列表
        projection_2d: 2D 投影坐标
        color_values: 着色值列表
        title: 图表标题

    Returns:
        Plotly Figure 对象
    """
    # 获取唯一的类别
    unique_categories = sorted(set(color_values))

    # 为每个类别创建 trace
    fig = go.Figure()

    for category in unique_categories:
        # 获取该类别的索引
        indices = [i for i, v in enumerate(color_values) if v == category]

        # 提取数据
        x_vals = projection_2d[indices, 0]
        y_vals = projection_2d[indices, 1]
        chunk_ids = [records[i]["chunk_id"] for i in indices]

        # 构建 hover 文本
        hover_texts = []
        for i in indices:
            record = records[i]
            hover_parts = [
                f"<b>ID:</b> {record['chunk_id']}",
                f"<b>File:</b> {extract_filename(record['file'])}",
                f"<b>Section:</b> {record['section']}",
                f"<b>Kind:</b> {record['kind']}",
                f"<b>Strategy:</b> {record['chunk_strategy']}",
                f"<b>Preview:</b> {record['text_preview']}",
            ]
            hover_texts.append("<br>".join(hover_parts))

        # 添加散点 trace
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='markers',
            name=str(category),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
            marker=dict(size=8, opacity=0.7),
        ))

    # 更新布局
    fig.update_layout(
        title=title,
        width=1200,
        height=800,
        template="plotly_white",
        xaxis_title="投影维度 1",
        yaxis_title="投影维度 2",
        legend=dict(
            title=dict(text="类别"),
            itemsizing="constant",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
        ),
        margin=dict(l=50, r=200, t=80, b=50),
    )

    return fig


def export_plotly_visualization(
    kb_name: str,
    kb_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    method: str = "umap",
    color_by: str = "file",
    title: Optional[str] = None,
) -> list[str]:
    """
    导出 Plotly 可视化

    Args:
        kb_name: 知识库名称
        kb_dir: 可选，手动指定 KB 目录
        out_dir: 可选，手动指定输出目录
        method: 降维方法，支持 "umap" 或 "pca"
        color_by: 着色字段，默认 "file"
        title: 图表标题，可选

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
        raise ValueError("没有可用的记录，无法生成可视化")

    # 提取 embeddings
    embeddings = np.array([r["embedding"] for r in records])
    embedding_dim = embeddings.shape[1]
    print(f"Embedding 维度: {embedding_dim}")

    # 降维投影
    print(f"使用 {method.upper()} 进行 2D 降维...")
    projection_2d = project_embeddings(embeddings, method=method)
    print(f"投影完成，坐标范围: X=[{projection_2d[:, 0].min():.2f}, {projection_2d[:, 0].max():.2f}], "
          f"Y=[{projection_2d[:, 1].min():.2f}, {projection_2d[:, 1].max():.2f}]")

    # 准备着色数据
    color_values = get_color_values(records, color_by)
    if color_by == "file":
        # 对于 file 字段，只显示文件名部分，避免路径过长
        color_values = [extract_filename(v) for v in color_values]
    color_values = limit_categories_for_color(color_values)

    # 确定图表标题
    if title is None:
        title = f"知识库 '{kb_name}' 向量可视化 ({method.upper()} 降维)"

    # 创建散点图
    fig = create_plotly_scatter(records, projection_2d, color_values, title=title)

    # 准备输出目录
    output_path = ensure_output_dir(kb_name, out_dir)
    print(f"输出目录: {output_path}")

    generated_files = []

    # 1. 导出 2D 投影数据
    projection_file = output_path / "projection_2d.json"
    projection_data = {
        "coordinates": projection_2d.tolist(),
        "record_ids": [r["chunk_id"] for r in records],
        "method": method,
    }
    write_json_file(projection_data, projection_file)
    generated_files.append("projection_2d.json")
    print(f"已生成: {projection_file}")

    # 2. 导出 Plotly HTML
    html_file = output_path / "plotly_scatter.html"
    fig.write_html(
        html_file,
        full_html=True,
        include_plotlyjs="cdn",
    )
    generated_files.append("plotly_scatter.html")
    print(f"已生成: {html_file}")

    # 3. 导出可视化元信息
    meta_file = output_path / "visualization_meta.json"
    meta = build_visualization_meta(
        kb_name=kb_name,
        kb_dir=kb_path,
        record_count=len(records),
        embedding_dim=embedding_dim,
        projection_method=method,
        color_by=color_by,
        generated_files=generated_files,
    )
    write_json_file(meta, meta_file)
    generated_files.append("visualization_meta.json")
    print(f"已生成: {meta_file}")

    return generated_files


def main():
    parser = argparse.ArgumentParser(
        description="导出知识库的 Plotly 可视化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置生成可视化
  python export_plotly.py --kb gdwx

  # 指定降维方法和着色字段
  python export_plotly.py --kb gdwx --method pca --color-by kind

  # 手动指定输入输出路径
  python export_plotly.py --kb gdwx --kb-dir /path/to/kb --out-dir /path/to/output
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
        "--method",
        type=str,
        default="umap",
        choices=["umap", "pca"],
        help="降维方法，支持 'umap' 或 'pca'，默认 'umap'（未安装时自动回退到 PCA）",
    )
    parser.add_argument(
        "--color-by",
        type=str,
        default="file",
        help="着色字段，默认 'file'（可选: file, kind, section, chunk_strategy）",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="图表标题，可选",
    )

    args = parser.parse_args()

    try:
        generated_files = export_plotly_visualization(
            kb_name=args.kb,
            kb_dir=args.kb_dir,
            out_dir=args.out_dir,
            method=args.method,
            color_by=args.color_by,
            title=args.title,
        )

        print("\n可视化导出完成！")
        print(f"生成文件: {', '.join(generated_files)}")
        print(f"\n可打开以下文件查看可视化结果:")
        out_path = ensure_output_dir(args.kb, args.out_dir)
        print(f"  file://{out_path / 'plotly_scatter.html'}")

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
