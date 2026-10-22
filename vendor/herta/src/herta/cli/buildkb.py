"""
buildkb 子命令实现

构建知识库 - 解析协议文档和源代码，构建可检索的向量知识库
支持四种分块策略：simple, semantic, autosize, proposition
"""

import json
from pathlib import Path
from typing import Optional, List

import typer
from typing_extensions import Annotated

from herta.logging import get_logger
from herta.config import get_settings
from herta.utils.paths import ensure_dir, project_relative
from herta.kb import KnowledgeBaseBuilder, KBMeta, ChunkConfig
from herta.kb.builder import build_kb

app = typer.Typer(
    name="buildkb",
    help="构建知识库 - 解析协议文档和源代码，构建可检索的向量知识库",
    rich_markup_mode="rich",
)

logger = get_logger(__name__)

SUPPORTED_STRATEGIES = ["simple", "semantic", "autosize", "proposition"]


def _parse_int_list(value: str) -> List[int]:
    """解析逗号分隔的整数列表"""
    if not value:
        return []
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_str_list(value: str) -> List[str]:
    """解析逗号分隔的字符串列表"""
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


@app.callback(invoke_without_command=True)
def main(
    source: Annotated[
        str,
        typer.Option(
            "--source",
            "-s",
            help="来源标识（如 gdwx）",
            rich_help_panel="输入选项",
        ),
    ],
    doc: Annotated[
        Optional[Path],
        typer.Option(
            "--doc",
            "-d",
            help="协议文档路径（docx/pdf）",
            rich_help_panel="输入选项",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    config: Annotated[
        Optional[Path],
        typer.Option(
            "--config",
            "-c",
            help="配置文件路径（json）",
            rich_help_panel="输入选项",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
    cmd: Annotated[
        Optional[Path],
        typer.Option(
            "--cmd",
            help="命令处理代码路径（c）",
            rich_help_panel="输入选项",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
    handler: Annotated[
        Optional[Path],
        typer.Option(
            "--handler",
            help="主处理代码路径（c）",
            rich_help_panel="输入选项",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
    internal: Annotated[
        Optional[Path],
        typer.Option(
            "--internal",
            "-i",
            help="内部处理代码路径（c）",
            rich_help_panel="输入选项",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
    out: Annotated[
        Optional[Path],
        typer.Option(
            "--out",
            "-o",
            help="知识库输出目录",
            rich_help_panel="输出选项",
        ),
    ] = None,
    chunk_strategy: Annotated[
        str,
        typer.Option(
            "--chunk-strategy",
            help="分块策略",
            rich_help_panel="分块选项",
        ),
    ] = "simple",
    chunk_size: Annotated[
        int,
        typer.Option(
            "--chunk-size",
            help="分块大小（字符数）",
            rich_help_panel="分块选项",
        ),
    ] = 500,
    chunk_overlap: Annotated[
        int,
        typer.Option(
            "--chunk-overlap",
            help="分块重叠大小",
            rich_help_panel="分块选项",
        ),
    ] = 50,
    # Semantic chunking 参数
    semantic_threshold_percentile: Annotated[
        float,
        typer.Option(
            "--semantic-threshold-percentile",
            help="语义分块阈值百分位（0.0-1.0）",
            rich_help_panel="Semantic 选项",
        ),
    ] = 0.5,
    semantic_min_sentences: Annotated[
        int,
        typer.Option(
            "--semantic-min-sentences",
            help="语义分块最小句子数",
            rich_help_panel="Semantic 选项",
        ),
    ] = 2,
    semantic_max_chars: Annotated[
        int,
        typer.Option(
            "--semantic-max-chars",
            help="语义分块最大字符数",
            rich_help_panel="Semantic 选项",
        ),
    ] = 2000,
    # Auto-size 参数
    candidate_sizes: Annotated[
        str,
        typer.Option(
            "--candidate-sizes",
            help="候选分块大小（逗号分隔，如300,500,800）",
            rich_help_panel="Auto-size 选项",
        ),
    ] = "300,500,800,1000",
    candidate_overlaps: Annotated[
        str,
        typer.Option(
            "--candidate-overlaps",
            help="候选重叠大小（逗号分隔，如50,100）",
            rich_help_panel="Auto-size 选项",
        ),
    ] = "50,100",
    eval_questions: Annotated[
        Optional[str],
        typer.Option(
            "--eval-questions",
            help="评估问题（逗号分隔，暂未启用）",
            rich_help_panel="Auto-size 选项",
        ),
    ] = None,
    selection_output: Annotated[
        Optional[Path],
        typer.Option(
            "--selection-output",
            help="选择报告输出路径",
            rich_help_panel="Auto-size 选项",
        ),
    ] = None,
    # Proposition 参数
    proposition_model: Annotated[
        str,
        typer.Option(
            "--proposition-model",
            help="命题提取使用的LLM模型",
            rich_help_panel="Proposition 选项",
        ),
    ] = "glm-5.1",
    max_propositions_per_chunk: Annotated[
        int,
        typer.Option(
            "--max-propositions-per-chunk",
            help="每个chunk的最大命题数",
            rich_help_panel="Proposition 选项",
        ),
    ] = 10,
    proposition_batch_size: Annotated[
        int,
        typer.Option(
            "--proposition-batch-size",
            help="命题提取批次大小（段落数）",
            rich_help_panel="Proposition 选项",
        ),
    ] = 5,
    proposition_cache: Annotated[
        bool,
        typer.Option(
            "--proposition-cache/--no-proposition-cache",
            help="是否启用命题提取缓存",
            rich_help_panel="Proposition 选项",
        ),
    ] = True,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="强制重建，覆盖已有知识库",
            rich_help_panel="执行选项",
        ),
    ] = False,
) -> None:
    """
    构建知识库

    支持四种分块策略：
    - simple: 固定大小滑动窗口分块
    - semantic: 基于语义相似度的分块
    - autosize: 自动选择最优分块参数
    - proposition: 基于原子事实命题的分块

    示例：
        # 使用默认 simple 策略
        herta buildkb --source gdwx

        # 使用 semantic 策略
        herta buildkb --source gdwx --chunk-strategy semantic --semantic-threshold-percentile 0.6

        # 使用 autosize 策略
        herta buildkb --source gdwx --chunk-strategy autosize --candidate-sizes "400,600,800"

        # 使用 proposition 策略
        herta buildkb --source gdwx --chunk-strategy proposition --max-propositions-per-chunk 5
    """
    logger.info("=" * 60)
    logger.info("开始执行 buildkb 命令")
    logger.info("=" * 60)

    # 验证分块策略
    if chunk_strategy not in SUPPORTED_STRATEGIES:
        raise typer.BadParameter(
            f"未知的分块策略: '{chunk_strategy}'\n"
            f"支持策略: {', '.join(SUPPORTED_STRATEGIES)}"
        )

    # 确定输出目录
    output_dir = out or project_relative(f"data/kb/{source}")
    output_dir = Path(output_dir)

    logger.info(f"Source: {source}")
    logger.info(f"输出目录: {output_dir.absolute()}")
    logger.info(f"分块策略: {chunk_strategy}")

    # 检查是否已存在
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            logger.warning(f"输出目录已存在且非空: {output_dir}")
            logger.warning("使用 --force 强制重建，或使用 --out 指定其他目录")
            raise typer.Exit(1)
        else:
            logger.info("强制重建模式：将覆盖已有知识库")

    # 解析输入文件
    input_files = _resolve_input_files(source, doc, config, cmd, handler, internal)
    logger.info(f"输入文件 ({len(input_files)} 个):")
    for f in input_files:
        logger.info(f"  - {f}")

    if not input_files:
        logger.error("没有有效的输入文件")
        raise typer.Exit(1)

    # 分类文件
    doc_files = [f for f in input_files if f.suffix.lower() in ['.docx', '.pdf']]
    code_files = [f for f in input_files if f.suffix.lower() in ['.json', '.c', '.h']]

    # 创建 ChunkConfig
    chunk_config = ChunkConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=chunk_strategy,
        # Semantic
        semantic_threshold_percentile=semantic_threshold_percentile,
        semantic_min_sentences=semantic_min_sentences,
        semantic_max_chars=semantic_max_chars,
        # Auto-size
        candidate_sizes=_parse_int_list(candidate_sizes),
        candidate_overlaps=_parse_int_list(candidate_overlaps),
        eval_questions=_parse_str_list(eval_questions) if eval_questions else None,
        selection_output=selection_output,
        # Proposition
        proposition_model=proposition_model,
        max_propositions_per_chunk=max_propositions_per_chunk,
        proposition_batch_size=proposition_batch_size,
        proposition_cache=proposition_cache,
    )

    logger.info(f"分块参数:")
    logger.info(f"  策略: {chunk_strategy}")
    logger.info(f"  大小: {chunk_size}, 重叠: {chunk_overlap}")

    if chunk_strategy == "semantic":
        logger.info(f"  Semantic: 阈值百分位={semantic_threshold_percentile}, "
                   f"最小句子={semantic_min_sentences}, 最大字符={semantic_max_chars}")
    elif chunk_strategy == "autosize":
        logger.info(f"  Auto-size: 候选大小={chunk_config.candidate_sizes}, "
                   f"候选重叠={chunk_config.candidate_overlaps}")
    elif chunk_strategy == "proposition":
        logger.info(f"  Proposition: 模型={proposition_model}, "
                   f"最大命题数={max_propositions_per_chunk}, 批次={proposition_batch_size}")

    # 获取配置
    settings = get_settings()
    logger.info(f"Embedding 模型: {settings.embedding_model}")
    logger.info(f"Embedding 维度: {settings.embedding_dimensions}")

    # 创建输出目录
    ensure_dir(output_dir)

    # 执行构建
    start_time = __import__('datetime').datetime.now()

    try:
        meta = build_kb(
            source=source,
            input_files=input_files,
            output_dir=output_dir,
            chunk_config=chunk_config,
        )
    except Exception as e:
        logger.error(f"知识库构建失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        raise typer.Exit(1)

    end_time = __import__('datetime').datetime.now()
    duration = (end_time - start_time).total_seconds()

    # 生成构建报告
    build_report = {
        "build_time": start_time.isoformat(),
        "duration_seconds": duration,
        "source": source,
        "input_files": [str(f) for f in input_files],
        "doc_files": len(doc_files),
        "code_files": len(code_files),
        "chunk_strategy": chunk_strategy,
        "chunk_params": chunk_config.to_dict(),
        "output": {
            "kb_dir": str(output_dir),
            "chunk_count": meta.chunk_count if hasattr(meta, 'chunk_count') else 0,
            "embedding_model": settings.embedding_model,
            "embedding_dimensions": settings.embedding_dimensions,
        },
        "files_generated": [
            "chunks.jsonl",
            "meta.json",
            "embeddings.npy",
            "vector.index",
            "build_report.json",
        ]
    }

    # 添加策略特有报告
    if chunk_strategy == "autosize" and selection_output:
        # 尝试读取选择报告
        try:
            with open(selection_output, 'r') as f:
                selection_report = json.load(f)
            build_report["selection_report"] = selection_report
        except Exception:
            pass

    # 保存构建报告
    report_path = output_dir / "build_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(build_report, f, indent=2, ensure_ascii=False)
    logger.info(f"构建报告已保存: {report_path}")

    # 输出结果摘要
    logger.info("")
    logger.info("=" * 60)
    logger.info("✅ 知识库构建完成！")
    logger.info("=" * 60)
    logger.info(f"  Source:        {source}")
    logger.info(f"  Chunk 数量:    {meta.chunk_count if hasattr(meta, 'chunk_count') else 'N/A'}")
    logger.info(f"  分块策略:      {chunk_strategy}")
    logger.info(f"  Embedding模型: {settings.embedding_model}")
    logger.info(f"  输出目录:      {output_dir.absolute()}")
    logger.info(f"  构建耗时:      {duration:.2f} 秒")
    logger.info("")
    logger.info("输出文件:")
    logger.info(f"  - chunks.jsonl    (分块数据)")
    logger.info(f"  - meta.json       (构建元数据)")
    logger.info(f"  - embeddings.npy  (向量数组)")
    logger.info(f"  - vector.index    (FAISS 索引)")
    logger.info(f"  - build_report.json (构建报告)")
    logger.info("=" * 60)


def _resolve_input_files(
    source: str,
    doc: Optional[Path],
    config: Optional[Path],
    cmd: Optional[Path],
    handler: Optional[Path],
    internal: Optional[Path],
) -> list[Path]:
    """解析输入文件路径"""
    files = []

    # 使用自动推断路径
    if doc is None and config is None and cmd is None and handler is None and internal is None:
        base_dir = project_relative(f"data/source_proto/{source}")

        inferred_files = {
            "doc": base_dir / f"{source}.docx",
            "config": base_dir / f"{source}_config.json",
            "cmd": base_dir / "cmd_handler.c",
            "handler": base_dir / f"{source}_handler.c",
            "internal": base_dir / "internal_handler.c",
        }

        logger.info(f"自动推断路径 (base: {base_dir}):")
        for name, path in inferred_files.items():
            if path.exists():
                logger.info(f"  ✓ {name}: {path}")
                files.append(path)
            else:
                logger.info(f"  ✗ {name}: {path} (不存在)")

        if not files:
            raise typer.BadParameter(
                f"无法为 source='{source}' 推断任何输入文件。\n"
                f"请确认目录存在: {base_dir}\n"
                f"或手动指定文件路径: --doc, --config, --cmd, --handler, --internal"
            )
    else:
        # 使用用户指定的路径
        if doc:
            files.append(doc)
        if config:
            files.append(config)
        if cmd:
            files.append(cmd)
        if handler:
            files.append(handler)
        if internal:
            files.append(internal)

    # 验证文件存在
    valid_files = [f for f in files if f.exists()]
    return valid_files


if __name__ == "__main__":
    app()
