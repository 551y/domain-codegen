"""
generate 子命令实现

负责基于目标协议生成代码和配置
当前支持：
- config: 生成 config.json
- handler: 生成 handler.c（两阶段生成）
"""

import typer
from typing_extensions import Annotated
from pathlib import Path
from typing import Optional

from herta.logging import get_logger
from herta.config import get_settings
from herta.utils.paths import ensure_dir, project_relative
from herta.generator import ConfigGenerator, HandlerGenerator

app = typer.Typer(
    name="generate",
    help="生成目标协议代码 - 基于 RAG 检索生成 config.json 和 handler.c",
    rich_markup_mode="rich",
)

logger = get_logger(__name__)


# 创建子命令分组
config_app = typer.Typer(name="config", help="生成配置文件 (config.json)")
handler_app = typer.Typer(name="handler", help="生成处理代码 (handler.c) - 两阶段生成")

app.add_typer(config_app)
app.add_typer(handler_app)


# ==================== CONFIG 子命令 ====================

@config_app.callback(invoke_without_command=True)
def generate_config(
    target: Annotated[
        str,
        typer.Option(
            "--target",
            "-t",
            help="目标协议标识（如 dllg）",
            rich_help_panel="输入选项",
        ),
    ],
    target_doc: Annotated[
        Optional[Path],
        typer.Option(
            "--target-doc",
            "-d",
            help="目标协议文档路径（docx/pdf）",
            rich_help_panel="输入选项",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    source_kb: Annotated[
        str,
        typer.Option(
            "--source-kb",
            "-s",
            help="源协议知识库标识（如 gdwx）",
            rich_help_panel="输入选项",
        ),
    ] = "gdwx",
    kb_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--kb-dir",
            "-k",
            help="知识库目录路径（覆盖 --source-kb）",
            rich_help_panel="输入选项",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ] = None,
    out: Annotated[
        Optional[Path],
        typer.Option(
            "--out",
            "-o",
            help="输出目录",
            rich_help_panel="输出选项",
        ),
    ] = None,
    top_k: Annotated[
        int,
        typer.Option(
            "--top-k",
            help="检索时返回的最相似片段数量",
            rich_help_panel="检索选项",
        ),
    ] = 5,
    model: Annotated[
        Optional[str],
        typer.Option(
            "--model",
            "-m",
            help="使用的 LLM 模型",
            rich_help_panel="生成选项",
        ),
    ] = None,
    temperature: Annotated[
        float,
        typer.Option(
            "--temperature",
            help="生成温度（0.0-1.0）",
            rich_help_panel="生成选项",
        ),
    ] = 0.1,
) -> None:
    """
    生成目标协议的配置文件 (config.json)

    该命令将：
    1. 解析目标协议文档
    2. 从源协议知识库检索相关参考信息
    3. 调用 LLM 生成 config.json
    4. 保存到输出目录

    默认路径规则：
    - 目标协议文档：data/target_proto/<target>/<target>.docx（或 .pdf）
    - 知识库：data/kb/<source_kb>/
    - 输出目录：output/generated/<target>/

    示例：
        # 使用默认路径生成 dllg 的配置（基于 gdwx 知识库）
        herta generate config --target dllg

        # 指定目标文档和知识库
        herta generate config --target dllg --target-doc ./custom.docx --source-kb gdwx

        # 指定知识库目录
        herta generate config --target dllg --kb-dir ./my_kb/gdwx

        # 调整检索参数和模型
        herta generate config --target dllg --top-k 10 --model glm-5.1 --temperature 0.2
    """
    logger.info("=" * 60)
    logger.info("开始执行 generate config 命令")
    logger.info("=" * 60)

    # 确定目标文档路径
    if target_doc is None:
        # 尝试推断默认路径
        base_dir = project_relative(f"data/target_proto/{target}")

        # 按优先级尝试不同格式
        possible_paths = [
            base_dir / f"{target}.docx",
            base_dir / f"{target}.pdf",
            base_dir / f"{target}.doc",
        ]

        target_doc = None
        for path in possible_paths:
            if path.exists():
                target_doc = path
                logger.info(f"自动推断目标文档: {path}")
                break

        if target_doc is None:
            raise typer.BadParameter(
                f"无法为 target='{target}' 推断目标文档路径。\n"
                f"尝试的路径: {[str(p) for p in possible_paths]}\n"
                f"请使用 --target-doc 手动指定文档路径"
            )
    else:
        logger.info(f"使用指定的目标文档: {target_doc}")

    # 确定知识库目录
    if kb_dir is None:
        kb_dir = project_relative(f"data/kb/{source_kb}")

    kb_dir = Path(kb_dir)
    logger.info(f"知识库目录: {kb_dir.absolute()}")

    # 检查知识库是否存在
    if not kb_dir.exists():
        raise typer.BadParameter(
            f"知识库目录不存在: {kb_dir}\n"
            f"请先使用 'herta buildkb --source {source_kb}' 构建知识库"
        )

    required_files = ["vector.index", "chunks.jsonl", "meta.json"]
    missing_files = [f for f in required_files if not (kb_dir / f).exists()]
    if missing_files:
        raise typer.BadParameter(
            f"知识库不完整，缺少文件: {missing_files}\n"
            f"请重新运行 'herta buildkb --source {source_kb}'"
        )

    # 确定输出目录
    output_dir = out or project_relative(f"output/generated/{target}")
    output_dir = ensure_dir(output_dir)
    logger.info(f"输出目录: {output_dir.absolute()}")

    # 获取配置
    settings = get_settings()
    model = model or settings.chat_model

    logger.info(f"参数:")
    logger.info(f"  Target: {target}")
    logger.info(f"  Source KB: {source_kb}")
    logger.info(f"  Top-K: {top_k}")
    logger.info(f"  LLM 模型: {model}")
    logger.info(f"  Temperature: {temperature}")

    # 创建配置生成器
    try:
        generator = ConfigGenerator()
    except Exception as e:
        logger.error(f"创建生成器失败: {e}")
        raise typer.Exit(1)

    # 执行生成
    try:
        result = generator.generate(
            target_doc_path=target_doc,
            source_kb_dir=kb_dir,
            target_name=target,
            top_k=top_k,
            model=model,
            temperature=temperature,
            output_dir=output_dir,
        )
    except Exception as e:
        logger.error(f"配置生成失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        raise typer.Exit(1)

    # 检查结果
    if result.error_message:
        logger.error(f"生成过程中出现错误: {result.error_message}")
        raise typer.Exit(1)

    # 输出结果摘要
    logger.info("")
    logger.info("=" * 60)
    logger.info("✅ 配置生成完成！")
    logger.info("=" * 60)
    logger.info(f"  Target:       {result.target}")
    logger.info(f"  Source KB:    {result.source_kb}")
    logger.info(f"  Top-K:        {top_k}")
    logger.info(f"  检索片段数:   {result.retrieved_chunk_count}")
    logger.info(f"  LLM 模型:     {result.model}")
    logger.info(f"  配置文件:     {result.output_path}")
    logger.info("")

    # 输出配置预览
    if result.config:
        logger.info("配置预览:")
        preview = str(result.config)[:500]
        logger.info(f"  {preview}...")

    logger.info("=" * 60)


# ==================== HANDLER 子命令 ====================

@handler_app.callback(invoke_without_command=True)
def generate_handler(
    target: Annotated[
        str,
        typer.Option(
            "--target",
            "-t",
            help="目标协议标识（如 dllg）",
            rich_help_panel="输入选项",
        ),
    ],
    target_doc: Annotated[
        Optional[Path],
        typer.Option(
            "--target-doc",
            "-d",
            help="目标协议文档路径（docx/pdf）",
            rich_help_panel="输入选项",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    source_kb: Annotated[
        str,
        typer.Option(
            "--source-kb",
            "-s",
            help="源协议知识库标识（如 gdwx）",
            rich_help_panel="输入选项",
        ),
    ] = "gdwx",
    kb_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--kb-dir",
            "-k",
            help="知识库目录路径（覆盖 --source-kb）",
            rich_help_panel="输入选项",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ] = None,
    out: Annotated[
        Optional[Path],
        typer.Option(
            "--out",
            "-o",
            help="代码输出目录",
            rich_help_panel="输出选项",
        ),
    ] = None,
    top_k: Annotated[
        int,
        typer.Option(
            "--top-k",
            help="检索时返回的最相似片段数量（handler生成需要更多参考）",
            rich_help_panel="检索选项",
        ),
    ] = 8,
    model: Annotated[
        Optional[str],
        typer.Option(
            "--model",
            "-m",
            help="使用的 LLM 模型",
            rich_help_panel="生成选项",
        ),
    ] = None,
    temperature: Annotated[
        float,
        typer.Option(
            "--temperature",
            help="生成温度（0.0-1.0）",
            rich_help_panel="生成选项",
        ),
    ] = 0.1,
    save_plan: Annotated[
        bool,
        typer.Option(
            "--save-plan",
            help="保存设计计划（handler_plan.md）到中间目录",
            rich_help_panel="输出选项",
        ),
    ] = False,
) -> None:
    """
    生成目标协议的 Handler 代码 (handler.c)

    采用两阶段生成策略：
    1. 阶段A：生成 Handler 设计草图（plan）
    2. 阶段B：基于 plan 生成完整 C 代码

    该命令将：
    1. 解析目标协议文档
    2. 从源协议知识库检索 handler 相关的代码和文档
    3. 阶段A：调用 LLM 生成设计草图（函数列表、命令分发逻辑等）
    4. 阶段B：基于设计草图生成完整 handler.c 代码
    5. 保存到输出目录

    默认路径规则：
    - 目标协议文档：data/target_proto/<target>/<target>.docx（或 .pdf）
    - 知识库：data/kb/<source_kb>/
    - 代码输出目录：output/generated/<target>/
    - 设计计划中间目录：output/intermediate/<target>/（需 --save-plan）

    示例：
        # 使用默认路径生成 dllg 的 handler（基于 gdwx 知识库）
        herta generate handler --target dllg

        # 同时保存设计计划
        herta generate handler --target dllg --save-plan

        # 指定参数
        herta generate handler --target dllg --top-k 12 --temperature 0.15

        # 手动指定路径
        herta generate handler --target dllg --target-doc ./custom.docx --kb-dir ./my_kb
    """
    logger.info("=" * 60)
    logger.info("开始执行 generate handler 命令")
    logger.info("=" * 60)

    # 确定目标文档路径
    if target_doc is None:
        base_dir = project_relative(f"data/target_proto/{target}")
        possible_paths = [
            base_dir / f"{target}.docx",
            base_dir / f"{target}.pdf",
            base_dir / f"{target}.doc",
        ]

        target_doc = None
        for path in possible_paths:
            if path.exists():
                target_doc = path
                logger.info(f"自动推断目标文档: {path}")
                break

        if target_doc is None:
            raise typer.BadParameter(
                f"无法为 target='{target}' 推断目标文档路径。\n"
                f"尝试的路径: {[str(p) for p in possible_paths]}\n"
                f"请使用 --target-doc 手动指定文档路径"
            )
    else:
        logger.info(f"使用指定的目标文档: {target_doc}")

    # 确定知识库目录
    if kb_dir is None:
        kb_dir = project_relative(f"data/kb/{source_kb}")

    kb_dir = Path(kb_dir)
    logger.info(f"知识库目录: {kb_dir.absolute()}")

    # 检查知识库
    if not kb_dir.exists():
        raise typer.BadParameter(
            f"知识库目录不存在: {kb_dir}\n"
            f"请先使用 'herta buildkb --source {source_kb}' 构建知识库"
        )

    required_files = ["vector.index", "chunks.jsonl", "meta.json"]
    missing_files = [f for f in required_files if not (kb_dir / f).exists()]
    if missing_files:
        raise typer.BadParameter(
            f"知识库不完整，缺少文件: {missing_files}"
        )

    # 确定输出目录
    output_dir = out or project_relative(f"output/generated/{target}")
    output_dir = ensure_dir(output_dir)
    logger.info(f"代码输出目录: {output_dir.absolute()}")

    # 确定中间目录（用于保存设计计划）
    intermediate_dir = None
    if save_plan:
        intermediate_dir = project_relative(f"output/intermediate/{target}")
        logger.info(f"设计计划将保存到: {intermediate_dir}")

    # 获取配置
    settings = get_settings()
    model = model or settings.chat_model

    logger.info(f"参数:")
    logger.info(f"  Target:       {target}")
    logger.info(f"  Source KB:    {source_kb}")
    logger.info(f"  Top-K:        {top_k}")
    logger.info(f"  LLM 模型:     {model}")
    logger.info(f"  Temperature:  {temperature}")
    logger.info(f"  Save Plan:    {save_plan}")

    # 创建 Handler 生成器
    try:
        generator = HandlerGenerator()
    except Exception as e:
        logger.error(f"创建生成器失败: {e}")
        raise typer.Exit(1)

    # 执行两阶段生成
    try:
        result = generator.generate(
            target_doc_path=target_doc,
            source_kb_dir=kb_dir,
            target_name=target,
            top_k=top_k,
            model=model,
            temperature=temperature,
            output_dir=output_dir,
            save_plan=save_plan,
            intermediate_dir=intermediate_dir,
        )
    except Exception as e:
        logger.error(f"Handler 生成失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        raise typer.Exit(1)

    # 检查结果
    if result.error_message:
        logger.error(f"生成过程中出现错误: {result.error_message}")
        raise typer.Exit(1)

    if not result.is_valid:
        logger.warning("生成的代码可能不完整，请检查")

    # 输出结果摘要
    logger.info("")
    logger.info("=" * 60)
    logger.info("✅ Handler 生成完成！")
    logger.info("=" * 60)
    logger.info(f"  Target:       {result.target}")
    logger.info(f"  Source KB:    {result.source_kb}")
    logger.info(f"  Top-K:        {top_k}")
    logger.info(f"  检索片段数:   {result.retrieved_chunk_count}")
    logger.info(f"  LLM 模型:     {result.model}")
    logger.info(f"  代码文件:     {result.code_output_path}")
    if result.plan_output_path:
        logger.info(f"  设计计划:     {result.plan_output_path}")
    logger.info("")

    # 代码预览
    if result.code:
        code_lines = result.code.split('\n')
        preview_lines = code_lines[:30]
        logger.info("代码预览 (前30行):")
        for line in preview_lines:
            logger.info(f"  {line}")
        if len(code_lines) > 30:
            logger.info(f"  ... ({len(code_lines)} 行总计)")

    logger.info("=" * 60)


@app.callback()
def main() -> None:
    """
    生成目标协议代码和配置

    子命令：
    - config: 生成 config.json
    - handler: 生成 handler.c（两阶段生成：plan + code）
    """
    pass


if __name__ == "__main__":
    app()
