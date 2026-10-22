"""
compare 子命令实现

对比 LLM 生成的 config.json 与 handler.c 和预期真实文件之间的相似度，
输出 metrics.json 与 comparison_report.md。
"""

from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

from herta.logging import get_logger
from herta.utils.paths import ensure_dir, project_relative
from herta.compare import (
    compare_configs,
    parse_c_functions,
    match_functions,
    generate_comparison_report,
)

app = typer.Typer(
    name="compare",
    help="对比评测 - 比较生成结果与真实代码的相似度",
    rich_markup_mode="rich",
)

logger = get_logger(__name__)


@app.callback(invoke_without_command=True)
def main(
    target: Annotated[
        str,
        typer.Option(
            "--target",
            "-t",
            help="目标协议标识（如 dllg）",
            rich_help_panel="输入选项",
        ),
    ],
    gen_config: Annotated[
        Optional[Path],
        typer.Option(
            "--gen-config",
            "-gc",
            help="生成的 config.json 路径",
            rich_help_panel="输入选项",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
    gt_config: Annotated[
        Optional[Path],
        typer.Option(
            "--gt-config",
            "-gc2",
            help="真实的 config.json 路径",
            rich_help_panel="输入选项",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
    gen_handler: Annotated[
        Optional[Path],
        typer.Option(
            "--gen-handler",
            "-gh",
            help="生成的 handler.c 路径",
            rich_help_panel="输入选项",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
    gt_handler: Annotated[
        Optional[Path],
        typer.Option(
            "--gt-handler",
            "-gh2",
            help="真实的 handler.c 路径",
            rich_help_panel="输入选项",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
    report: Annotated[
        Optional[Path],
        typer.Option(
            "--report",
            "-r",
            help="Markdown 报告输出路径",
            rich_help_panel="输出选项",
        ),
    ] = None,
    metrics: Annotated[
        Optional[Path],
        typer.Option(
            "--metrics",
            "-m",
            help="JSON 指标文件输出路径",
            rich_help_panel="输出选项",
        ),
    ] = None,
) -> None:
    """
    对比评测：比较生成结果与真实代码的相似度

    比较维度：
    - Config JSON：拓扑结构、命令表、遥测表、内部处理表
    - Handler C：基于 AST 的函数级相似度

    输出：
    - metrics.json：结构化指标数据
    - comparison_report.md：可读性强的 Markdown 报告

    默认路径规则（当只提供 --target 时）：
    - 生成配置: output/generated/<target>/<target>_config.json
    - 生成代码: output/generated/<target>/<target>_handler.c
    - 真实配置: data/target_proto/<target>/<target>_config.json
    - 真实代码: data/target_proto/<target>/<target>_handler.c
    - 报告输出: output/reports/<target>/

    示例：
        # 使用默认路径
        herta compare --target dllg

        # 手动指定路径
        herta compare --target dllg \\
            --gen-config ./output/dllg_config.json \\
            --gt-config ./data/dllg_config.json \\
            --gen-handler ./output/dllg_handler.c \\
            --gt-handler ./data/dllg_handler.c
    """
    logger.info("=" * 60)
    logger.info("开始执行 compare 命令")
    logger.info("=" * 60)

    # 推断默认路径
    if gen_config is None:
        gen_config = project_relative(f"output/generated/{target}/{target}_config.json")
    if gt_config is None:
        gt_config = project_relative(f"data/target_proto/{target}/{target}_config.json")
    if gen_handler is None:
        gen_handler = project_relative(f"output/generated/{target}/{target}_handler.c")
    if gt_handler is None:
        gt_handler = project_relative(f"data/target_proto/{target}/{target}_handler.c")

    # 确定报告输出目录
    if report is None and metrics is None:
        report_dir = project_relative(f"output/reports/{target}")
    else:
        # 如果指定了其中一个，使用另一个的目录或默认
        if report:
            report_dir = report.parent
        elif metrics:
            report_dir = metrics.parent
        else:
            report_dir = project_relative(f"output/reports/{target}")

    report_dir = ensure_dir(report_dir)

    if report is None:
        report = report_dir / "comparison_report.md"
    if metrics is None:
        metrics = report_dir / "metrics.json"

    logger.info(f"Target: {target}")
    logger.info(f"生成配置: {gen_config}")
    logger.info(f"真实配置: {gt_config}")
    logger.info(f"生成代码: {gen_handler}")
    logger.info(f"真实代码: {gt_handler}")
    logger.info(f"报告输出: {report_dir}")

    # 检查文件是否存在
    for path, name in [
        (gen_config, "生成配置"),
        (gt_config, "真实配置"),
        (gen_handler, "生成代码"),
        (gt_handler, "真实代码"),
    ]:
        if not path.exists():
            logger.error(f"{name} 文件不存在: {path}")
            raise typer.Exit(1)

    # 1. 比较 Config JSON
    logger.info("-" * 60)
    logger.info("步骤 1: 比较 Config JSON...")
    try:
        config_result = compare_configs(str(gen_config), str(gt_config))
        logger.info(f"✓ Config 比较完成")
        logger.info(f"  拓扑相似度: {config_result.topology_similarity.score:.2f}%")
        logger.info(f"  命令表相似度: {config_result.cmd_list_similarity.score:.2f}%")
        logger.info(f"  遥测表相似度: {config_result.tele_handler_similarity.score:.2f}%")
        logger.info(f"  内部处理相似度: {config_result.internal_handler_similarity.score:.2f}%")
        logger.info(f"  Config 总分: {config_result.config_similarity:.2f}%")
    except Exception as e:
        logger.error(f"Config 比较失败: {e}")
        raise typer.Exit(1)

    # 2. 解析 Handler C 代码
    logger.info("-" * 60)
    logger.info("步骤 2: 解析 Handler C 代码...")
    try:
        gen_functions = parse_c_functions(str(gen_handler))
        gt_functions = parse_c_functions(str(gt_handler))
        logger.info(f"✓ 解析完成: 生成 {len(gen_functions)} 个函数, 真实 {len(gt_functions)} 个函数")
    except Exception as e:
        logger.error(f"Handler 解析失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        raise typer.Exit(1)

    # 3. 函数匹配
    logger.info("-" * 60)
    logger.info("步骤 3: 函数匹配...")
    try:
        handler_result = match_functions(gen_functions, gt_functions)
        logger.info(f"✓ 函数匹配完成")
        logger.info(f"  成功匹配: {len([m for m in handler_result.matches if m.is_matched])} 对")
        logger.info(f"  未匹配真实: {len(handler_result.unmatched_gt_functions)} 个")
        logger.info(f"  未匹配生成: {len(handler_result.unmatched_gen_functions)} 个")
        logger.info(f"  Handler 相似度: {handler_result.handler_similarity:.2f}%")
    except Exception as e:
        logger.error(f"函数匹配失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        raise typer.Exit(1)

    # 4. 生成报告
    logger.info("-" * 60)
    logger.info("步骤 4: 生成报告...")
    try:
        metrics_path, report_path = generate_comparison_report(
            config_result=config_result,
            handler_result=handler_result,
            target=target,
            output_dir=str(report_dir),
        )
        logger.info(f"✓ 报告生成完成")
        logger.info(f"  Metrics: {metrics_path}")
        logger.info(f"  Report: {report_path}")
    except Exception as e:
        logger.error(f"报告生成失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        raise typer.Exit(1)

    # 输出总结
    logger.info("")
    logger.info("=" * 60)
    logger.info(" 对比评测完成！")
    logger.info("=" * 60)



    logger.info(f"  Target: {target}")
    logger.info(f"  Config 相似度: {config_result.config_similarity:.2f}%")
    logger.info(f"  Handler 相似度: {handler_result.handler_similarity:.2f}%")
    logger.info("")
    logger.info("  输出文件:")
    logger.info(f"    - {metrics_path}")
    logger.info(f"    - {report_path}")
    logger.info("=" * 60)




if __name__ == "__main__":
    app()
