"""
CLI 根应用入口

使用 Typer 构建命令行接口，注册所有子命令
"""

import typer
from typing_extensions import Annotated

from herta import configure_logging, get_settings
from herta.cli import api, buildkb, generate, compare
from herta.logging import get_logger

# 创建主应用
app = typer.Typer(
    name="herta",
    help="Herta - 基于 RAG 的协议生成系统",
    rich_markup_mode="rich",
    add_completion=True,
)

logger = get_logger(__name__)

# 注册子命令
app.add_typer(buildkb.app, name="buildkb")
app.add_typer(generate.app, name="generate")
app.add_typer(compare.app, name="compare")
app.add_typer(api.app, name="api")


@app.callback()
def main(
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="启用详细日志输出",
            rich_help_panel="全局选项",
        ),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet",
            "-q",
            help="只输出错误信息",
            rich_help_panel="全局选项",
        ),
    ] = False,
) -> None:
    """
    Herta - 基于 RAG 的协议生成系统

    当前支持的功能：
    - buildkb: 构建知识库
    - generate: 基于目标协议生成代码
    - compare: 对比评测生成结果与真实代码
    - api: 启动 Web API 服务

    示例：
        herta buildkb --help
        herta generate --help
        herta compare --help
        herta api
    """
    # 配置日志级别
    if verbose:
        configure_logging("DEBUG")
        logger.debug("已启用详细日志模式")
    elif quiet:
        configure_logging("ERROR")
    else:
        configure_logging("INFO")

    # 验证配置（仅检查环境变量是否设置，不创建客户端）
    try:
        settings = get_settings()
        logger.debug(f"配置加载成功，数据目录: {settings.data_dir}")
    except ValueError as e:
        # API 密钥未设置，在非交互模式下只记录警告
        logger.warning(f"配置警告: {e}")


# 入口函数
def run() -> None:
    """CLI 入口函数"""
    app()


if __name__ == "__main__":
    run()
