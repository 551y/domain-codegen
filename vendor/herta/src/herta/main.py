"""
项目主入口

可用于程序启动或调试入口
"""

from herta.cli.app import app


def main() -> None:
    """
    CLI 主入口函数

    该函数被配置为 herta 命令的入口点
    定义在 pyproject.toml: [project.scripts] herta = "herta.main:main"
    """
    app()


if __name__ == "__main__":
    main()
