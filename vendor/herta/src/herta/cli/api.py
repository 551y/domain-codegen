"""
API 服务命令

启动 FastAPI Web API 服务，供 Next.js 前端调用。
"""

import typer
from typing_extensions import Annotated

from herta.logging import get_logger

app = typer.Typer(help="启动 Web API 服务")

logger = get_logger(__name__)


@app.callback(invoke_without_command=True)
def serve(
    host: Annotated[
        str,
        typer.Option("--host", help="监听地址"),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="监听端口"),
    ] = 8000,
    reload: Annotated[
        bool,
        typer.Option("--reload", help="开发模式（热重载）"),
    ] = False,
) -> None:
    """启动 Web API 服务（默认 http://127.0.0.1:8000）"""
    try:
        import uvicorn
    except ImportError:
        typer.secho(
            "缺少 uvicorn，请先安装 Web 依赖: pip install 'herta[web]'",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    logger.info(f"启动 Herta Web API: http://{host}:{port}")
    uvicorn.run("herta.api.app:app", host=host, port=port, reload=reload)
