"""
Herta Web API - FastAPI 应用

提供知识库管理、代码生成、评测对比的 REST API。
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from herta.api.routes import kb, generate, compare, settings as settings_route
from herta.logging import get_logger

logger = get_logger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Herta API",
        description="工业协议代码生成系统 - RAG 驱动的协议处理器自动生成",
        version="0.1.0",
    )

    # CORS：前端经 next.config.mjs rewrite 同源代理，仅需放行本机开发端口
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3001",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    app.include_router(kb.router, prefix="/api/kb", tags=["知识库"])
    app.include_router(generate.router, prefix="/api/generate", tags=["代码生成"])
    app.include_router(compare.router, prefix="/api/compare", tags=["评测对比"])
    app.include_router(settings_route.router, prefix="/api/settings", tags=["设置"])

    @app.get("/api/health")
    async def health():
        """健康检查"""
        return {"status": "ok"}

    return app


app = create_app()
