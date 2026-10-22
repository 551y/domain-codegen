"""
设置路由
"""

import os
from fastapi import APIRouter
from pydantic import BaseModel

from herta.api.task_store import tasks
from herta.config import get_settings
from herta.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class UpdateSettingsRequest(BaseModel):
    api_key: str = ""            # 智谱统一密钥（向后兼容，写入 ZAI_API_KEY）
    chat_model: str = ""
    embedding_model: str = ""
    chat_provider: str = ""      # zai / openai
    chat_base_url: str = ""
    chat_api_key: str = ""
    embedding_provider: str = ""
    embedding_base_url: str = ""
    embedding_api_key: str = ""


def _preview(key: str) -> str:
    return (key[:8] + "...") if key else ""


@router.get("")
async def get_settings_api():
    """获取当前设置"""
    try:
        s = get_settings()
        return {
            "api_key_set": bool(s.effective_chat_key() or s.effective_embedding_key()),
            "api_key_preview": _preview(s.zai_api_key),
            "chat_provider": s.chat_provider,
            "chat_api_key_preview": _preview(s.effective_chat_key()),
            "chat_base_url": s.chat_base_url,
            "chat_model": s.chat_model,
            "embedding_provider": s.embedding_provider,
            "embedding_api_key_preview": _preview(s.effective_embedding_key()),
            "embedding_base_url": s.embedding_base_url,
            "embedding_model": s.embedding_model,
            "embedding_dimensions": s.embedding_dimensions,
        }
    except Exception:
        return {
            "api_key_set": bool(os.environ.get("ZAI_API_KEY")),
            "api_key_preview": "",
            "chat_provider": "zai",
            "chat_api_key_preview": "",
            "chat_base_url": "",
            "chat_model": "glm-5.1",
            "embedding_provider": "zai",
            "embedding_api_key_preview": "",
            "embedding_base_url": "",
            "embedding_model": "embedding-3",
            "embedding_dimensions": 1024,
        }


@router.post("")
async def update_settings(req: UpdateSettingsRequest):
    """更新设置（仅当前进程生效，重启后恢复环境变量配置）"""
    from herta.config import reload_settings

    mapping = {
        "ZAI_API_KEY": req.api_key,
        "HERTA_CHAT_MODEL": req.chat_model,
        "HERTA_EMBEDDING_MODEL": req.embedding_model,
        "HERTA_CHAT_PROVIDER": req.chat_provider,
        "HERTA_CHAT_BASE_URL": req.chat_base_url,
        "HERTA_CHAT_API_KEY": req.chat_api_key,
        "HERTA_EMBEDDING_PROVIDER": req.embedding_provider,
        "HERTA_EMBEDDING_BASE_URL": req.embedding_base_url,
        "HERTA_EMBEDDING_API_KEY": req.embedding_api_key,
    }
    for env_name, value in mapping.items():
        if value:
            os.environ[env_name] = value

    reload_settings()
    logger.info("设置已更新并刷新缓存")
    return {"status": "ok"}


@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """获取后台任务状态"""
    if task_id not in tasks:
        return {"exists": False}

    task = tasks[task_id]
    return {
        "exists": True,
        "type": task["type"],
        "status": task["status"],
        "progress": task["progress"],
        "message": task["message"],
        "result": task["result"],
        "error": task["error"],
    }
