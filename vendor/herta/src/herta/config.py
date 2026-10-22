"""
配置系统模块

支持从环境变量读取配置，提供 Settings dataclass 和 get_settings() 单例接口
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from herta.constants import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_CHAT_PROVIDER,
    DEFAULT_EMBEDDING_PROVIDER,
)


@dataclass
class Settings:
    """
    应用配置类

    LLM 支持两个供应商：
        - zai: 智谱 AI（默认，zai-sdk 原生协议）
        - openai: OpenAI 兼容协议（DeepSeek、硅基流动、Moonshot 等）

    Chat 与 Embedding 可分别配置供应商（DeepSeek 不提供 embedding 接口，
    典型用法是 chat 走 DeepSeek、embedding 仍走智谱）。

    属性:
        zai_api_key: 智谱 AI API 密钥（ZAI_API_KEY，向后兼容的统一密钥）
        chat_provider / chat_api_key / chat_base_url / chat_model: 对话配置
        embedding_provider / embedding_api_key / embedding_base_url / embedding_model: 向量配置
        embedding_dimensions: Embedding 向量维度
        data_dir: 数据目录路径
        output_dir: 输出目录路径
    """

    # 智谱 AI 配置（向后兼容：同时作为 chat/embedding 密钥的兜底）
    zai_api_key: str = field(default="")

    # Chat 供应商配置
    chat_provider: str = field(default=DEFAULT_CHAT_PROVIDER)
    chat_api_key: str = field(default="")
    chat_base_url: str = field(default="")
    chat_model: str = field(default=DEFAULT_CHAT_MODEL)

    # Embedding 供应商配置
    embedding_provider: str = field(default=DEFAULT_EMBEDDING_PROVIDER)
    embedding_api_key: str = field(default="")
    embedding_base_url: str = field(default="")
    embedding_model: str = field(default=DEFAULT_EMBEDDING_MODEL)
    embedding_dimensions: int = field(default=DEFAULT_EMBEDDING_DIMENSIONS)

    # 目录配置
    data_dir: str = field(default=DEFAULT_DATA_DIR)
    output_dir: str = field(default=DEFAULT_OUTPUT_DIR)

    def effective_chat_key(self) -> str:
        """Chat 实际使用的密钥：HERTA_CHAT_API_KEY > ZAI_API_KEY"""
        return self.chat_api_key or self.zai_api_key

    def effective_embedding_key(self) -> str:
        """Embedding 实际使用的密钥：HERTA_EMBEDDING_API_KEY > ZAI_API_KEY"""
        return self.embedding_api_key or self.zai_api_key

    def __post_init__(self):
        """
        初始化后不强制校验密钥：允许仅配置供应商/模型而不设密钥
        （如先保存设置再补 key）；密钥校验推迟到 LLM 客户端实际调用时进行，
        避免设置接口在无 key 环境下整体不可用。
        """
        pass


# 全局配置缓存
_settings_instance: Optional[Settings] = None


def _get_env_or_default(env_name: str, default_value: str) -> str:
    """从环境变量获取值，如果不存在则返回默认值"""
    return os.environ.get(env_name, default_value)


def _get_env_int_or_default(env_name: str, default_value: int) -> int:
    """从环境变量获取整数值，如果不存在或解析失败则返回默认值"""
    env_value = os.environ.get(env_name)
    if env_value is None:
        return default_value
    try:
        return int(env_value)
    except ValueError:
        return default_value


def load_settings_from_env() -> Settings:
    """
    从环境变量加载配置

    支持的环境变量:
        - ZAI_API_KEY: 智谱 AI API 密钥（向后兼容的统一密钥）
        - HERTA_CHAT_PROVIDER: 对话供应商（zai / openai，默认 zai）
        - HERTA_CHAT_API_KEY: 对话密钥（缺省回退 ZAI_API_KEY）
        - HERTA_CHAT_BASE_URL: 对话接口地址（openai 供应商时必填，如 https://api.deepseek.com）
        - HERTA_CHAT_MODEL: 对话模型名称
        - HERTA_EMBEDDING_PROVIDER / HERTA_EMBEDDING_API_KEY / HERTA_EMBEDDING_BASE_URL: 向量同上
        - HERTA_EMBEDDING_MODEL: Embedding 模型名称
        - HERTA_EMBEDDING_DIMENSIONS: Embedding 向量维度
        - HERTA_DATA_DIR: 数据目录路径
        - HERTA_OUTPUT_DIR: 输出目录路径

    Returns:
        Settings 配置对象

    Raises:
        ValueError: 当所有密钥均未设置时
    """
    return Settings(
        zai_api_key=_get_env_or_default("ZAI_API_KEY", ""),
        chat_provider=_get_env_or_default("HERTA_CHAT_PROVIDER", DEFAULT_CHAT_PROVIDER),
        chat_api_key=_get_env_or_default("HERTA_CHAT_API_KEY", ""),
        chat_base_url=_get_env_or_default("HERTA_CHAT_BASE_URL", ""),
        chat_model=_get_env_or_default("HERTA_CHAT_MODEL", DEFAULT_CHAT_MODEL),
        embedding_provider=_get_env_or_default("HERTA_EMBEDDING_PROVIDER", DEFAULT_EMBEDDING_PROVIDER),
        embedding_api_key=_get_env_or_default("HERTA_EMBEDDING_API_KEY", ""),
        embedding_base_url=_get_env_or_default("HERTA_EMBEDDING_BASE_URL", ""),
        embedding_model=_get_env_or_default(
            "HERTA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        ),
        embedding_dimensions=_get_env_int_or_default(
            "HERTA_EMBEDDING_DIMENSIONS", DEFAULT_EMBEDDING_DIMENSIONS
        ),
        data_dir=_get_env_or_default("HERTA_DATA_DIR", DEFAULT_DATA_DIR),
        output_dir=_get_env_or_default("HERTA_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
    )


def get_settings() -> Settings:
    """
    获取配置单例

    首次调用时会从环境变量加载配置，后续调用返回缓存实例

    Returns:
        Settings 配置对象

    Example:
        >>> from herta.config import get_settings
        >>> settings = get_settings()
        >>> print(settings.chat_model)
    """
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = load_settings_from_env()
    return _settings_instance


def reload_settings() -> Settings:
    """
    重新加载配置（用于配置变更后刷新）

    Returns:
        新的 Settings 配置对象
    """
    global _settings_instance
    _settings_instance = load_settings_from_env()
    return _settings_instance
