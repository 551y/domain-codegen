"""
日志系统模块

提供 configure_logging() 函数，配置简洁的控制台日志输出
"""

import logging
import sys
from typing import Optional

from herta.constants import DEFAULT_LOG_LEVEL, LOG_FORMAT_CONSOLE


def configure_logging(
    level: Optional[str] = None,
    format_string: Optional[str] = None,
    use_color: bool = True,
) -> None:
    """
    配置日志系统

    默认配置:
        - 输出级别: INFO
        - 输出目标: 控制台（stdout）
        - 格式: 简洁的消息格式

    Args:
        level: 日志级别，默认为 "INFO"
        format_string: 自定义格式字符串
        use_color: 是否使用彩色输出（简化版暂不支持，预留参数）

    Example:
        >>> from herta.logging import configure_logging
        >>> configure_logging("DEBUG")
        >>> logging.debug("调试信息")
        >>> logging.info("普通信息")
    """
    log_level = level or DEFAULT_LOG_LEVEL
    log_format = format_string or LOG_FORMAT_CONSOLE

    # 配置根日志记录器
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # 强制重新配置
    )

    # 降低第三方库的日志级别，减少干扰
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的日志记录器

    Args:
        name: 日志记录器名称，通常使用 __name__

    Returns:
        logging.Logger 实例

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("这是一条信息")
    """
    return logging.getLogger(name)
