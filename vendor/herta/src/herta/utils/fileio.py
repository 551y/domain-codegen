"""
文件读写工具函数模块

提供文本文件和 JSON 文件的读写功能
"""

import json
import os
from pathlib import Path
from typing import Any, Union

from herta.constants import DEFAULT_ENCODING


def read_text(
    file_path: Union[str, Path],
    encoding: str = DEFAULT_ENCODING,
    errors: str = "strict",
) -> str:
    """
    读取文本文件内容

    Args:
        file_path: 文件路径（字符串或 Path 对象）
        encoding: 文件编码，默认为 utf-8
        errors: 编码错误处理方式，默认为 strict

    Returns:
        文件内容字符串

    Raises:
        FileNotFoundError: 文件不存在时
        IOError: 读取失败时

    Example:
        >>> content = read_text("/path/to/file.txt")
        >>> print(content)
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    if not path.is_file():
        raise IOError(f"路径不是文件: {path}")

    try:
        return path.read_text(encoding=encoding, errors=errors)
    except Exception as e:
        raise IOError(f"读取文件失败 {path}: {e}") from e


def write_text(
    file_path: Union[str, Path],
    content: str,
    encoding: str = DEFAULT_ENCODING,
    newline: str = "\n",
) -> None:
    """
    写入文本文件内容

    如果父目录不存在，会自动创建

    Args:
        file_path: 文件路径（字符串或 Path 对象）
        content: 要写入的文本内容
        encoding: 文件编码，默认为 utf-8
        newline: 换行符，默认为 \\n
    Raises:
        IOError: 写入失败时

    Example:
        >>> write_text("/path/to/file.txt", "Hello World")
    """
    path = Path(file_path)

    # 确保父目录存在
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        path.write_text(content, encoding=encoding, newline=newline)
    except Exception as e:
        raise IOError(f"写入文件失败 {path}: {e}") from e


def write_json(
    file_path: Union[str, Path],
    data: Any,
    encoding: str = DEFAULT_ENCODING,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    """
    写入 JSON 文件

    如果父目录不存在，会自动创建

    Args:
        file_path: 文件路径（字符串或 Path 对象）
        data: 要序列化的数据（可以是 dict、list 等）
        encoding: 文件编码，默认为 utf-8
        indent: JSON 缩进空格数，默认为 2
        ensure_ascii: 是否转义非 ASCII 字符，默认为 False（支持中文）

    Raises:
        TypeError: 数据无法序列化为 JSON 时
        IOError: 写入失败时

    Example:
        >>> data = {"name": "测试", "value": 123}
        >>> write_json("/path/to/data.json", data)
    """
    path = Path(file_path)

    # 确保父目录存在
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        json_str = json.dumps(
            data,
            indent=indent,
            ensure_ascii=ensure_ascii,
            default=str,  # 处理无法序列化的类型
        )
        path.write_text(json_str, encoding=encoding)
    except TypeError as e:
        raise TypeError(f"数据无法序列化为 JSON: {e}") from e
    except Exception as e:
        raise IOError(f"写入 JSON 文件失败 {path}: {e}") from e


def read_json(
    file_path: Union[str, Path],
    encoding: str = DEFAULT_ENCODING,
) -> Any:
    """
    读取 JSON 文件

    Args:
        file_path: 文件路径（字符串或 Path 对象）
        encoding: 文件编码，默认为 utf-8

    Returns:
        解析后的 JSON 数据

    Raises:
        FileNotFoundError: 文件不存在时
        json.JSONDecodeError: JSON 解析失败时

    Example:
        >>> data = read_json("/path/to/data.json")
        >>> print(data["name"])
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    try:
        content = path.read_text(encoding=encoding)
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"JSON 解析失败 {path}: {e.msg}",
            e.doc,
            e.pos,
        ) from e
    except Exception as e:
        raise IOError(f"读取 JSON 文件失败 {path}: {e}") from e
