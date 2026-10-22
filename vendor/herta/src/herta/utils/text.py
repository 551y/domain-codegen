"""
文本处理工具函数模块

提供文本清洗、格式化等基础功能
"""

import re
from typing import Optional


def normalize_text(
    text: str,
    encoding: str = "utf-8",
    preserve_newlines: bool = True,
) -> str:
    """
    标准化文本

    处理内容包括：
    - 统一换行符为 \\n
    - 去除 BOM 头
    - 去除首尾空白
    - 规范化空白字符（可选保留换行）

    Args:
        text: 原始文本
        encoding: 编码（用于处理 bytes 输入，当前简化版主要处理 str）
        preserve_newlines: 是否保留换行符，默认为 True

    Returns:
        标准化后的文本

    Example:
        >>> raw = "  Hello\\r\\nWorld\\r\t\t\n  "
        >>> clean = normalize_text(raw)
        >>> print(clean)  # "Hello\\nWorld"
    """
    if not isinstance(text, str):
        # 如果是 bytes，尝试解码
        if isinstance(text, bytes):
            text = text.decode(encoding, errors="replace")
        else:
            text = str(text)

    # 去除 BOM 头
    if text.startswith("\ufeff"):
        text = text[1:]

    # 统一换行符
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if preserve_newlines:
        # 保留换行；仅去行尾空白，保留行首缩进（代码的缩进即语义结构）
        lines = text.split("\n")
        lines = [line.rstrip() for line in lines]
        text = "\n".join(lines)
    else:
        # 将所有空白（包括换行）替换为单个空格
        text = " ".join(text.split())

    # 去除首尾空白
    return text.strip()


def strip_extra_blank_lines(
    text: str,
    max_consecutive: int = 2,
) -> str:
    """
    去除多余的连续空行

    Args:
        text: 原始文本
        max_consecutive: 允许的最大连续空行数，默认为 2

    Returns:
        处理后的文本

    Example:
        >>> raw = "Line 1\\n\\n\\n\\nLine 2"
        >>> clean = strip_extra_blank_lines(raw, max_consecutive=1)
        >>> print(clean)  # "Line 1\\nLine 2"
    """
    if not text:
        return ""

    lines = text.split("\n")
    result = []
    blank_count = 0

    for line in lines:
        is_blank = not line.strip()

        if is_blank:
            blank_count += 1
            if blank_count <= max_consecutive:
                result.append(line)
        else:
            blank_count = 0
            result.append(line)

    # 去除尾部的多余空行
    while result and not result[-1].strip():
        result.pop()

    return "\n".join(result)


def truncate_text(
    text: str,
    max_length: int,
    suffix: str = "...",
) -> str:
    """
    截断文本到指定长度

    Args:
        text: 原始文本
        max_length: 最大长度（包含后缀）
        suffix: 截断后缀，默认为 "..."

    Returns:
        截断后的文本

    Example:
        >>> text = "这是一段很长的文本内容"
        >>> short = truncate_text(text, 10)
        >>> print(short)  # "这是一段很..."
    """
    if len(text) <= max_length:
        return text

    if max_length <= len(suffix):
        return suffix[:max_length]

    return text[: max_length - len(suffix)] + suffix


def count_tokens_approximate(
    text: str,
    chars_per_token: float = 2.0,
) -> int:
    """
    估算文本的 token 数量（简化版，基于字符数）

    注意：这是粗略估算，实际 token 数取决于具体 tokenizer

    Args:
        text: 原始文本
        chars_per_token: 每个 token 的平均字符数，中文建议 1.5-2.0

    Returns:
        估算的 token 数量

    Example:
        >>> tokens = count_tokens_approximate("Hello 世界", chars_per_token=2.0)
        >>> print(tokens)  # 约 5
    """
    if not text:
        return 0

    return int(len(text) / chars_per_token) + 1


def clean_code_comments(
    code: str,
    language: str = "c",
) -> str:
    """
    去除代码中的注释（简化版）

    当前仅支持 C 风格注释（// 和 /* */）

    Args:
        code: 原始代码
        language: 编程语言，默认为 "c"

    Returns:
        去除注释后的代码

    Note:
        这是简化版实现，复杂场景（如字符串中的注释标记）可能处理不完美

    Example:
        >>> code = "int x = 1; // 变量\\nint y = 2;"
        >>> clean = clean_code_comments(code)
        >>> print(clean)  # "int x = 1; \\nint y = 2;"
    """
    if language.lower() in ("c", "cpp", "java", "javascript", "json"):
        # 去除 /* */ 多行注释
        code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
        # 去除 // 单行注释
        code = re.sub(r"//.*?$", "", code, flags=re.MULTILINE)

    return strip_extra_blank_lines(code, max_consecutive=1)


def extract_section_keywords(
    text: str, keywords: list[str], window: int = 200
) -> str:
    """
    从文本中按关键词提取上下文片段

    在 text 中查找 keywords 中的关键词，取每个关键词周围 window 字符的文本，
    拼接后返回。适合用于构造 RAG 检索 query。

    Args:
        text: 完整文本
        keywords: 关键词列表
        window: 提取窗口大小

    Returns:
        拼接的上下文片段
    """
    snippets = []
    text_lower = text.lower()
    for kw in keywords:
        pos = text_lower.find(kw.lower())
        if pos >= 0:
            start = max(0, pos - window // 2)
            end = min(len(text), pos + window // 2)
            snippets.append(text[start:end].strip())
    return " ... ".join(snippets) if snippets else ""
