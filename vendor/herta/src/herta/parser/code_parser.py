"""
代码解析模块

支持解析代码文件：
- .json - 配置文件
- .c - C 源代码
- .h - C 头文件

直接按文本方式读取，保留文件结构和注释
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Dict, Any, TYPE_CHECKING

from herta.utils.text import normalize_text, strip_extra_blank_lines
from herta.utils.fileio import read_text
from herta.logging import get_logger

if TYPE_CHECKING:
    from herta.kb.schemas import SourceDocument

logger = get_logger(__name__)


class CodeParser:
    """
    代码文件解析器

    支持 json, c, h 等代码文件
    """

    SUPPORTED_EXTS = ['.json', '.c', '.h']

    def __init__(self):
        self.stats = {
            "parsed": 0,
            "failed": 0,
        }

    def parse(self, file_path: Path, source: str) -> List[SourceDocument]:
        """
        解析代码文件

        Args:
            file_path: 文件路径
            source: 来源标识

        Returns:
            SourceDocument 列表

        Raises:
            ValueError: 不支持的文件格式
            FileNotFoundError: 文件不存在
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = file_path.suffix.lower()

        if ext == '.json':
            return self._parse_json(file_path, source)
        elif ext in ['.c', '.h']:
            return self._parse_c_file(file_path, source)
        else:
            raise ValueError(
                f"不支持的文件格式: {ext}\n"
                f"支持格式: {', '.join(self.SUPPORTED_EXTS)}"
            )

    def _parse_json(self, file_path: Path, source: str) -> List[SourceDocument]:
        """
        解析 JSON 配置文件

        读取整个文件作为单个文档，同时尝试提取关键字段
        """
        from herta.kb.schemas import SourceDocument

        logger.info(f"解析 json: {file_path}")

        try:
            content = read_text(file_path)

            # 尝试解析为 JSON 以验证有效性
            try:
                data = json.loads(content)
                is_valid_json = True
            except json.JSONDecodeError:
                is_valid_json = False
                logger.warning(f"JSON 解析警告（将作为纯文本处理）: {file_path}")
                data = None

            # 创建文档对象
            documents = []

            # 主文档：完整内容
            main_doc = SourceDocument(
                source=source,
                file=str(file_path),
                kind="code_json",
                section="full_content",
                text=content,
                metadata={
                    "valid_json": is_valid_json,
                    "file_type": "json",
                }
            )
            documents.append(main_doc)

            # 如果 JSON 有效，尝试提取各个字段作为独立文档
            if is_valid_json and isinstance(data, dict):
                for key, value in data.items():
                    # 将复杂值转换为字符串表示
                    if isinstance(value, (dict, list)):
                        value_text = json.dumps(value, ensure_ascii=False, indent=2)
                    else:
                        value_text = str(value)

                    field_doc = SourceDocument(
                        source=source,
                        file=str(file_path),
                        kind="code_json",
                        section=f"field_{key}",
                        text=f"{key}: {value_text}",
                        metadata={
                            "field_name": key,
                            "field_type": type(value).__name__,
                            "file_type": "json",
                        }
                    )
                    documents.append(field_doc)

            self.stats["parsed"] += 1
            logger.info(f"json 解析完成: {file_path} -> {len(documents)} 个文档")
            return documents

        except Exception as e:
            logger.error(f"解析 json 失败: {file_path}, 错误: {e}")
            self.stats["failed"] += 1
            raise

    def _parse_c_file(self, file_path: Path, source: str) -> List[SourceDocument]:
        """
        解析 C 代码文件（.c 或 .h）

        读取整个文件，同时尝试提取函数定义作为独立文档
        """
        from herta.kb.schemas import SourceDocument

        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        kind = "code_c" if ext == ".c" else "code_h"

        logger.info(f"解析 {ext} 文件: {file_path}")

        try:
            content = read_text(file_path)
            content = normalize_text(content, preserve_newlines=True)

            documents = []

            # 主文档：完整代码
            main_doc = SourceDocument(
                source=source,
                file=str(file_path),
                kind=kind,
                section="full_content",
                text=content,
                metadata={
                    "file_type": ext[1:],  # 去掉点号
                    "line_count": len(content.split('\n')),
                }
            )
            documents.append(main_doc)

            # 尝试提取函数定义（简化版：按大括号匹配）
            functions = self._extract_c_functions(content)
            for func_idx, func_text in enumerate(functions):
                func_doc = SourceDocument(
                    source=source,
                    file=str(file_path),
                    kind=kind,
                    section=f"function_{func_idx:04d}",
                    text=func_text,
                    metadata={
                        "type": "function",
                        "function_index": func_idx,
                        "file_type": ext[1:],
                    }
                )
                documents.append(func_doc)

            self.stats["parsed"] += 1
            logger.info(f"{ext} 解析完成: {file_path} -> {len(documents)} 个文档（含 {len(functions)} 个函数）")
            return documents

        except Exception as e:
            logger.error(f"解析 {ext} 文件失败: {file_path}, 错误: {e}")
            self.stats["failed"] += 1
            raise

    def _extract_c_functions(self, content: str) -> List[str]:
        """
        从 C 代码中提取函数定义（简化版）

        基于大括号匹配进行简单提取，不完美但可用
        """
        functions = []
        lines = content.split('\n')

        func_start = -1
        brace_count = 0
        in_function = False

        for i, line in enumerate(lines):
            stripped = line.strip()

            # 检测函数开始（简化：以 { 结尾且包含 ( ）
            if not in_function and stripped.endswith('{') and '(' in stripped:
                # 向前查找函数签名
                func_lines = []
                for j in range(max(0, i-5), i+1):
                    func_lines.append(lines[j])

                func_start = i
                brace_count = 1
                in_function = True

            elif in_function:
                brace_count += stripped.count('{')
                brace_count -= stripped.count('}')

                if brace_count == 0:
                    # 函数结束
                    func_lines = lines[func_start:i+1]
                    func_text = '\n'.join(func_lines)
                    functions.append(func_text)
                    in_function = False

        return functions

    def can_parse(self, file_path: Path) -> bool:
        """
        检查是否可以解析该文件

        Args:
            file_path: 文件路径

        Returns:
            是否支持解析
        """
        ext = Path(file_path).suffix.lower()
        return ext in self.SUPPORTED_EXTS


def parse_code(file_path: str, source: str) -> List[SourceDocument]:
    """
    便捷函数：解析单个代码文件

    Args:
        file_path: 文件路径
        source: 来源标识

    Returns:
        SourceDocument 列表

    Example:
        >>> docs = parse_code("data/source_proto/gdwx/gdwx_config.json", "gdwx")
        >>> print(len(docs))
    """
    parser = CodeParser()
    return parser.parse(Path(file_path), source)
