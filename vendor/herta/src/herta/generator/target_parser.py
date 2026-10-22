"""
目标协议文档解析模块

解析目标协议文档，提取结构化信息
支持格式：docx, pdf（doc 需要转换）
"""

from pathlib import Path
import re
from typing import Optional, List, Dict, Any

from herta.generator.schemas import TargetProtocolSpec
from herta.parser.doc_parser import DocParser, parse_document
from herta.utils.text import normalize_text, strip_extra_blank_lines
from herta.logging import get_logger

logger = get_logger(__name__)


class TargetParser:
    """
    目标协议文档解析器

    将目标协议文档解析为结构化的 TargetProtocolSpec
    """

    SUPPORTED_EXTS = ['.docx', '.pdf']

    def __init__(self):
        self.doc_parser = DocParser()

    def parse(self, file_path: Path, target_name: Optional[str] = None) -> TargetProtocolSpec:
        """
        解析目标协议文档

        Args:
            file_path: 文档路径
            target_name: 目标协议名称，默认从文件名推断

        Returns:
            TargetProtocolSpec 结构化规范对象

        Raises:
            ValueError: 不支持的文件格式
            FileNotFoundError: 文件不存在
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 确定目标名称
        if target_name is None:
            target_name = file_path.stem

        ext = file_path.suffix.lower()

        # 检查格式支持
        if ext == '.doc':
            raise ValueError(
                f"不支持 .doc 格式: {file_path}\n"
                f"请将文档转换为 .docx 或 .pdf 格式后再解析。\n"
                f"转换方式：使用 Microsoft Word 或 LibreOffice 另存为 docx/pdf"
            )

        if ext not in self.SUPPORTED_EXTS:
            raise ValueError(
                f"不支持的文件格式: {ext}\n"
                f"支持格式: {', '.join(self.SUPPORTED_EXTS)}"
            )

        logger.info(f"解析目标协议文档: {file_path}")

        # 使用 doc_parser 解析文档
        try:
            source_docs = parse_document(str(file_path), target_name)
        except Exception as e:
            logger.error(f"解析文档失败: {file_path}, 错误: {e}")
            raise

        # 合并所有文本
        full_text = "\n\n".join(doc.text for doc in source_docs)
        full_text = strip_extra_blank_lines(full_text)

        # 提取章节结构
        sections = self._extract_sections(source_docs)

        # 生成摘要（简化版：提取前500字符作为摘要）
        summary = self._generate_summary(full_text)

        spec = TargetProtocolSpec(
            target_name=target_name,
            document_path=str(file_path),
            raw_text=full_text,
            summary=summary,
            sections=sections,
            metadata={
                "file_size": file_path.stat().st_size,
                "doc_count": len(source_docs),
                "text_length": len(full_text),
            }
        )

        logger.info(f"目标协议解析完成: {target_name}, 文本长度: {len(full_text)}, 章节数: {len(sections)}")
        return spec

    def _extract_sections(self, source_docs: List[Any]) -> List[Dict[str, Any]]:
        """
        从解析的文档中提取章节结构

        Args:
            source_docs: SourceDocument 列表

        Returns:
            章节列表
        """
        sections = []

        for doc in source_docs:
            section = {
                "name": doc.section,
                "kind": doc.kind,
                "file": doc.file,
                "preview": doc.text[:200] + "..." if len(doc.text) > 200 else doc.text,
            }
            sections.append(section)

        return sections

    def _generate_summary(self, text: str, max_length: int = 16000) -> str:
        """
        生成文本摘要（结构化提取版）

        尽可能保留完整文档内容，仅在文档极长时截断。
        """
        if len(text) <= max_length:
            return text.strip()

        # 文档很长时，保留开头 + 各关键段落 + 数量提示
        sections = []

        # 开头 2000 字符（协议概述）
        sections.append(f"[协议概述] {text[:2000]}")

        # 按关键词类别提取关键片段，每个窗口放大到 2000
        key_terms = [
            ("命令列表", ["命令", "指令", "命令字", "command", "cmd"]),
            ("遥测帧", ["遥测", "帧格式", "数据帧", "telemetry", "frame"]),
            ("内部处理", ["内部处理", "内部通信", "同步", "校验", "internal"]),
            ("配置参数", ["版本", "作者", "起始", "version", "author"]),
            ("处理函数", ["处理函数", "handler", "入口", "entry", "分发", "dispatch"]),
            ("数据结构", ["数据结构", "结构体", "帧格式", "数据包", "struct"]),
        ]

        for category, terms in key_terms:
            snippet = self._extract_keyword_snippet(text, terms, window=2000)
            if snippet:
                sections.append(f"[{category}] {snippet}")

        # 提取数量信息
        counts = self._extract_counts(text)
        count_lines = [f"【数量提示】{k}: {v}" for k, v in counts.items() if v]
        if count_lines:
            sections.append("\n".join(count_lines))

        summary = "\n\n".join(sections)

        if len(summary) > max_length:
            summary = summary[:max_length] + " ...（内容已截断）"

        return summary

    def _extract_counts(self, text: str) -> Dict[str, Optional[int]]:
        """从文档文本中提取各类条目数量"""
        counts: Dict[str, Optional[int]] = {}
        patterns = [
            ("命令", [r"(\d+)\s*个命令", r"命令\D*共\D*(\d+)", r"(\d+)\s*条命令"]),
            ("遥测", [r"(\d+)\s*个遥测", r"遥测\D*共\D*(\d+)", r"(\d+)\s*帧",
                      r"遥测帧.*?(\d+)"]),
            ("内部处理", [r"(\d+)\s*个内部", r"内部\D*共\D*(\d+)",
                         r"内部处理.*?(\d+)"]),
        ]
        for category, pats in patterns:
            for pat in pats:
                m = re.search(pat, text)
                if m:
                    counts[category] = int(m.group(1))
                    break
        return counts

    def _extract_keyword_snippet(
        self, text: str, keywords: list[str], window: int = 400
    ) -> str:
        """
        从文本中提取关键词周围的上下文片段

        Args:
            text: 完整文本
            keywords: 关键词列表
            window: 提取窗口大小

        Returns:
            提取的片段字符串
        """
        text_lower = text.lower()
        for kw in keywords:
            pos = text_lower.find(kw.lower())
            if pos >= 0:
                start = max(0, pos - 50)
                end = min(len(text), pos + window)
                return text[start:end].strip()
        return ""

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


def parse_target_protocol(file_path: str, target_name: Optional[str] = None) -> TargetProtocolSpec:
    """
    便捷函数：解析目标协议文档

    Args:
        file_path: 文档路径
        target_name: 目标协议名称

    Returns:
        TargetProtocolSpec 结构化规范对象

    Example:
        >>> spec = parse_target_protocol("data/target_proto/dllg/dllg.docx", "dllg")
        >>> print(spec.summary)
    """
    parser = TargetParser()
    return parser.parse(Path(file_path), target_name)
