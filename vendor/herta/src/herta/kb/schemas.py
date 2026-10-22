"""
知识库数据结构定义模块

定义统一的数据结构：
- SourceDocument: 源文档对象
- Chunk: 分块后的数据单元
- KBMeta: 知识库元数据
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
import json


@dataclass
class SourceDocument:
    """
    统一源文档对象

    表示从任何来源（文档、代码）解析出的内容单元

    Attributes:
        source: 来源标识（如 gdwx, dllg）
        file: 原始文件路径
        kind: 文档类型（doc, code_json, code_c 等）
        section: 章节/段落标识
        text: 文档内容文本
        metadata: 额外元数据
    """

    source: str
    file: str
    kind: str  # doc, code_json, code_c, code_h
    section: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceDocument":
        """从字典创建"""
        return cls(**data)


@dataclass
class Chunk:
    """
    分块后的数据单元

    Attributes:
        chunk_id: 全局唯一分块 ID（格式：{source}_{kind}_{index}）
        source: 来源标识
        file: 原始文件路径
        kind: 文档类型
        section: 章节/段落标识
        chunk_strategy: 分块策略名称
        chunk_index: 分块在文档中的序号
        text: 分块文本内容
        token_count: 估算的 token 数量
        metadata: 额外元数据
    """

    chunk_id: str
    source: str
    file: str
    kind: str
    section: str
    chunk_strategy: str
    chunk_index: int
    text: str
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """从字典创建"""
        return cls(**data)

    def to_jsonl(self) -> str:
        """转换为 JSONL 格式的一行"""
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class KBMeta:
    """
    知识库元数据

    记录知识库的构建参数和统计信息
    """

    source: str
    input_files: List[str]
    chunk_strategy: str
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    embedding_dimensions: int
    chunk_count: int
    build_time: Optional[str] = None
    version: str = "1.0"

    def __post_init__(self):
        """初始化后处理"""
        if self.build_time is None:
            from datetime import datetime
            self.build_time = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KBMeta":
        """从字典创建"""
        return cls(**data)

    def to_json(self, indent: int = 2) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "KBMeta":
        """从 JSON 字符串创建"""
        return cls.from_dict(json.loads(json_str))
