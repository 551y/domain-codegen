"""
生成器数据结构定义模块

定义配置生成相关的数据结构：
- TargetProtocolSpec: 目标协议规范
- RetrievedChunk: 检索到的知识库片段
- GenerationContext: 生成上下文
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
import json


@dataclass
class TargetProtocolSpec:
    """
    目标协议规范

    从目标协议文档解析出的结构化信息
    """

    target_name: str
    document_path: str
    raw_text: str
    summary: str
    sections: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TargetProtocolSpec":
        """从字典创建"""
        return cls(**data)


@dataclass
class RetrievedChunk:
    """
    检索到的知识库片段

    包含从源协议知识库检索到的相关信息
    """

    chunk_id: str
    source: str
    file: str
    kind: str
    section: str
    text: str
    score: float  # 相似度分数
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievedChunk":
        """从字典创建"""
        return cls(**data)

    def format_for_prompt(self) -> str:
        """
        格式化为 Prompt 中的引用文本

        Returns:
            格式化的字符串，包含来源和片段内容
        """
        return f"""【来源: {self.source} | 文件: {self.file} | 章节: {self.section} | 相关度: {self.score:.3f}】
{self.text}
---"""


@dataclass
class GenerationContext:
    """
    配置生成上下文

    整合目标协议信息、检索到的参考信息、生成参数等
    """

    target: TargetProtocolSpec
    retrieved_chunks: List[RetrievedChunk]
    source_kb: str
    top_k: int
    model: str
    temperature: float
    output_schema: str = """{
  "version": "版本号，字符串，如 1.0",
  "author": "作者标识，字符串",
  "start_index": "起始命令编号，整数",
  "cmd_list": [
    {
      "desc": "命令描述，字符串，格式为 {target}_功能名称",
      "cmd": "命令编号，整数，从 start_index 开始递增",
      "handler": "处理函数名，字符串，格式为 {target}_功能名称"
    }
  ],
  "tele_handler": [
    {
      "desc": "遥测帧描述，字符串，格式为 {target}_tele_frame_XX",
      "len": "帧数据长度，整数（字节数）",
      "handler": "处理函数名，字符串，格式为 {target}_tele_frame_XX"
    }
  ],
  "internal_handler": [
    {
      "desc": "内部处理项描述，字符串",
      "len": "数据长度，整数（字节数）",
      "handler": "处理函数名，字符串"
    }
  ]
}"""

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "target": self.target.to_dict(),
            "retrieved_chunks": [c.to_dict() for c in self.retrieved_chunks],
            "source_kb": self.source_kb,
            "top_k": self.top_k,
            "model": self.model,
            "temperature": self.temperature,
            "output_schema": self.output_schema,
            "metadata": self.metadata,
        }

    def format_retrieved_context(self, max_length: Optional[int] = None) -> str:
        """
        格式化检索到的上下文，用于 Prompt

        按结构优先级排序：完整 JSON 模板 > JSON 字段 > C 代码 > 文档段落，
        尽可能保留全部检索结果不做截断。
        """
        if not self.retrieved_chunks:
            return "（未检索到相关参考信息）"

        kind_priority = {"code_json": 0, "code_c": 1, "code_h": 1, "doc": 2}

        def sort_key(chunk):
            kind_pri = kind_priority.get(chunk.kind, 3)
            section_pri = 0 if chunk.section == "full_content" else 1
            return (kind_pri, section_pri, -chunk.score)

        sorted_chunks = sorted(self.retrieved_chunks, key=sort_key)

        sections = []
        for i, chunk in enumerate(sorted_chunks, 1):
            sections.append(f"【参考片段 {i}】\n{chunk.format_for_prompt()}")

        context = "\n\n".join(sections)

        if max_length and len(context) > max_length:
            context = context[:max_length] + "\n...（上下文已截断）"

        return context

    def format_target_summary(self) -> str:
        """
        格式化目标协议摘要，用于 Prompt

        传入完整文档原文和所有章节结构，让 LLM 获得最大信息量。
        """
        sections_info = ""
        if self.target.sections:
            shown = self.target.sections[:20]
            sections_info = "\n文档结构:\n" + "\n".join(
                f"  - {s.get('title', s.get('name', '未命名章节'))}"
                for s in shown
            )
            if len(self.target.sections) > 20:
                sections_info += f"\n  ... 等共 {len(self.target.sections)} 个章节"

        # 完整原文（不截断）
        full_doc = self.target.raw_text

        return f"""【目标协议】
名称: {self.target.target_name}
文档路径: {self.target.document_path}
结构化摘要: {self.target.summary}{sections_info}

【目标协议完整原文】
{full_doc}
"""


@dataclass
class GenerationResult:
    """
    生成结果

    包含生成的配置内容和元数据
    """

    config: Dict[str, Any]
    raw_response: str
    target: str
    source_kb: str
    retrieved_chunk_count: int
    model: str
    temperature: float
    output_path: Optional[str] = None
    generation_time: Optional[str] = None
    is_valid_json: bool = True
    error_message: Optional[str] = None

    def __post_init__(self):
        """初始化后处理"""
        if self.generation_time is None:
            from datetime import datetime
            self.generation_time = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save_config(self, output_path: Path) -> None:
        """
        保存配置到文件

        Args:
            output_path: 输出文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

        self.output_path = str(output_path)
        logger.info(f"配置已保存: {output_path}")


from herta.logging import get_logger
logger = get_logger(__name__)
