"""
分块策略基类模块

定义分块策略的抽象接口，支持四种分块策略：
- SimpleChunker: 固定大小分块
- SemanticChunker: 基于语义相似度的分块
- AutoSizeChunker: 自动选择最优分块大小
- PropositionChunker: 基于原子事实命题的分块
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path

from herta.kb.schemas import SourceDocument, Chunk
from herta.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ChunkConfig:
    """分块配置参数基类"""

    chunk_size: int = 500
    chunk_overlap: int = 50
    strategy: str = "simple"

    # Semantic chunking 参数
    semantic_threshold_percentile: float = 0.5
    semantic_min_sentences: int = 2
    semantic_max_chars: int = 2000

    # Auto-size 参数
    candidate_sizes: List[int] = field(default_factory=lambda: [300, 500, 800, 1000])
    candidate_overlaps: List[int] = field(default_factory=lambda: [50, 100])
    eval_questions: Optional[List[str]] = None
    selection_output: Optional[Path] = None

    # Proposition 参数
    proposition_model: str = "glm-5.1"
    max_propositions_per_chunk: int = 10
    proposition_batch_size: int = 5
    proposition_cache: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化）"""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "strategy": self.strategy,
            "semantic_threshold_percentile": self.semantic_threshold_percentile,
            "semantic_min_sentences": self.semantic_min_sentences,
            "semantic_max_chars": self.semantic_max_chars,
            "candidate_sizes": self.candidate_sizes,
            "candidate_overlaps": self.candidate_overlaps,
            "eval_questions": self.eval_questions,
            "selection_output": str(self.selection_output) if self.selection_output else None,
            "proposition_model": self.proposition_model,
            "max_propositions_per_chunk": self.max_propositions_per_chunk,
            "proposition_batch_size": self.proposition_batch_size,
            "proposition_cache": self.proposition_cache,
        }


@dataclass
class ChunkReport:
    """分块报告"""

    strategy: str
    total_chunks: int
    total_chars: int
    avg_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int
    params: Dict[str, Any]
    selection_details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "total_chunks": self.total_chunks,
            "total_chars": self.total_chars,
            "avg_chunk_size": self.avg_chunk_size,
            "min_chunk_size": self.min_chunk_size,
            "max_chunk_size": self.max_chunk_size,
            "params": self.params,
            "selection_details": self.selection_details,
        }


class BaseChunker(ABC):
    """
    分块策略抽象基类

    所有分块策略的基类，定义统一接口
    """

    def __init__(self, config: ChunkConfig):
        """
        初始化分块器

        Args:
            config: 分块配置参数
        """
        self.config = config
        self._chunk_counter = 0
        self._report: Optional[ChunkReport] = None
        logger.debug(f"初始化分块器: {config.strategy}")

    @abstractmethod
    def chunk(self, doc: SourceDocument) -> List[Chunk]:
        """
        将单个文档分块

        Args:
            doc: 源文档对象

        Returns:
            分块后的 Chunk 列表
        """
        pass

    def chunk_batch(
        self,
        docs: List[SourceDocument],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Chunk]:
        """
        批量分块多个文档

        Args:
            docs: 源文档列表
            progress_callback: 进度回调函数，接收 (current, total) 参数

        Returns:
            所有分块后的 Chunk 列表
        """
        all_chunks = []
        total_docs = len(docs)

        for i, doc in enumerate(docs):
            chunks = self.chunk(doc)
            all_chunks.extend(chunks)

            if progress_callback:
                progress_callback(i + 1, total_docs)

        # 生成报告
        self._report = self._generate_report(all_chunks)

        logger.info(f"批量分块完成: {total_docs} 个文档 -> {len(all_chunks)} 个 chunks")
        return all_chunks

    def _generate_chunk_id(self, source: str, kind: str) -> str:
        """生成全局唯一 chunk ID"""
        chunk_id = f"{source}_{kind}_{self._chunk_counter:06d}"
        self._chunk_counter += 1
        return chunk_id

    def _estimate_tokens(self, text: str) -> int:
        """
        估算文本的 token 数量

        简化版：按空白字符分词估算
        """
        words = len(text.split())
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        return int(words + chinese_chars * 0.5) + 1

    def reset_counter(self):
        """重置 chunk 计数器"""
        self._chunk_counter = 0

    def get_report(self) -> Optional[ChunkReport]:
        """获取分块报告"""
        return self._report

    def _generate_report(self, chunks: List[Chunk]) -> ChunkReport:
        """生成分块统计报告"""
        if not chunks:
            return ChunkReport(
                strategy=self.config.strategy,
                total_chunks=0,
                total_chars=0,
                avg_chunk_size=0,
                min_chunk_size=0,
                max_chunk_size=0,
                params=self.config.to_dict(),
            )

        sizes = [len(c.text) for c in chunks]
        total_chars = sum(sizes)

        return ChunkReport(
            strategy=self.config.strategy,
            total_chunks=len(chunks),
            total_chars=total_chars,
            avg_chunk_size=total_chars / len(chunks),
            min_chunk_size=min(sizes),
            max_chunk_size=max(sizes),
            params=self.config.to_dict(),
        )


class DirectChunker(BaseChunker):
    """
    直接注入策略：每个文件保持完整，不做任何分块

    适用于源文件总量较小（< 200KB）的场景，
    把所有参考文件完整注入 Prompt，最大化信息量。
    """

    def chunk(self, doc: SourceDocument) -> List[Chunk]:
        """每个文档保持完整，只生成一个 chunk"""
        text = doc.text.strip()
        if not text:
            return []

        # 粗略估算 token 数（中文约 1.5 字符/token，英文约 4 字符/token）
        token_count = len(text) // 2

        chunk = Chunk(
            chunk_id=self._next_chunk_id(),
            source=doc.source,
            file=doc.file,
            kind=doc.kind,
            section="full_content",
            chunk_strategy="direct",
            chunk_index=0,
            text=text,
            token_count=token_count,
            metadata={
                "strategy": "direct",
                "char_count": len(text),
            },
        )
        return [chunk]

    def _next_chunk_id(self) -> str:
        self._chunk_counter += 1
        return f"direct_{self._chunk_counter:04d}"


def create_chunker(
    strategy: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    **kwargs
) -> BaseChunker:
    """
    工厂函数：创建分块器实例

    Args:
        strategy: 分块策略名称 (simple, semantic, autosize, proposition)
        chunk_size: 分块大小
        chunk_overlap: 分块重叠大小
        **kwargs: 其他策略特定参数

    Returns:
        分块器实例

    Raises:
        ValueError: 未知的分块策略
    """
    config = ChunkConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=strategy,
    )

    # 设置其他参数
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    if strategy == "simple":
        from herta.kb.simple_chunker import SimpleChunker
        return SimpleChunker(config)
    elif strategy == "semantic":
        from herta.kb.semantic_chunker import SemanticChunker
        return SemanticChunker(config)
    elif strategy == "autosize":
        from herta.kb.selector import AutoSizeChunker
        return AutoSizeChunker(config)
    elif strategy == "proposition":
        from herta.kb.proposition_chunker import PropositionChunker
        return PropositionChunker(config)
    elif strategy == "code_aware":
        from herta.kb.code_chunker import CodeAwareChunker
        return CodeAwareChunker(config)
    elif strategy == "direct":
        return DirectChunker(config)
    else:
        raise ValueError(f"未知的分块策略: {strategy}。支持: simple, semantic, autosize, proposition, code_aware, direct")
