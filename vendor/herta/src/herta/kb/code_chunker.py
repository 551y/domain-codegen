"""
代码感知分块策略

对代码文件（code_c, code_h）保持函数边界完整，不切断函数实现。
对非代码文件委托给 SimpleChunker。
"""

from typing import List

from herta.kb.schemas import SourceDocument, Chunk
from herta.kb.chunker import BaseChunker, ChunkConfig
from herta.kb.simple_chunker import SimpleChunker
from herta.logging import get_logger

logger = get_logger(__name__)


class CodeAwareChunker(BaseChunker):
    """
    代码感知分块策略

    对代码文件（kind=code_c, code_h, code_json）：
    - 函数级 SourceDocument（section=function_*）保持完整，不分块
    - 全文 SourceDocument（section=full_content）使用更大窗口分块

    对非代码文件：委托给 SimpleChunker
    """

    def __init__(self, config: ChunkConfig):
        super().__init__(config)
        # 为内部 SimpleChunker 创建 strategy=simple 的配置
        simple_config = ChunkConfig(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            strategy="simple",
        )
        self._simple_chunker = SimpleChunker(simple_config)

    def chunk(self, doc: SourceDocument) -> List[Chunk]:
        """
        分块：代码文件保持函数完整，其他文件委托给 SimpleChunker
        """
        if doc.kind in ("code_c", "code_h", "code_json"):
            return self._chunk_code(doc)
        return self._simple_chunker.chunk(doc)

    def _chunk_code(self, doc: SourceDocument) -> List[Chunk]:
        """
        处理代码文件的分块
        """
        text = doc.text
        if not text.strip():
            return []

        # 函数级 SourceDocument，保持完整
        if doc.section.startswith("function_"):
            return [Chunk(
                chunk_id=self._generate_chunk_id(doc.source, doc.kind),
                source=doc.source,
                file=doc.file,
                kind=doc.kind,
                section=doc.section,
                chunk_strategy="code_aware",
                chunk_index=0,
                text=text,
                token_count=self._estimate_tokens(text),
                metadata={"doc_metadata": doc.metadata},
            )]

        # JSON 配置文件，保持完整
        if doc.kind == "code_json":
            return [Chunk(
                chunk_id=self._generate_chunk_id(doc.source, doc.kind),
                source=doc.source,
                file=doc.file,
                kind=doc.kind,
                section=doc.section,
                chunk_strategy="code_aware",
                chunk_index=0,
                text=text,
                token_count=self._estimate_tokens(text),
                metadata={"doc_metadata": doc.metadata},
            )]

        # 全文 SourceDocument，使用更大窗口分块
        code_config = ChunkConfig(
            chunk_size=max(self.config.chunk_size, 1500),
            chunk_overlap=self.config.chunk_overlap,
            strategy="simple",
        )
        code_chunker = SimpleChunker(code_config)
        chunks = code_chunker.chunk(doc)
        for chunk in chunks:
            # 重新用外层计数器分配 id：内层 SimpleChunker 计数器从 0 开始，
            # 会与函数级 chunk 及其他文件的全文 chunk 撞号（导致检索串文本）
            chunk.chunk_id = self._generate_chunk_id(doc.source, doc.kind)
            chunk.chunk_strategy = "code_aware"
        return chunks
