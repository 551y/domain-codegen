"""
简单分块策略实现模块

使用固定大小的滑动窗口进行分块
"""

from typing import List, Optional, Callable

from herta.kb.schemas import SourceDocument, Chunk
from herta.kb.chunker import BaseChunker, ChunkConfig
from herta.logging import get_logger

logger = get_logger(__name__)


class SimpleChunker(BaseChunker):
    """
    简单分块策略

    使用固定大小的滑动窗口进行分块，支持字符级别或 token 级别
    """

    def __init__(self, config: ChunkConfig):
        """
        初始化简单分块器

        Args:
            config: 分块配置，需要 strategy="simple"
        """
        super().__init__(config)
        if config.strategy != "simple":
            raise ValueError(f"SimpleChunker 需要 strategy='simple', 得到 '{config.strategy}'")

    def chunk(self, doc: SourceDocument) -> List[Chunk]:
        """
        将文档分块

        使用滑动窗口方式，按照固定大小切分文本

        Args:
            doc: 源文档对象

        Returns:
            分块后的 Chunk 列表
        """
        text = doc.text
        if not text.strip():
            logger.warning(f"文档内容为空，跳过: {doc.file}")
            return []

        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        # 确保重叠小于块大小
        if overlap >= chunk_size:
            overlap = chunk_size // 4
            logger.warning(f"重叠大小 ({self.config.chunk_overlap}) >= 块大小 ({chunk_size})，"
                         f"自动调整为 {overlap}")

        # 滑动窗口分块
        start = 0
        chunk_index = 0

        while start < len(text):
            # 计算当前块的结束位置
            end = start + chunk_size

            # 如果不是最后一个块，尝试在句子边界截断
            if end < len(text):
                # 寻找最近的句子结束符
                sentence_ends = ['。', '！', '？', '. ', '! ', '? ', '\n', '；', ';']
                best_end = end
                for i in range(end - 1, start, -1):
                    if i < len(text) and text[i] in sentence_ends:
                        best_end = i + 1
                        break
                end = best_end

            # 提取当前块文本
            chunk_text = text[start:end].strip()

            if chunk_text:  # 只保留非空块
                chunk = Chunk(
                    chunk_id=self._generate_chunk_id(doc.source, doc.kind),
                    source=doc.source,
                    file=doc.file,
                    kind=doc.kind,
                    section=doc.section,
                    chunk_strategy="simple",
                    chunk_index=chunk_index,
                    text=chunk_text,
                    token_count=self._estimate_tokens(chunk_text),
                    metadata={
                        "doc_metadata": doc.metadata,
                        "char_start": start,
                        "char_end": end,
                    }
                )
                chunks.append(chunk)
                chunk_index += 1

            # 移动到下一个窗口位置（考虑重叠）
            start = end - overlap if end < len(text) else len(text)

            # 防止无限循环
            if start <= 0 or (chunk_index > 0 and start >= len(text)):
                break

        logger.debug(f"文档分块完成: {doc.file} -> {len(chunks)} 个 chunks")
        return chunks

    def chunk_batch(
        self,
        docs: List[SourceDocument],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Chunk]:
        """
        批量分块多个文档

        Args:
            docs: 源文档列表
            progress_callback: 进度回调函数

        Returns:
            所有分块后的 Chunk 列表
        """
        return super().chunk_batch(docs, progress_callback)
