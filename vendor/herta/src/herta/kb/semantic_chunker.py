"""
语义分块策略实现模块

基于相邻句子/段落的 embedding 相似度进行分块
当相似度低于阈值时切分，保持语义连贯性
"""

import re
import numpy as np
from typing import List

from herta.kb.schemas import SourceDocument, Chunk
from herta.kb.chunker import BaseChunker, ChunkConfig, ChunkReport
from herta.llm.client import BigModelClient
from herta.logging import get_logger

logger = get_logger(__name__)


class SemanticChunker(BaseChunker):
    """
    语义分块策略

    计算相邻句子/段落的 embedding 相似度，当相似度低于阈值时切分
    """

    def __init__(self, config: ChunkConfig):
        super().__init__(config)
        if config.strategy != "semantic":
            raise ValueError(f"SemanticChunker 需要 strategy='semantic', 得到 '{config.strategy}'")

        # 初始化 LLM 客户端用于生成 embeddings
        self.llm_client = BigModelClient()

        # 参数
        self.threshold_percentile = config.semantic_threshold_percentile
        self.min_sentences = config.semantic_min_sentences
        self.max_chars = config.semantic_max_chars

    def chunk(self, doc: SourceDocument) -> List[Chunk]:
        """
        基于语义相似度分块

        流程：
        1. 将文本切分为句子
        2. 批量生成句子 embeddings
        3. 计算相邻句子相似度
        4. 根据阈值确定切分点
        5. 组合成 chunks
        """
        text = doc.text
        if not text.strip():
            logger.warning(f"文档内容为空，跳过: {doc.file}")
            return []

        # 1. 切分为句子
        sentences = self._split_into_sentences(text)
        if len(sentences) < self.min_sentences:
            # 句子太少，直接作为一个 chunk
            return self._create_single_chunk(doc, text)

        logger.debug(f"文档切分为 {len(sentences)} 个句子")

        # 2. 生成句子 embeddings（分批处理，每批最多64条）
        embeddings = self._generate_embeddings_batch(sentences)

        # 3. 计算相邻句子相似度
        similarities = self._calculate_similarities(embeddings)

        # 4. 根据阈值确定切分点
        split_indices = self._find_split_points(similarities)

        # 5. 组合成 chunks
        chunks = self._create_chunks_from_splits(doc, sentences, split_indices)

        logger.debug(f"语义分块完成: {len(sentences)} 个句子 -> {len(chunks)} 个 chunks")
        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        将文本切分为句子

        支持中英文句子结束符
        """
        # 句子结束符：。！？. ! ? ；;
        sentence_endings = r'([。！？.!?；;])'

        # 先按结束符切分
        parts = re.split(sentence_endings, text)

        sentences = []
        current = ""

        for i, part in enumerate(parts):
            if not part:
                continue

            current += part

            # 如果当前部分以结束符结尾，或者到了最后一个部分
            if re.match(sentence_endings, part) or i == len(parts) - 1:
                stripped = current.strip()
                if stripped and len(stripped) > 5:  # 过滤太短的片段
                    sentences.append(stripped)
                current = ""

        # 处理剩余内容
        if current.strip() and len(current.strip()) > 5:
            sentences.append(current.strip())

        return sentences

    def _generate_embeddings_batch(self, sentences: List[str]) -> np.ndarray:
        """
        批量生成句子 embeddings

        每批最多64条（智谱API限制）
        """
        batch_size = 64
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(sentences) + batch_size - 1) // batch_size

            logger.debug(f"生成 embeddings 批次 {batch_num}/{total_batches}: {len(batch)} 条")

            try:
                embeddings = self.llm_client.embed_texts(batch)
                all_embeddings.extend(embeddings)
            except Exception as e:
                logger.error(f"生成 embeddings 失败 (批次 {batch_num}): {e}")
                raise

        return np.array(all_embeddings, dtype=np.float32)

    def _calculate_similarities(self, embeddings: np.ndarray) -> np.ndarray:
        """
        计算相邻句子之间的余弦相似度
        """
        # 归一化
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # 避免除零
        normalized = embeddings / norms

        # 计算相邻向量的点积（即余弦相似度）
        similarities = np.sum(normalized[:-1] * normalized[1:], axis=1)

        return similarities

    def _find_split_points(self, similarities: np.ndarray) -> List[int]:
        """
        根据相似度阈值确定切分点

        策略：
        1. 根据阈值百分位数确定动态阈值
        2. 低于阈值的相邻句子间作为切分点
        3. 同时考虑最大字符数限制
        """
        if len(similarities) == 0:
            return []

        # 计算动态阈值
        threshold = np.percentile(similarities, self.threshold_percentile * 100)
        logger.debug(f"相似度阈值 (percentile={self.threshold_percentile}): {threshold:.4f}")

        # 找出低于阈值的切分点
        split_indices = []
        for i, sim in enumerate(similarities):
            if sim < threshold:
                split_indices.append(i + 1)  # 在第 i+1 个句子前切分

        return split_indices

    def _create_chunks_from_splits(
        self,
        doc: SourceDocument,
        sentences: List[str],
        split_indices: List[int]
    ) -> List[Chunk]:
        """
        根据切分点组合成 chunks
        """
        chunks = []
        chunk_index = 0

        # 添加首尾边界
        boundaries = [0] + split_indices + [len(sentences)]

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]

            # 组合句子
            chunk_sentences = sentences[start:end]
            chunk_text = " ".join(chunk_sentences)

            # 检查最大字符数限制
            if len(chunk_text) > self.max_chars and len(chunk_sentences) > self.min_sentences:
                # 需要进一步切分
                sub_chunks = self._split_oversized_chunk(doc, chunk_sentences, chunk_index)
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)
            else:
                chunk = Chunk(
                    chunk_id=self._generate_chunk_id(doc.source, doc.kind),
                    source=doc.source,
                    file=doc.file,
                    kind=doc.kind,
                    section=f"{doc.section}_semantic_{chunk_index:04d}",
                    chunk_strategy="semantic",
                    chunk_index=chunk_index,
                    text=chunk_text,
                    token_count=self._estimate_tokens(chunk_text),
                    metadata={
                        "doc_metadata": doc.metadata,
                        "sentence_count": len(chunk_sentences),
                        "start_sentence": start,
                        "end_sentence": end,
                    }
                )
                chunks.append(chunk)
                chunk_index += 1

        return chunks

    def _split_oversized_chunk(
        self,
        doc: SourceDocument,
        sentences: List[str],
        start_index: int
    ) -> List[Chunk]:
        """
        将超过最大字符数的 chunk 进一步切分
        """
        chunks = []
        current_sentences = []
        current_length = 0
        chunk_index = start_index

        for sentence in sentences:
            sentence_length = len(sentence)

            # 如果添加这个句子会超出限制，且当前已有足够句子
            if (current_length + sentence_length > self.max_chars and
                len(current_sentences) >= self.min_sentences):
                # 创建 chunk
                chunk_text = " ".join(current_sentences)
                chunk = Chunk(
                    chunk_id=self._generate_chunk_id(doc.source, doc.kind),
                    source=doc.source,
                    file=doc.file,
                    kind=doc.kind,
                    section=f"{doc.section}_semantic_{chunk_index:04d}",
                    chunk_strategy="semantic",
                    chunk_index=chunk_index,
                    text=chunk_text,
                    token_count=self._estimate_tokens(chunk_text),
                    metadata={
                        "doc_metadata": doc.metadata,
                        "sentence_count": len(current_sentences),
                        "split_reason": "oversized",
                    }
                )
                chunks.append(chunk)
                chunk_index += 1

                # 重置
                current_sentences = [sentence]
                current_length = sentence_length
            else:
                current_sentences.append(sentence)
                current_length += sentence_length

        # 处理剩余
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunk = Chunk(
                chunk_id=self._generate_chunk_id(doc.source, doc.kind),
                source=doc.source,
                file=doc.file,
                kind=doc.kind,
                section=f"{doc.section}_semantic_{chunk_index:04d}",
                chunk_strategy="semantic",
                chunk_index=chunk_index,
                text=chunk_text,
                token_count=self._estimate_tokens(chunk_text),
                metadata={
                    "doc_metadata": doc.metadata,
                    "sentence_count": len(current_sentences),
                }
            )
            chunks.append(chunk)

        return chunks

    def _create_single_chunk(self, doc: SourceDocument, text: str) -> List[Chunk]:
        """创建单个 chunk"""
        return [Chunk(
            chunk_id=self._generate_chunk_id(doc.source, doc.kind),
            source=doc.source,
            file=doc.file,
            kind=doc.kind,
            section=f"{doc.section}_semantic_0000",
            chunk_strategy="semantic",
            chunk_index=0,
            text=text,
            token_count=self._estimate_tokens(text),
            metadata={
                "doc_metadata": doc.metadata,
                "sentence_count": 1,
                "note": "insufficient_sentences",
            }
        )]
