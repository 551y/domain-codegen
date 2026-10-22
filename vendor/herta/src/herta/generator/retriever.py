"""
知识库检索模块

负责从知识库中检索相关信息：
- 加载索引和元数据
- 执行向量检索
- 返回结构化的检索结果
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np

from herta.generator.schemas import RetrievedChunk
from herta.kb.store import VectorStore
from herta.kb.schemas import Chunk
from herta.llm.client import BigModelClient
from herta.logging import get_logger

logger = get_logger(__name__)


def _char_bigrams(text: str) -> set[str]:
    """生成字符 bigram 集合，用于中文友好的文本重叠度计算"""
    t = text.lower()
    if len(t) < 2:
        return {t} if t else set()
    return {t[i:i+2] for i in range(len(t) - 1)}


class KnowledgeRetriever:
    """
    知识库检索器

    从源协议知识库中检索与查询相关的信息片段
    """

    def __init__(self, kb_dir: Path, llm_client: Optional[BigModelClient] = None):
        """
        初始化知识库检索器

        Args:
            kb_dir: 知识库目录路径
            llm_client: LLM 客户端实例，用于生成 query embedding
        """
        self.kb_dir = Path(kb_dir)
        self.llm_client = llm_client or BigModelClient()

        # 加载知识库组件
        self.vector_store: Optional[VectorStore] = None
        self.chunks: Dict[str, Chunk] = {}  # chunk_id -> Chunk
        self.meta: Optional[Dict[str, Any]] = None

        self._loaded = False

    def load(self) -> None:
        """
        加载知识库文件

        加载：
        - vector.index (FAISS 索引)
        - chunks.jsonl (Chunk 数据)
        - meta.json (元数据)
        """
        if self._loaded:
            return

        logger.info(f"加载知识库: {self.kb_dir}")

        # 1. 加载元数据
        meta_path = self.kb_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path, 'r', encoding='utf-8') as f:
                self.meta = json.load(f)
            logger.debug(f"加载元数据: source={self.meta.get('source')}, chunks={self.meta.get('chunk_count')}")
        else:
            logger.warning(f"元数据文件不存在: {meta_path}")

        # 2. 加载 chunks
        chunks_path = self.kb_dir / "chunks.jsonl"
        if chunks_path.exists():
            self._load_chunks(chunks_path)
        else:
            raise FileNotFoundError(f"chunks 文件不存在: {chunks_path}")

        # 3. 加载向量索引
        index_path = self.kb_dir / "vector.index"
        if index_path.exists():
            self.vector_store = VectorStore(
                dimension=self.meta.get("embedding_dimensions", 1024) if self.meta else 1024,
                metric="ip",  # 使用内积（余弦相似度）
            )
            self.vector_store.load(self.kb_dir)
            logger.info(f"加载向量索引: {self.vector_store.vector_count} 个向量")
        else:
            raise FileNotFoundError(f"向量索引不存在: {index_path}")

        self._loaded = True
        logger.info(f"知识库加载完成: {len(self.chunks)} 个 chunks")

    def _load_chunks(self, chunks_path: Path) -> None:
        """
        加载 chunks.jsonl 文件

        Args:
            chunks_path: chunks.jsonl 文件路径
        """
        logger.debug(f"加载 chunks: {chunks_path}")

        count = 0
        with open(chunks_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    chunk = Chunk.from_dict(data)
                    self.chunks[chunk.chunk_id] = chunk
                    count += 1
                except json.JSONDecodeError as e:
                    logger.warning(f"解析 chunk 失败: {e}, line={line[:100]}...")
                except Exception as e:
                    logger.warning(f"加载 chunk 失败: {e}")

        logger.debug(f"加载了 {count} 个 chunks")

    def retrieve_all(self) -> List[RetrievedChunk]:
        """
        直接返回所有 chunk，不做向量检索。

        用于 direct 策略：跳过检索，把所有参考文件完整注入。
        """
        if not self._loaded:
            self.load()

        results = []
        for chunk in self.chunks.values():
            results.append(RetrievedChunk(
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                file=chunk.file,
                kind=chunk.kind,
                section=chunk.section,
                text=chunk.text,
                score=1.0,
                token_count=chunk.token_count,
                metadata=chunk.metadata,
            ))

        # 按 kind 优先级排序：JSON > C/H > doc
        kind_priority = {"code_json": 0, "code_c": 1, "code_h": 1, "doc": 2}
        results.sort(key=lambda c: (kind_priority.get(c.kind, 3), c.chunk_id))

        logger.info(f"Direct 模式: 直接注入全部 {len(results)} 个 chunk")
        return results

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        """
        检索相关知识库片段

        流程：
        1. 使用 LLM 生成 query embedding
        2. 在 FAISS 中检索 top_k 相似向量
        3. 根据 chunk_id 获取完整 chunk 信息
        4. 返回结构化的检索结果

        Args:
            query: 查询文本
            top_k: 返回的最相似结果数量
            min_score: 最小相似度阈值（可选）

        Returns:
            RetrievedChunk 列表，按相似度降序排列

        Raises:
            RuntimeError: 知识库未加载时
        """
        if not self._loaded:
            self.load()

        logger.info(f"执行检索: query='{query[:50]}...', top_k={top_k}")

        # 1. 生成 query embedding
        try:
            query_vector = self.llm_client.embed_query(query)
            logger.debug(f"Query embedding 生成完成: dimension={len(query_vector)}")
        except Exception as e:
            logger.error(f"生成 query embedding 失败: {e}")
            raise

        # 2. 向量检索
        try:
            indices, scores = self.vector_store.search(
                query_vector=np.array(query_vector),
                top_k=top_k,
            )
            logger.debug(f"向量检索完成: 返回 {len(indices)} 个结果")
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            raise

        if not indices:
            logger.warning("检索未返回任何结果")
            return []

        # 3. 组装检索结果
        results = []
        for idx, score in zip(indices, scores):
            chunk_id = self.vector_store.get_chunk_id_by_index(idx)
            if chunk_id is None:
                logger.warning(f"无法获取 chunk_id for index={idx}")
                continue

            chunk = self.chunks.get(chunk_id)
            if chunk is None:
                logger.warning(f"无法找到 chunk: {chunk_id}")
                continue

            # 应用分数阈值
            if min_score is not None and score < min_score:
                continue

            retrieved = RetrievedChunk(
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                file=chunk.file,
                kind=chunk.kind,
                section=chunk.section,
                text=chunk.text,
                score=float(score),
                token_count=chunk.token_count,
                metadata=chunk.metadata,
            )
            results.append(retrieved)

        # 按相似度降序排列（FAISS IP 索引返回的分数已经是降序）
        logger.info(f"检索完成: 返回 {len(results)} 个 chunks")
        for i, r in enumerate(results[:3], 1):
            logger.debug(f"  Top {i}: {r.chunk_id} (score={r.score:.3f}, source={r.source})")

        return results

    def get_source_info(self) -> Dict[str, Any]:
        """
        获取知识库源信息

        Returns:
            知识库元数据字典
        """
        if not self._loaded:
            self.load()

        return {
            "source": self.meta.get("source") if self.meta else None,
            "chunk_count": len(self.chunks),
            "vector_count": self.vector_store.vector_count if self.vector_store else 0,
            "embedding_model": self.meta.get("embedding_model") if self.meta else None,
            "chunk_strategy": self.meta.get("chunk_strategy") if self.meta else None,
        }

    def retrieve_multi(
        self,
        queries: List[str],
        top_k_per_query: int = 3,
        max_total: int = 10,
        deduplicate: bool = True,
    ) -> List[RetrievedChunk]:
        """
        执行多个查询并合并结果

        对每个 query 分别检索（带重排序），然后去重合并，按分数降序返回。

        Args:
            queries: 查询文本列表
            top_k_per_query: 每个 query 返回的结果数
            max_total: 合并后最大返回数
            deduplicate: 是否按 chunk_id 去重

        Returns:
            合并后的 RetrievedChunk 列表，按分数降序
        """
        if not self._loaded:
            self.load()

        all_results: Dict[str, RetrievedChunk] = {}

        for i, query in enumerate(queries):
            logger.info(f"Multi-retrieval query {i+1}/{len(queries)}: '{query[:50]}...'")
            try:
                results = self.retrieve_with_rerank(query, top_k=top_k_per_query)
                for chunk in results:
                    if deduplicate and chunk.chunk_id in all_results:
                        existing = all_results[chunk.chunk_id]
                        if chunk.score > existing.score:
                            all_results[chunk.chunk_id] = chunk
                    else:
                        all_results[chunk.chunk_id] = chunk
            except Exception as e:
                logger.warning(f"Multi-retrieval query {i+1} failed: {e}")
                continue

        sorted_results = sorted(
            all_results.values(), key=lambda c: c.score, reverse=True
        )
        logger.info(f"Multi-retrieval 完成: {len(sorted_results)} 个去重结果, "
                    f"返回 top {max_total}")
        return sorted_results[:max_total]

    def retrieve_with_rerank(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.2,
    ) -> List[RetrievedChunk]:
        """
        检索并重排序

        1. 向量检索 3x 候选
        2. 用关键词重叠度重排序（混合向量分数 70% + 关键词重叠 30%）
        3. 按 min_score 过滤低质量结果
        4. 返回 top_k

        Args:
            query: 查询文本
            top_k: 最终返回数量
            min_score: 最低分数阈值

        Returns:
            重排序后的 RetrievedChunk 列表
        """
        if not self._loaded:
            self.load()

        # 过度检索候选
        candidate_count = min(top_k * 3, self.vector_store.vector_count if self.vector_store else top_k)
        candidates = self.retrieve(query, top_k=candidate_count)

        if not candidates:
            return []

        # 字符 bigram 重排序（支持中文）
        query_bg = _char_bigrams(query)
        for chunk in candidates:
            chunk_bg = _char_bigrams(chunk.text[:500])
            overlap = len(query_bg & chunk_bg)
            overlap_ratio = overlap / max(len(query_bg), 1)
            # 混合分数：70% 向量 + 30% 关键词重叠
            chunk.score = 0.7 * chunk.score + 0.3 * overlap_ratio

        # 按混合分数排序
        candidates.sort(key=lambda c: c.score, reverse=True)

        # 过滤低分结果
        filtered = [c for c in candidates if c.score >= min_score]

        logger.debug(f"Rerank: {len(candidates)} 候选 -> {len(filtered)} 过滤后 -> top {top_k}")
        return filtered[:top_k]


def retrieve_from_kb(
    kb_dir: str,
    query: str,
    top_k: int = 5,
) -> List[RetrievedChunk]:
    """
    便捷函数：从知识库检索

    Args:
        kb_dir: 知识库目录路径
        query: 查询文本
        top_k: 返回结果数量

    Returns:
        RetrievedChunk 列表

    Example:
        >>> results = retrieve_from_kb("data/kb/gdwx", "协议配置字段")
        >>> for r in results:
        ...     print(f"{r.chunk_id}: {r.text[:100]}...")
    """
    retriever = KnowledgeRetriever(Path(kb_dir))
    return retriever.retrieve(query, top_k=top_k)
