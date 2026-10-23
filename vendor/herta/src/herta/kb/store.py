"""
向量存储模块

封装 FAISS 索引操作，提供：
- save_index(): 保存索引到磁盘
- load_index(): 从磁盘加载索引
- search(): 向量相似度检索
"""

import json
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

from herta.logging import get_logger

logger = get_logger(__name__)

# 尝试导入 FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS 未安装/不可用，向量检索将回退为 numpy 内积扫描（仅支持 load/search）")


class VectorStore:
    """
    向量存储类

    封装 FAISS 索引的创建、保存、加载和检索操作。
    FAISS 不可用（如 Windows 上 pip 版 faiss-cpu 原生 DLL 加载失败）时，
    load/search 自动回退为 numpy 实现：直接加载随库交付的 embeddings.npy
    + chunk_ids.json，归一化后做内积扫描（与 IndexFlatIP 余弦检索等价），
    保证运行态检索不依赖原生扩展；构建态（create/add/save）仍需 FAISS。
    """

    def __init__(self, dimension: int, metric: str = "ip"):
        """
        初始化向量存储

        Args:
            dimension: 向量维度
            metric: 距离度量方式，"ip" (内积/余弦相似度) 或 "l2" (欧氏距离)
        """
        self.dimension = dimension
        self.metric = metric
        self.index = None
        self.chunk_ids: List[str] = []  # 存储 chunk_id 列表，与索引位置对应
        # numpy 回退后端的向量矩阵 (n, dimension)，仅 FAISS 不可用时由 load() 填充
        self._vectors: Optional[np.ndarray] = None

        logger.debug(f"初始化 VectorStore: dimension={dimension}, metric={metric}, "
                     f"backend={'faiss' if FAISS_AVAILABLE else 'numpy'}")

    def create_index(self, embeddings: Optional[np.ndarray] = None) -> None:
        """
        创建 FAISS 索引

        Args:
            embeddings: 可选的初始向量数组 (n, dimension)
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS 未安装")

        if self.metric == "ip":
            # 内积索引（用于余弦相似度，需要先归一化向量）
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.metric == "l2":
            # L2 距离索引
            self.index = faiss.IndexFlatL2(self.dimension)
        else:
            raise ValueError(f"不支持的距离度量: {self.metric}，支持 'ip' 或 'l2'")

        logger.debug(f"创建 FAISS 索引: {self.metric}, dimension={self.dimension}")

        # 如果有初始向量，添加到索引
        if embeddings is not None and len(embeddings) > 0:
            self.add_vectors(embeddings)

    def add_vectors(self, embeddings: np.ndarray, chunk_ids: Optional[List[str]] = None) -> None:
        """
        添加向量到索引

        Args:
            embeddings: 向量数组 (n, dimension)，必须是 float32 类型
            chunk_ids: 对应的 chunk_id 列表
        """
        if self.index is None:
            self.create_index()

        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype=np.float32)

        # 确保是 float32 类型
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        # 确保是二维数组
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # 归一化向量（如果使用 IP 索引进行余弦相似度计算）
        if self.metric == "ip":
            faiss.normalize_L2(embeddings)

        # 添加到索引
        self.index.add(embeddings)

        # 记录 chunk_ids
        if chunk_ids:
            self.chunk_ids.extend(chunk_ids)

        logger.debug(f"添加 {len(embeddings)} 个向量到索引，当前总数: {self.index.ntotal}")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> Tuple[List[int], List[float]]:
        """
        检索相似向量

        Args:
            query_vector: 查询向量，可以是 1D 或 2D 数组
            top_k: 返回的最相似结果数量

        Returns:
            (indices, scores) 元组：
            - indices: 相似向量的索引列表
            - scores: 相似度分数列表（越大越相似；L2 在 numpy 回退下以负距离表示）
        """
        # 准备查询向量
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)

        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)

        # 确保是 2D 数组 (1, dimension)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # numpy 回退后端：内积扫描（IP 下与 FAISS 余弦检索等价）
        if self.index is None:
            if self._vectors is None or len(self._vectors) == 0:
                logger.warning("索引为空，无法检索")
                return [], []
            q = query_vector[0]
            if q.shape[0] != self._vectors.shape[1]:
                logger.warning(f"查询向量维度 {q.shape[0]} 与索引维度 {self._vectors.shape[1]} 不一致")
                return [], []
            if self.metric == "ip":
                norm = float(np.linalg.norm(q))
                if norm > 0:
                    q = q / norm
                scores = self._vectors @ q
            else:  # l2：负距离，越大越相似，与返回值的降序语义一致
                scores = -np.sum((self._vectors - q) ** 2, axis=1)
            k = min(top_k, len(scores))
            # 先取前 k 候选再按分数降序排序
            part = np.argpartition(-scores, k - 1)[:k]
            order = part[np.argsort(-scores[part])]
            return order.tolist(), [float(scores[i]) for i in order]

        if self.index.ntotal == 0:
            logger.warning("索引为空，无法检索")
            return [], []

        # 归一化查询向量
        if self.metric == "ip":
            faiss.normalize_L2(query_vector)

        # 检索
        scores, indices = self.index.search(query_vector, top_k)

        # 转换为 Python 列表
        indices = indices[0].tolist()
        scores = scores[0].tolist()

        # 过滤掉 -1（表示未找到足够结果）
        results = [(idx, score) for idx, score in zip(indices, scores) if idx != -1]

        if results:
            indices, scores = zip(*results)
            return list(indices), list(scores)
        else:
            return [], []

    def save(
        self,
        output_dir: Path,
        embeddings: Optional[np.ndarray] = None,
    ) -> None:
        """
        保存索引到磁盘

        Args:
            output_dir: 输出目录
            embeddings: 原始向量数组（可选，用于保存 embeddings.npy）
        """
        import tempfile
        import shutil

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存 FAISS 索引（先写到临时文件再复制，解决中文路径问题）
        if self.index is not None:
            index_path = output_dir / "vector.index"
            with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                faiss.write_index(self.index, tmp_path)
                shutil.copy2(tmp_path, str(index_path))
            finally:
                import os
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            logger.debug(f"保存索引: {index_path}")

        # 保存 chunk_ids 映射
        chunk_ids_path = output_dir / "chunk_ids.json"
        with open(chunk_ids_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunk_ids, f, indent=2, ensure_ascii=False)
        logger.debug(f"保存 chunk_ids: {chunk_ids_path}")

        # 保存 embeddings（如果提供）
        if embeddings is not None:
            embeddings_path = output_dir / "embeddings.npy"
            np.save(embeddings_path, embeddings)
            logger.debug(f"保存 embeddings: {embeddings_path}")

        logger.info(f"向量存储已保存到: {output_dir}")

    def load(self, output_dir: Path) -> None:
        """
        从磁盘加载索引

        FAISS 可用时加载 vector.index；不可用时回退为 numpy 后端——
        直接加载随知识库交付的 embeddings.npy（与索引同批向量）做内积扫描。

        Args:
            output_dir: 索引所在目录
        """
        output_dir = Path(output_dir)

        if not FAISS_AVAILABLE:
            embeddings_path = output_dir / "embeddings.npy"
            if not embeddings_path.exists():
                raise FileNotFoundError(
                    f"FAISS 不可用且未找到回退向量文件: {embeddings_path}"
                    "（知识库需随附 embeddings.npy 才能脱离 FAISS 检索）"
                )
            vectors = np.load(embeddings_path).astype(np.float32)
            if vectors.ndim != 2:
                raise ValueError(f"embeddings.npy 形状异常: {vectors.shape}")
            # 防御性归一化（交付的向量一般已 L2 归一化，归一化幂等）
            if self.metric == "ip":
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                vectors = vectors / norms
            self._vectors = vectors
            self.dimension = vectors.shape[1]

            # 加载 chunk_ids
            chunk_ids_path = output_dir / "chunk_ids.json"
            if chunk_ids_path.exists():
                with open(chunk_ids_path, 'r', encoding='utf-8') as f:
                    self.chunk_ids = json.load(f)
            else:
                raise FileNotFoundError(f"chunk_ids 文件不存在: {chunk_ids_path}")
            if len(self.chunk_ids) != len(vectors):
                raise ValueError(
                    f"chunk_ids 数量({len(self.chunk_ids)})与向量数量({len(vectors)})不一致"
                )
            logger.info(f"向量存储已加载（numpy 回退后端）: {output_dir}, vectors={len(vectors)}")
            return

        import tempfile
        import shutil

        # 加载 FAISS 索引（先复制到临时文件解决中文路径问题）
        index_path = output_dir / "vector.index"
        if index_path.exists():
            with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                shutil.copy2(str(index_path), tmp_path)
                self.index = faiss.read_index(tmp_path)
            finally:
                import os
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            self.dimension = self.index.d
            logger.debug(f"加载索引: {index_path}, vectors={self.index.ntotal}")
        else:
            raise FileNotFoundError(f"索引文件不存在: {index_path}")

        # 加载 chunk_ids
        chunk_ids_path = output_dir / "chunk_ids.json"
        if chunk_ids_path.exists():
            with open(chunk_ids_path, 'r', encoding='utf-8') as f:
                self.chunk_ids = json.load(f)
            logger.debug(f"加载 chunk_ids: {len(self.chunk_ids)} 个")
        else:
            logger.warning(f"chunk_ids 文件不存在: {chunk_ids_path}")
            self.chunk_ids = []

        logger.info(f"向量存储已加载: {output_dir}")

    def get_chunk_id_by_index(self, index: int) -> Optional[str]:
        """
        通过索引位置获取 chunk_id

        Args:
            index: 索引位置

        Returns:
            chunk_id 或 None
        """
        if 0 <= index < len(self.chunk_ids):
            return self.chunk_ids[index]
        return None

    @property
    def vector_count(self) -> int:
        """返回索引中的向量数量"""
        if self.index is not None:
            return self.index.ntotal
        if self._vectors is not None:
            return len(self._vectors)
        return 0
