"""
知识库构建器模块

负责完整的知识库构建流程：
1. 接收文档列表
2. 解析文档（doc_parser + code_parser）
3. 分块处理
4. 生成 embeddings
5. 建立 FAISS 索引
6. 保存到输出目录
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import numpy as np

from herta.kb.schemas import SourceDocument, Chunk, KBMeta
from herta.kb.chunker import create_chunker, BaseChunker, ChunkConfig
from herta.kb.store import VectorStore
from herta.parser.doc_parser import DocParser
from herta.parser.code_parser import CodeParser
from herta.llm.client import BigModelClient
from herta.config import get_settings
from herta.logging import get_logger

logger = get_logger(__name__)


class KnowledgeBaseBuilder:
    """
    知识库构建器

    完整的知识库构建流程管理器
    """

    def __init__(
        self,
        source: str,
        output_dir: Path,
        chunk_config: Optional[ChunkConfig] = None,
        embedding_model: Optional[str] = None,
    ):
        """
        初始化知识库构建器

        Args:
            source: 来源标识（如 gdwx, dllg）
            output_dir: 输出目录
            chunk_config: 分块配置对象，默认使用 simple 策略
            embedding_model: 指定 embedding 模型，默认使用配置中的模型
        """
        self.source = source
        self.output_dir = Path(output_dir)

        # 分块配置
        self.chunk_config = chunk_config or ChunkConfig()
        self.chunk_strategy = self.chunk_config.strategy

        # 获取配置
        settings = get_settings()
        self.embedding_model = embedding_model or settings.embedding_model
        self.embedding_dimensions = settings.embedding_dimensions

        # 初始化组件
        self.doc_parser = DocParser()
        self.code_parser = CodeParser()
        self.chunker: Optional[BaseChunker] = None
        self.llm_client: Optional[BigModelClient] = None
        self.vector_store: Optional[VectorStore] = None

        # 统计数据
        self.stats = {
            "input_files": [],
            "parsed_docs": 0,
            "chunks": 0,
            "embeddings": 0,
        }

        logger.info(f"初始化 KnowledgeBaseBuilder: source={source}, strategy={self.chunk_strategy}")

    def build(
        self,
        doc_files: Optional[List[Path]] = None,
        code_files: Optional[List[Path]] = None,
    ) -> KBMeta:
        """
        构建知识库

        完整流程：解析 -> 分块 -> Embedding -> 索引 -> 保存

        Args:
            doc_files: 文档文件列表（docx/pdf）
            code_files: 代码文件列表（json/c/h）

        Returns:
            KBMeta 知识库元数据
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info(f"开始构建知识库: {self.source}")
        logger.info("=" * 60)

        # 1. 收集输入文件
        all_input_files = []
        if doc_files:
            all_input_files.extend([str(f) for f in doc_files])
        if code_files:
            all_input_files.extend([str(f) for f in code_files])
        self.stats["input_files"] = all_input_files

        logger.info(f"输入文件: {len(all_input_files)} 个")

        # 2. 解析文档
        source_docs = self._parse_all(doc_files or [], code_files or [])
        self.stats["parsed_docs"] = len(source_docs)
        logger.info(f"解析完成: {len(source_docs)} 个文档对象")

        if not source_docs:
            raise ValueError("没有可处理的文档，请检查输入文件")

        # 3. 分块处理
        chunks = self._chunk_all(source_docs)
        self.stats["chunks"] = len(chunks)
        logger.info(f"分块完成: {len(chunks)} 个 chunks")

        if not chunks:
            raise ValueError("分块后没有产生任何 chunk，请检查文档内容")

        # 4. 生成 Embeddings
        embeddings = self._generate_embeddings(chunks)
        self.stats["embeddings"] = len(embeddings)
        logger.info(f"Embedding 完成: {len(embeddings)} 个向量")

        # 5. 构建索引
        self._build_index(embeddings, chunks)
        logger.info("索引构建完成")

        # 6. 保存知识库
        self._save_kb(chunks, embeddings)
        logger.info(f"知识库已保存到: {self.output_dir}")

        # 7. 生成元数据
        meta = KBMeta(
            source=self.source,
            input_files=all_input_files,
            chunk_strategy=self.chunk_strategy,
            chunk_size=self.chunk_config.chunk_size,
            chunk_overlap=self.chunk_config.chunk_overlap,
            embedding_model=self.embedding_model,
            embedding_dimensions=self.embedding_dimensions,
            chunk_count=len(chunks),
        )

        # 保存元数据
        self._save_meta(meta)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info("=" * 60)
        logger.info(f"知识库构建完成")
        logger.info(f"  耗时: {duration:.2f} 秒")
        logger.info(f"  Source: {self.source}")
        logger.info(f"  Chunks: {len(chunks)}")
        logger.info(f"  策略: {self.chunk_strategy}")
        logger.info(f"  Embedding模型: {self.embedding_model}")
        logger.info(f"  输出目录: {self.output_dir}")
        logger.info("=" * 60)

        return meta

    def _parse_all(
        self,
        doc_files: List[Path],
        code_files: List[Path],
    ) -> List[SourceDocument]:
        """解析所有文件"""
        all_docs = []

        # 解析文档
        for file_path in doc_files:
            try:
                docs = self.doc_parser.parse(file_path, self.source)
                all_docs.extend(docs)
                logger.debug(f"解析文档: {file_path} -> {len(docs)} 个对象")
            except Exception as e:
                logger.error(f"解析文档失败: {file_path}, 错误: {e}")
                raise

        # 解析代码
        for file_path in code_files:
            try:
                docs = self.code_parser.parse(file_path, self.source)
                all_docs.extend(docs)
                logger.debug(f"解析代码: {file_path} -> {len(docs)} 个对象")
            except Exception as e:
                logger.error(f"解析代码失败: {file_path}, 错误: {e}")
                raise

        return all_docs

    def _chunk_all(self, source_docs: List[SourceDocument]) -> List[Chunk]:
        """对所有文档进行分块"""
        # Direct 策略：所有文件都不切，直接注入
        if self.chunk_strategy == "direct":
            self.chunker = create_chunker(strategy="direct")
            all_chunks = self.chunker.chunk_batch(source_docs)
            logger.info(f"Direct 分块: {len(source_docs)} 个文档 -> {len(all_chunks)} 个 chunks（每个文件完整保留）")
            return all_chunks

        # 其他策略：代码文件走 code_aware，文档文件走用户选择的策略
        kwargs = self.chunk_config.to_dict()
        for key in ['strategy', 'chunk_size', 'chunk_overlap', 'eval_questions', 'selection_output']:
            if key in kwargs:
                del kwargs[key]

        code_docs = [d for d in source_docs if d.kind in ("code_c", "code_h", "code_json")]
        other_docs = [d for d in source_docs if d.kind not in ("code_c", "code_h", "code_json")]

        all_chunks = []

        if code_docs:
            code_chunker = create_chunker(
                strategy="code_aware",
                chunk_size=self.chunk_config.chunk_size,
                chunk_overlap=self.chunk_config.chunk_overlap,
            )
            code_chunks = code_chunker.chunk_batch(code_docs)
            all_chunks.extend(code_chunks)
            logger.info(f"Code-aware 分块: {len(code_docs)} 个代码文档 -> {len(code_chunks)} 个 chunks")

        if other_docs:
            merged_docs = self._merge_short_docs(other_docs)
            self.chunker = create_chunker(
                strategy=self.chunk_strategy,
                chunk_size=self.chunk_config.chunk_size,
                chunk_overlap=self.chunk_config.chunk_overlap,
                **kwargs
            )
            other_chunks = self.chunker.chunk_batch(merged_docs)
            all_chunks.extend(other_chunks)
            logger.info(f"{self.chunk_strategy} 分块: {len(other_docs)} 个文档段落"
                        f" -> 合并为 {len(merged_docs)} 个 -> {len(other_chunks)} 个 chunks")

        if not code_docs:
            self.chunker = create_chunker(
                strategy=self.chunk_strategy,
                chunk_size=self.chunk_config.chunk_size,
                chunk_overlap=self.chunk_config.chunk_overlap,
                **kwargs
            )

        return all_chunks

    def _merge_short_docs(
        self, docs: List[SourceDocument], min_length: int = 100
    ) -> List[SourceDocument]:
        """
        合并短段落文档，避免生成大量无意义的碎片 chunk。

        docx 解析时每个段落（含标题、目录、表头）都变成独立 SourceDocument，
        导致 53% 的 doc chunk 不到 20 字符。此方法将同一文件的短段落合并，
        直到达到 min_length 或遇到较长的段落。

        Args:
            docs: 原始 SourceDocument 列表
            min_length: 合并后段落的最小字符数

        Returns:
            合并后的 SourceDocument 列表
        """
        if not docs:
            return docs

        # 按文件分组
        from collections import OrderedDict
        file_groups: OrderedDict[str, List[SourceDocument]] = OrderedDict()
        for doc in docs:
            key = doc.file
            if key not in file_groups:
                file_groups[key] = []
            file_groups[key].append(doc)

        merged = []
        for file_key, file_docs in file_groups.items():
            buffer_text = ""
            buffer_source = file_docs[0].source
            buffer_file = file_docs[0].file
            buffer_kind = file_docs[0].kind

            for doc in file_docs:
                text = doc.text.strip()
                if not text:
                    continue

                # 如果当前段落本身足够长，先 flush 缓冲区，再单独输出
                if len(text) >= min_length:
                    if buffer_text:
                        merged.append(SourceDocument(
                            source=buffer_source,
                            file=buffer_file,
                            kind=buffer_kind,
                            section="merged",
                            text=buffer_text.strip(),
                            metadata={"merged": True},
                        ))
                        buffer_text = ""
                    merged.append(doc)
                else:
                    # 短段落加入缓冲区
                    buffer_text += text + "\n"

                    # 缓冲区达到阈值时 flush
                    if len(buffer_text) >= min_length:
                        merged.append(SourceDocument(
                            source=buffer_source,
                            file=buffer_file,
                            kind=buffer_kind,
                            section="merged",
                            text=buffer_text.strip(),
                            metadata={"merged": True},
                        ))
                        buffer_text = ""

            # flush 剩余缓冲区
            if buffer_text.strip():
                merged.append(SourceDocument(
                    source=buffer_source,
                    file=buffer_file,
                    kind=buffer_kind,
                    section="merged",
                    text=buffer_text.strip(),
                    metadata={"merged": True},
                ))

        logger.info(f"短段落合并: {len(docs)} -> {len(merged)} 个文档")
        return merged

    def _generate_embeddings(self, chunks: List[Chunk]) -> np.ndarray:
        """生成 embeddings"""
        if not self.llm_client:
            self.llm_client = BigModelClient()

        # 提取所有文本
        texts = [chunk.text for chunk in chunks]

        logger.info(f"开始生成 embeddings: {len(texts)} 个文本")

        # 批量调用 embedding API
        embeddings_list = self.llm_client.embed_texts(
            texts=texts,
            model=self.embedding_model,
            dimensions=self.embedding_dimensions,
        )

        # 转换为 numpy 数组
        embeddings = np.array(embeddings_list, dtype=np.float32)

        logger.info(f"Embeddings 生成完成: shape={embeddings.shape}")

        return embeddings

    def _build_index(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        """构建 FAISS 索引"""
        # 创建向量存储（使用内积索引，用于余弦相似度）
        self.vector_store = VectorStore(
            dimension=self.embedding_dimensions,
            metric="ip",
        )

        # 提取 chunk_ids
        chunk_ids = [chunk.chunk_id for chunk in chunks]

        # 添加向量和 ID
        self.vector_store.add_vectors(embeddings, chunk_ids)

        logger.info(f"索引构建完成: {self.vector_store.vector_count} 个向量")

    def _save_kb(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """保存知识库文件"""
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 保存 chunks.jsonl
        chunks_file = self.output_dir / "chunks.jsonl"
        with open(chunks_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(chunk.to_jsonl() + '\n')
        logger.debug(f"保存 chunks: {chunks_file}")

        # 2. 保存 embeddings.npy
        embeddings_file = self.output_dir / "embeddings.npy"
        np.save(embeddings_file, embeddings)
        logger.debug(f"保存 embeddings: {embeddings_file}")

        # 3. 保存索引（通过 vector_store）
        self.vector_store.save(self.output_dir, embeddings)

    def _save_meta(self, meta: KBMeta) -> None:
        """保存元数据"""
        meta_file = self.output_dir / "meta.json"
        with open(meta_file, 'w', encoding='utf-8') as f:
            f.write(meta.to_json())
        logger.debug(f"保存元数据: {meta_file}")


def build_kb(
    source: str,
    input_files: List[Path],
    output_dir: Path,
    chunk_config: Optional[ChunkConfig] = None,
) -> KBMeta:
    """
    便捷函数：构建知识库

    Args:
        source: 来源标识
        input_files: 输入文件列表
        output_dir: 输出目录
        chunk_config: 分块配置对象，默认使用 simple 策略

    Returns:
        KBMeta 知识库元数据

    Example:
        >>> from pathlib import Path
        >>> from herta.kb import ChunkConfig
        >>> files = [
        ...     Path("data/source_proto/gdwx/gdwx.docx"),
        ...     Path("data/source_proto/gdwx/gdwx_config.json"),
        ... ]
        >>> config = ChunkConfig(strategy="semantic", chunk_size=500)
        >>> meta = build_kb("gdwx", files, Path("data/kb/gdwx"), config)
        >>> print(meta.chunk_count)
    """
    # 分类文件
    doc_files = []
    code_files = []

    for file_path in input_files:
        ext = file_path.suffix.lower()
        if ext in ['.docx', '.pdf']:
            doc_files.append(file_path)
        elif ext in ['.json', '.c', '.h']:
            code_files.append(file_path)
        else:
            logger.warning(f"忽略不支持的文件: {file_path}")

    # 创建构建器并执行
    builder = KnowledgeBaseBuilder(
        source=source,
        output_dir=output_dir,
        chunk_config=chunk_config,
    )

    return builder.build(doc_files=doc_files, code_files=code_files)
