"""
知识库（KB）模块

提供知识库的构建、存储和检索功能，支持四种分块策略：
- simple: 固定大小分块
- semantic: 基于语义相似度分块
- autosize: 自动选择最优分块大小
- proposition: 基于原子事实命题分块
"""

from herta.kb.schemas import SourceDocument, Chunk, KBMeta
from herta.kb.chunker import (
    BaseChunker,
    ChunkConfig,
    ChunkReport,
    create_chunker,
)
from herta.kb.simple_chunker import SimpleChunker
from herta.kb.semantic_chunker import SemanticChunker
from herta.kb.selector import AutoSizeChunker, SizeCandidate
from herta.kb.proposition_chunker import PropositionChunker
from herta.kb.store import VectorStore
from herta.kb.builder import KnowledgeBaseBuilder, build_kb

__all__ = [
    # Schemas
    "SourceDocument",
    "Chunk",
    "KBMeta",
    # Chunker Base
    "BaseChunker",
    "ChunkConfig",
    "ChunkReport",
    "create_chunker",
    # Chunker Implementations
    "SimpleChunker",
    "SemanticChunker",
    "AutoSizeChunker",
    "SizeCandidate",
    "PropositionChunker",
    # Store & Builder
    "VectorStore",
    "KnowledgeBaseBuilder",
    "build_kb",
]
