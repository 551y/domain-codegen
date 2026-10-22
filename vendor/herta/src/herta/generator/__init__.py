"""
生成器模块

提供配置生成功能：
- 目标协议文档解析
- 知识库检索
- Prompt 模板加载
- 配置生成
"""

from herta.generator.schemas import (
    TargetProtocolSpec,
    RetrievedChunk,
    GenerationContext,
    GenerationResult,
)
from herta.generator.target_parser import TargetParser, parse_target_protocol
from herta.generator.retriever import KnowledgeRetriever, retrieve_from_kb
from herta.generator.prompt_loader import PromptLoader, load_prompt, render_prompt
from herta.generator.config_generator import ConfigGenerator, generate_config
from herta.generator.handler_generator import (
    HandlerGenerator,
    HandlerGenerationResult,
    generate_handler,
)

__all__ = [
    # Schemas
    "TargetProtocolSpec",
    "RetrievedChunk",
    "GenerationContext",
    "GenerationResult",
    # Target Parser
    "TargetParser",
    "parse_target_protocol",
    # Retriever
    "KnowledgeRetriever",
    "retrieve_from_kb",
    # Prompt Loader
    "PromptLoader",
    "load_prompt",
    "render_prompt",
    # Config Generator
    "ConfigGenerator",
    "generate_config",
    # Handler Generator
    "HandlerGenerator",
    "HandlerGenerationResult",
    "generate_handler",
]
