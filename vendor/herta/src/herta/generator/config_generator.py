"""
配置生成器模块

负责完整的配置生成流程：
1. 解析目标协议文档
2. 从知识库检索参考信息
3. 构造 Prompt 并调用 LLM
4. 校验和保存生成的配置
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from herta.generator.schemas import (
    TargetProtocolSpec,
    RetrievedChunk,
    GenerationContext,
    GenerationResult,
)
from herta.generator.target_parser import TargetParser, parse_target_protocol
from herta.generator.retriever import KnowledgeRetriever
from herta.generator.prompt_loader import PromptLoader
from herta.llm.client import BigModelClient
from herta.config import get_settings
from herta.logging import get_logger
from herta.utils.paths import ensure_dir

logger = get_logger(__name__)


class ConfigGenerator:
    """
    配置生成器

    整合目标协议解析、知识库检索、LLM 调用，生成配置文件
    """

    DEFAULT_TEMPLATE = "generate_config.md"
    DEFAULT_TOP_K = 30
    DEFAULT_TEMPERATURE = 0.1

    def __init__(
        self,
        llm_client: Optional[BigModelClient] = None,
        prompt_loader: Optional[PromptLoader] = None,
    ):
        """
        初始化配置生成器

        Args:
            llm_client: LLM 客户端实例
            prompt_loader: Prompt 加载器实例
        """
        self.llm_client = llm_client or BigModelClient()
        self.prompt_loader = prompt_loader or PromptLoader()

        settings = get_settings()
        self.default_model = settings.chat_model

        self.target_parser = TargetParser()

    def generate(
        self,
        target_doc_path: Path,
        source_kb_dir: Path,
        target_name: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
        model: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        output_dir: Optional[Path] = None,
    ) -> GenerationResult:
        """
        生成目标协议的配置文件

        完整流程：
        1. 解析目标协议文档
        2. 从源知识库检索相关信息
        3. 构造生成上下文
        4. 调用 LLM 生成配置
        5. 校验和保存结果

        Args:
            target_doc_path: 目标协议文档路径
            source_kb_dir: 源协议知识库目录
            target_name: 目标协议名称（默认从文件名推断）
            top_k: 检索的参考片段数量
            model: 使用的 LLM 模型
            temperature: 生成温度
            output_dir: 输出目录（默认 output/generated/<target>/）

        Returns:
            GenerationResult 生成结果
        """
        start_time = datetime.now()

        model = model or self.default_model
        if target_name is None:
            target_name = Path(target_doc_path).stem

        logger.info("=" * 60)
        logger.info(f"开始生成配置: target={target_name}, source_kb={source_kb_dir}")
        logger.info("=" * 60)

        # 1. 解析目标协议文档
        try:
            target_spec = self.target_parser.parse(target_doc_path, target_name)
            logger.info(f"目标协议解析完成: {target_name}")
            logger.info(f"  文档长度: {len(target_spec.raw_text)} 字符")
            logger.info(f"  章节数: {len(target_spec.sections)}")
        except Exception as e:
            logger.error(f"解析目标协议文档失败: {e}")
            return self._create_error_result(
                target_name, str(source_kb_dir), top_k, model, temperature, f"解析失败: {e}"
            )

        # 2. 检索：direct 策略直接全量注入，其他策略走向量检索
        try:
            retriever = KnowledgeRetriever(source_kb_dir, self.llm_client)
            retriever.load()

            kb_strategy = (retriever.meta or {}).get("chunk_strategy", "")

            if kb_strategy == "direct":
                # Direct 模式：跳过检索，直接注入所有 chunk
                retrieved_chunks = retriever.retrieve_all()
                logger.info(f"Direct 模式: 直接注入 {len(retrieved_chunks)} 个完整 chunk")
            else:
                # 常规模式：多查询向量检索
                queries = self._build_multi_queries(target_spec)
                for i, q in enumerate(queries):
                    logger.info(f"检索 Query {i+1}/{len(queries)}: {q[:100]}...")

                retrieved_chunks = retriever.retrieve_multi(
                    queries=queries,
                    top_k_per_query=10,
                    max_total=top_k,
                )
                logger.info(f"多查询检索完成: 找到 {len(retrieved_chunks)} 个相关片段")
        except Exception as e:
            logger.error(f"知识库检索失败: {e}")
            return self._create_error_result(
                target_name, str(source_kb_dir), top_k, model, temperature, f"检索失败: {e}"
            )

        # 3. 构造生成上下文
        context = GenerationContext(
            target=target_spec,
            retrieved_chunks=retrieved_chunks,
            source_kb=str(source_kb_dir),
            top_k=top_k,
            model=model,
            temperature=temperature,
        )

        # 4. 调用 LLM 生成配置
        try:
            config, raw_response = self._generate_config_with_llm(context)
            logger.info("LLM 生成完成")
        except Exception as e:
            logger.error(f"LLM 生成失败: {e}")
            return self._create_error_result(
                target_name, str(source_kb_dir), top_k, model, temperature, f"生成失败: {e}"
            )

        # 5. 创建生成结果
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        result = GenerationResult(
            config=config,
            raw_response=raw_response,
            target=target_name,
            source_kb=str(source_kb_dir),
            retrieved_chunk_count=len(retrieved_chunks),
            model=model,
            temperature=temperature,
            is_valid_json=True,
            error_message=None,
        )

        # 6. 保存结果
        if output_dir is None:
            output_dir = Path(f"output/generated/{target_name}")

        output_dir = ensure_dir(output_dir)
        output_path = output_dir / f"{target_name}_config.json"

        try:
            result.save_config(output_path)
            logger.info(f"配置已保存: {output_path}")
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            result.error_message = f"保存失败: {e}"

        # 输出摘要
        logger.info("")
        logger.info("=" * 60)
        logger.info("✅ 配置生成完成！")
        logger.info("=" * 60)
        logger.info(f"  Target:       {target_name}")
        logger.info(f"  Source KB:    {source_kb_dir}")
        logger.info(f"  Top-K:        {top_k}")
        logger.info(f"  检索片段数:   {len(retrieved_chunks)}")
        logger.info(f"  LLM 模型:     {model}")
        logger.info(f"  生成耗时:     {duration:.2f} 秒")
        logger.info(f"  输出路径:     {output_path}")
        logger.info("=" * 60)

        return result

    def _build_retrieval_query(self, target_spec: TargetProtocolSpec) -> str:
        """
        构造检索 Query

        从目标协议文档内容中提取关键上下文片段构造检索 query，
        而非使用固定关键词。

        Args:
            target_spec: 目标协议规范

        Returns:
            检索 query 字符串
        """
        from herta.utils.text import extract_section_keywords

        parts = [f"目标协议: {target_spec.target_name}"]

        raw = target_spec.raw_text

        # 从文档中提取命令相关上下文
        cmd_ctx = extract_section_keywords(
            raw, ["命令", "指令", "命令字", "command", "cmd"]
        )
        if cmd_ctx:
            parts.append(f"命令相关: {cmd_ctx[:300]}")

        # 提取遥测相关上下文
        tele_ctx = extract_section_keywords(
            raw, ["遥测", "帧", "telemetry", "frame"]
        )
        if tele_ctx:
            parts.append(f"遥测相关: {tele_ctx[:300]}")

        # 提取内部处理相关上下文
        internal_ctx = extract_section_keywords(
            raw, ["内部", "internal", "通信", "数据帧", "同步"]
        )
        if internal_ctx:
            parts.append(f"内部处理: {internal_ctx[:200]}")

        # 添加摘要作为补充
        if target_spec.summary:
            parts.append(f"协议摘要: {target_spec.summary[:300]}")

        return "\n".join(parts)

    def _build_multi_queries(self, target_spec: TargetProtocolSpec) -> list[str]:
        """
        构造多个目标 query 用于多维度检索

        尽可能多地从文档提取上下文构造语义丰富的 query
        """
        from herta.utils.text import extract_section_keywords

        queries = []
        target = target_spec.target_name
        raw = target_spec.raw_text

        # Query 1: 命令列表
        cmd_query_parts = [f"{target} 命令列表 cmd_list", "命令字 handler 处理函数"]
        cmd_keywords = extract_section_keywords(raw, ["命令", "指令", "command"])
        if cmd_keywords:
            cmd_query_parts.append(cmd_keywords[:500])
        queries.append(" ".join(cmd_query_parts))

        # Query 2: 遥测处理器
        tele_query_parts = [f"{target} 遥测 tele_handler", "遥测帧 数据帧 frame telemetry"]
        tele_keywords = extract_section_keywords(raw, ["遥测", "帧", "telemetry"])
        if tele_keywords:
            tele_query_parts.append(tele_keywords[:500])
        queries.append(" ".join(tele_query_parts))

        # Query 3: 内部处理器
        internal_query_parts = [f"{target} internal_handler 内部处理", "数据长度 len handler 同步 校验"]
        internal_keywords = extract_section_keywords(raw, ["内部处理", "内部通信", "同步", "校验"])
        if internal_keywords:
            internal_query_parts.append(internal_keywords[:500])
        queries.append(" ".join(internal_query_parts))

        # Query 4: 完整文档摘要作为语义 query
        queries.append(target_spec.summary[:800])

        # Query 5: 配置 JSON 结构模板
        queries.append("config json cmd_list tele_handler internal_handler version start_index")

        # Query 6: handler 函数命名模式
        queries.append(f"{target}_handler 处理函数名 命名规则 handler desc cmd len")

        return queries

    def _generate_config_with_llm(self, context: GenerationContext) -> tuple[Dict[str, Any], str]:
        """
        使用 LLM 生成配置

        Args:
            context: 生成上下文

        Returns:
            (config_dict, raw_response) 元组

        Raises:
            RuntimeError: LLM 调用失败或返回无效 JSON
        """
        # 准备模板变量
        variables = {
            "target_summary": context.format_target_summary(),
            "retrieved_context": context.format_retrieved_context(),
            "output_schema": context.output_schema,
        }

        # 加载并渲染 Prompt
        try:
            prompt = self.prompt_loader.load_and_render(self.DEFAULT_TEMPLATE, variables)
            logger.debug(f"Prompt 渲染完成: {len(prompt)} 字符")
        except FileNotFoundError:
            # 如果模板不存在，使用内置默认模板
            logger.warning(f"模板 {self.DEFAULT_TEMPLATE} 不存在，使用默认模板")
            prompt = self._get_default_prompt(variables)

        # 调用 LLM
        messages = [
            {"role": "system", "content": "你是一个专业的协议配置生成助手。请严格按照指定的 JSON 结构输出配置文件，不要增减任何顶层字段。"},
            {"role": "user", "content": prompt},
        ]

        try:
            raw_response = self.llm_client.chat(
                messages=messages,
                model=context.model,
                temperature=context.temperature,
            )
            logger.debug(f"LLM 响应: {raw_response[:200]}...")
        except Exception as e:
            raise RuntimeError(f"LLM 调用失败: {e}") from e

        # 解析 JSON
        try:
            config = self._extract_and_parse_json(raw_response)
            return config, raw_response
        except Exception as e:
            raise RuntimeError(f"解析 LLM 返回的 JSON 失败: {e}\n原始响应:\n{raw_response[:500]}...") from e

    def _extract_and_parse_json(self, text: str) -> Dict[str, Any]:
        """
        从文本中提取并解析 JSON

        处理 LLM 返回的各种格式：
        - 纯 JSON
        - 包含在 ```json ... ``` 代码块中
        - 包含在 ``` ... ``` 代码块中
        - 包含额外文本

        Args:
            text: 原始文本

        Returns:
            解析后的 JSON 字典

        Raises:
            json.JSONDecodeError: 解析失败
        """
        text = text.strip()

        # 尝试 1: 直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 尝试 2: 提取 ```json ... ``` 代码块
        import re
        json_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        matches = re.findall(json_block_pattern, text, re.DOTALL)

        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # 尝试 3: 查找第一个 { 和最后一个 } 之间的内容
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and start < end:
            try:
                return json.loads(text[start:end+1])
            except json.JSONDecodeError:
                pass

        # 尝试 4: 查找第一个 [ 和最后一个 ] 之间的内容（如果配置是数组）
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1 and start < end:
            try:
                return json.loads(text[start:end+1])
            except json.JSONDecodeError:
                pass

        # 所有尝试都失败
        raise json.JSONDecodeError("无法从文本中提取有效的 JSON", text, 0)

    def _get_default_prompt(self, variables: Dict[str, str]) -> str:
        """
        获取默认 Prompt 模板（当模板文件不存在时使用）

        Args:
            variables: 模板变量

        Returns:
            默认 Prompt 字符串
        """
        return f"""【任务】
根据以下目标协议信息和参考上下文，生成一份完整的协议配置文件（config.json）。

{variables.get('target_summary', '')}

【参考上下文】
{variables.get('retrieved_context', '')}

【输出要求】
请输出严格符合以下 JSON 结构的配置（不要增减任何顶层字段）：
{variables.get('output_schema', '')}

【关键规则】
1. handler 命名必须遵循 {{target}}_功能名称 的模式
2. cmd 和 len 必须是整数类型
3. 所有值必须从参考上下文或目标文档中提取，禁止虚构
4. 顶层只能有 version、author、start_index、cmd_list、tele_handler、internal_handler 六个字段
5. cmd_list/tele_handler/internal_handler 的数量必须与目标文档描述一致

请直接输出 JSON："""

    def _create_error_result(
        self,
        target: str,
        source_kb: str,
        top_k: int,
        model: str,
        temperature: float,
        error_message: str,
    ) -> GenerationResult:
        """创建错误结果"""
        return GenerationResult(
            config={"error": error_message, "target": target},
            raw_response="",
            target=target,
            source_kb=source_kb,
            retrieved_chunk_count=0,
            model=model,
            temperature=temperature,
            is_valid_json=False,
            error_message=error_message,
        )


def generate_config(
    target_doc_path: str,
    source_kb_dir: str,
    target_name: Optional[str] = None,
    top_k: int = 5,
    output_dir: Optional[str] = None,
) -> GenerationResult:
    """
    便捷函数：生成配置文件

    Args:
        target_doc_path: 目标协议文档路径
        source_kb_dir: 源协议知识库目录
        target_name: 目标协议名称
        top_k: 检索片段数量
        output_dir: 输出目录

    Returns:
        GenerationResult 生成结果

    Example:
        >>> result = generate_config(
        ...     "data/target_proto/dllg/dllg.docx",
        ...     "data/kb/gdwx",
        ...     target_name="dllg",
        ... )
        >>> print(result.output_path)
    """
    generator = ConfigGenerator()
    return generator.generate(
        target_doc_path=Path(target_doc_path),
        source_kb_dir=Path(source_kb_dir),
        target_name=target_name,
        top_k=top_k,
        output_dir=Path(output_dir) if output_dir else None,
    )
