"""
Handler 生成器模块

采用两阶段生成策略：
1. 阶段A：生成 handler 设计草图（plan）
2. 阶段B：基于 plan 生成完整 C 代码

输出：
- handler.c：完整的 C 代码文件
- handler_plan.md（可选）：设计草图文档
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from herta.generator.schemas import TargetProtocolSpec, RetrievedChunk
from herta.generator.target_parser import TargetParser
from herta.generator.retriever import KnowledgeRetriever
from herta.generator.prompt_loader import PromptLoader
from herta.llm.client import BigModelClient
from herta.config import get_settings
from herta.logging import get_logger
from herta.utils.paths import ensure_dir

logger = get_logger(__name__)


@dataclass
class HandlerGenerationResult:
    """Handler 生成结果"""

    code: str
    plan: str
    target: str
    source_kb: str
    retrieved_chunk_count: int
    model: str
    temperature: float
    code_output_path: Optional[str] = None
    plan_output_path: Optional[str] = None
    generation_time: Optional[str] = None
    is_valid: bool = True
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.generation_time is None:
            self.generation_time = datetime.now().isoformat()

    def save_code(self, output_path: Path) -> None:
        """保存代码到文件"""
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        output_path.write_text(self.code, encoding='utf-8')
        self.code_output_path = str(output_path)
        logger.info(f"Handler 代码已保存: {output_path}")

    def save_plan(self, output_path: Path) -> None:
        """保存设计计划到文件"""
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        output_path.write_text(self.plan, encoding='utf-8')
        self.plan_output_path = str(output_path)
        logger.info(f"Handler 设计计划已保存: {output_path}")


class HandlerGenerator:
    """
    Handler 生成器

    两阶段生成：
    1. 生成设计草图（plan）
    2. 基于 plan 生成完整 C 代码
    """

    PLAN_TEMPLATE = "generate_handler_plan.md"
    CODE_TEMPLATE = "generate_handler_code.md"
    DEFAULT_TOP_K = 30  # handler 生成需要更多参考
    DEFAULT_TEMPERATURE = 0.1

    def __init__(
        self,
        llm_client: Optional[BigModelClient] = None,
        prompt_loader: Optional[PromptLoader] = None,
    ):
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
        save_plan: bool = False,
        intermediate_dir: Optional[Path] = None,
    ) -> HandlerGenerationResult:
        """
        生成 Handler 代码（两阶段）

        Args:
            target_doc_path: 目标协议文档路径
            source_kb_dir: 源协议知识库目录
            target_name: 目标协议名称
            top_k: 检索的参考片段数量
            model: 使用的 LLM 模型
            temperature: 生成温度
            output_dir: 代码输出目录
            save_plan: 是否保存设计计划
            intermediate_dir: 设计计划输出目录

        Returns:
            HandlerGenerationResult 生成结果
        """
        start_time = datetime.now()

        model = model or self.default_model
        if target_name is None:
            target_name = Path(target_doc_path).stem

        logger.info("=" * 60)
        logger.info(f"开始生成 Handler: target={target_name}, source_kb={source_kb_dir}")
        logger.info("=" * 60)

        # 1. 解析目标协议文档
        try:
            target_spec = self.target_parser.parse(target_doc_path, target_name)
            logger.info(f"目标协议解析完成: {target_name}")
        except Exception as e:
            logger.error(f"解析目标协议文档失败: {e}")
            return self._create_error_result(
                target_name, str(source_kb_dir), top_k, model, temperature,
                f"解析失败: {e}"
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
                queries = self._build_handler_multi_queries(target_spec)
                for i, q in enumerate(queries):
                    logger.info(f"Handler 检索 Query {i+1}/{len(queries)}: {q[:100]}...")

                retrieved_chunks = retriever.retrieve_multi(
                    queries=queries,
                    top_k_per_query=10,
                    max_total=top_k,
                )
                logger.info(f"多查询检索完成: 找到 {len(retrieved_chunks)} 个相关片段")
        except Exception as e:
            logger.error(f"知识库检索失败: {e}")
            return self._create_error_result(
                target_name, str(source_kb_dir), top_k, model, temperature,
                f"检索失败: {e}"
            )

        # ========== 阶段A：生成设计草图 ==========
        logger.info("-" * 60)
        logger.info("阶段A：生成 Handler 设计草图...")
        logger.info("-" * 60)

        try:
            plan = self._generate_plan(target_spec, retrieved_chunks, model, temperature)
            logger.info("设计草图生成完成")
            logger.debug(f"Plan 预览: {plan[:300]}...")
        except Exception as e:
            logger.error(f"生成设计草图失败: {e}")
            return self._create_error_result(
                target_name, str(source_kb_dir), top_k, model, temperature,
                f"阶段A失败: {e}"
            )

        # 保存设计计划（如果需要）
        plan_path = None
        if save_plan:
            if intermediate_dir is None:
                intermediate_dir = Path(f"output/intermediate/{target_name}")
            plan_path = ensure_dir(intermediate_dir) / "handler_plan.md"

        # ========== 阶段B：生成完整代码 ==========
        logger.info("-" * 60)
        logger.info("阶段B：生成 Handler C 代码...")
        logger.info("-" * 60)

        try:
            code = self._generate_code(target_spec, plan, retrieved_chunks, model, temperature)
            logger.info("代码生成完成")

            # 验证代码有效性
            if not self._validate_code(code):
                logger.warning("生成的代码可能不完整或格式异常")

        except Exception as e:
            logger.error(f"生成代码失败: {e}")
            return self._create_error_result(
                target_name, str(source_kb_dir), top_k, model, temperature,
                f"阶段B失败: {e}"
            )

        # 创建结果
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        result = HandlerGenerationResult(
            code=code,
            plan=plan,
            target=target_name,
            source_kb=str(source_kb_dir),
            retrieved_chunk_count=len(retrieved_chunks),
            model=model,
            temperature=temperature,
            is_valid=True,
            error_message=None,
        )

        # 保存输出
        if output_dir is None:
            output_dir = Path(f"output/generated/{target_name}")
        output_dir = ensure_dir(output_dir)

        # 保存代码
        code_path = output_dir / f"{target_name}_handler.c"
        result.save_code(code_path)

        # 保存计划（如果需要）
        if save_plan and plan_path:
            result.save_plan(plan_path)

        # 输出摘要
        logger.info("")
        logger.info("=" * 60)
        logger.info("✅ Handler 生成完成！")
        logger.info("=" * 60)
        logger.info(f"  Target:       {target_name}")
        logger.info(f"  Source KB:    {source_kb_dir}")
        logger.info(f"  Top-K:        {top_k}")
        logger.info(f"  检索片段数:   {len(retrieved_chunks)}")
        logger.info(f"  LLM 模型:     {model}")
        logger.info(f"  生成耗时:     {duration:.2f} 秒")
        logger.info(f"  代码文件:     {result.code_output_path}")
        if result.plan_output_path:
            logger.info(f"  设计计划:     {result.plan_output_path}")
        logger.info("=" * 60)

        return result

    def _build_handler_retrieval_query(self, target_spec: TargetProtocolSpec) -> str:
        """
        构造面向 handler 的检索 Query

        从目标协议文档内容中提取函数设计、命令分发、数据结构相关的
        上下文片段构造检索 query。
        """
        from herta.utils.text import extract_section_keywords

        parts = [f"目标协议 {target_spec.target_name} handler 实现"]

        raw = target_spec.raw_text

        # 提取函数/处理相关的上下文
        func_ctx = extract_section_keywords(
            raw, ["处理函数", "handler", "入口", "entry", "分发", "dispatch"]
        )
        if func_ctx:
            parts.append(f"函数设计: {func_ctx[:400]}")

        # 提取命令分发相关上下文
        dispatch_ctx = extract_section_keywords(
            raw, ["命令字", "命令码", "指令码", "switch", "case"]
        )
        if dispatch_ctx:
            parts.append(f"命令分发: {dispatch_ctx[:300]}")

        # 提取数据结构相关上下文
        struct_ctx = extract_section_keywords(
            raw, ["数据结构", "结构体", "帧格式", "数据包"]
        )
        if struct_ctx:
            parts.append(f"数据结构: {struct_ctx[:300]}")

        # 协议摘要作为补充
        if target_spec.summary:
            parts.append(f"协议摘要: {target_spec.summary[:200]}")

        return "\n".join(parts)

    def _build_handler_multi_queries(self, target_spec: TargetProtocolSpec) -> list[str]:
        """
        构造多个 handler 专用检索 query

        尽可能多地覆盖函数模式、命令分发、遥测解析、内部通信、数据结构等维度
        """
        from herta.utils.text import extract_section_keywords

        queries = []
        target = target_spec.target_name
        raw = target_spec.raw_text

        # Query 1: 函数模式和入口
        func_parts = [f"{target} handler 函数实现", "处理函数 入口 handler C代码"]
        func_ctx = extract_section_keywords(raw, ["处理函数", "handler", "入口"])
        if func_ctx:
            func_parts.append(func_ctx[:500])
        queries.append(" ".join(func_parts))

        # Query 2: 命令分发逻辑
        dispatch_parts = [f"{target} 命令分发 switch case", "cmd_handler command dispatch"]
        dispatch_ctx = extract_section_keywords(raw, ["命令字", "命令码", "switch", "case"])
        if dispatch_ctx:
            dispatch_parts.append(dispatch_ctx[:500])
        queries.append(" ".join(dispatch_parts))

        # Query 3: 遥测解析函数
        tele_parts = [f"{target} 遥测解析 tele_handler", "帧解析 telemetry parse"]
        tele_ctx = extract_section_keywords(raw, ["遥测", "帧解析", "telemetry"])
        if tele_ctx:
            tele_parts.append(tele_ctx[:500])
        queries.append(" ".join(tele_parts))

        # Query 4: 内部通信处理
        internal_parts = [f"{target} internal_handler 内部通信", "数据帧 处理 同步 校验"]
        internal_ctx = extract_section_keywords(raw, ["内部通信", "内部处理", "同步"])
        if internal_ctx:
            internal_parts.append(internal_ctx[:500])
        queries.append(" ".join(internal_parts))

        # Query 5: 数据结构和帧格式
        struct_parts = [f"{target} 数据结构 struct", "帧格式 数据包 解析 C语言"]
        struct_ctx = extract_section_keywords(raw, ["数据结构", "结构体", "帧格式"])
        if struct_ctx:
            struct_parts.append(struct_ctx[:500])
        queries.append(" ".join(struct_parts))

        # Query 6: 参考实现代码（用摘要找整体代码模式）
        queries.append(target_spec.summary[:800])

        return queries

    def _generate_plan(
        self,
        target_spec: TargetProtocolSpec,
        retrieved_chunks: list[RetrievedChunk],
        model: str,
        temperature: float,
    ) -> str:
        """
        阶段A：生成 Handler 设计草图
        """
        # 准备上下文（不截断）
        context_text = self._format_retrieved_context(retrieved_chunks)
        target_summary = self._format_target_summary(target_spec)

        # 准备模板变量
        variables = {
            "target_summary": target_summary,
            "retrieved_context": context_text,
        }

        # 加载并渲染 Prompt
        try:
            prompt = self.prompt_loader.load_and_render(self.PLAN_TEMPLATE, variables)
        except FileNotFoundError:
            logger.warning(f"模板 {self.PLAN_TEMPLATE} 不存在，使用默认模板")
            prompt = self._get_default_plan_prompt(variables)

        # 调用 LLM
        messages = [
            {"role": "system", "content": "你是一位资深的嵌入式系统架构师，专注于协议处理模块的设计。"},
            {"role": "user", "content": prompt},
        ]

        plan = self.llm_client.chat(
            messages=messages,
            model=model,
            temperature=temperature,
        )

        return plan

    def _generate_code(
        self,
        target_spec: TargetProtocolSpec,
        plan: str,
        retrieved_chunks: list[RetrievedChunk],
        model: str,
        temperature: float,
    ) -> str:
        """
        阶段B：基于 Plan 生成完整 C 代码
        """
        # 准备上下文（不截断，给 LLM 完整信息）
        context_text = self._format_retrieved_context(retrieved_chunks)
        target_summary = self._format_target_summary(target_spec)

        # 准备模板变量
        variables = {
            "target_summary": target_summary,
            "handler_plan": plan,
            "retrieved_context": context_text,
        }

        # 加载并渲染 Prompt
        try:
            prompt = self.prompt_loader.load_and_render(self.CODE_TEMPLATE, variables)
        except FileNotFoundError:
            logger.warning(f"模板 {self.CODE_TEMPLATE} 不存在，使用默认模板")
            prompt = self._get_default_code_prompt(variables)

        # 调用 LLM
        messages = [
            {"role": "system", "content": "你是一位资深的嵌入式 C 语言开发者，擅长编写高效、健壮的协议处理代码。"},
            {"role": "user", "content": prompt},
        ]

        code = self.llm_client.chat(
            messages=messages,
            model=model,
            temperature=temperature,
        )

        # 清理代码（移除可能的 markdown 围栏）
        code = self._clean_code_output(code)

        return code

    def _format_retrieved_context(
        self,
        chunks: list[RetrievedChunk],
        max_length: int = 0,
    ) -> str:
        """格式化检索到的上下文，默认不截断"""
        if not chunks:
            return "（未检索到相关参考信息）"

        sections = []
        for i, chunk in enumerate(chunks, 1):
            sections.append(f"【参考片段 {i}】来源: {chunk.source} | 文件: {chunk.file} | 相关度: {chunk.score:.3f}\n{chunk.text}\n---")

        context = "\n\n".join(sections)

        if max_length and len(context) > max_length:
            context = context[:max_length] + "\n...（上下文已截断）"

        return context

    def _format_target_summary(self, target_spec: TargetProtocolSpec) -> str:
        """格式化目标协议信息，传入完整原文"""
        sections_info = ""
        if target_spec.sections:
            shown = target_spec.sections[:20]
            sections_info = "\n文档结构:\n" + "\n".join(
                f"  - {s.get('title', s.get('name', '未命名章节'))}"
                for s in shown
            )
            if len(target_spec.sections) > 20:
                sections_info += f"\n  ... 等共 {len(target_spec.sections)} 个章节"

        return f"""【目标协议】
名称: {target_spec.target_name}
文档路径: {target_spec.document_path}
结构化摘要: {target_spec.summary}{sections_info}

【目标协议完整原文】
{target_spec.raw_text}
"""

    def _clean_code_output(self, code: str) -> str:
        """清理 LLM 输出的代码"""
        code = code.strip()

        # 移除 ```c 或 ``` 围栏
        if code.startswith("```"):
            lines = code.split('\n')
            # 移除第一行的 ```c 或 ```
            if lines[0].startswith("```"):
                lines = lines[1:]
            # 移除最后一行的 ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = '\n'.join(lines).strip()

        return code

    def _validate_code(self, code: str) -> bool:
        """
        简单验证代码有效性

        检查：
        - 代码不为空
        - 包含基本的 C 语法结构
        - 不是纯文本/说明
        """
        if not code or len(code) < 100:
            return False

        # 检查是否包含 C 代码特征
        c_indicators = [
            "#include",
            "int ",
            "void ",
            "static ",
            "{",
            "}",
            ";",
            "/*",
            "//",
        ]

        indicator_count = sum(1 for ind in c_indicators if ind in code)
        return indicator_count >= 3  # 至少包含3个C特征

    def _get_default_plan_prompt(self, variables: Dict[str, str]) -> str:
        """默认设计计划 Prompt（模板不存在时使用）"""
        return f"""请为以下目标协议设计一份 handler 实现计划：

{variables.get('target_summary', '')}

参考信息：
{variables.get('retrieved_context', '')}

请输出结构化的设计计划，包括：
1. 概述和职责
2. 函数列表（含输入输出）
3. 主入口函数设计
4. 命令分发逻辑
5. 遥测处理逻辑
6. 内部通信逻辑
7. 辅助函数设计
8. 未知点与 TODO

只输出设计文档，不要包含具体的 C 代码实现。"""

    def _get_default_code_prompt(self, variables: Dict[str, str]) -> str:
        """默认代码生成 Prompt（模板不存在时使用）"""
        return f"""请根据以下设计计划，生成完整的 C 语言 handler 实现代码：

{variables.get('target_summary', '')}

设计计划：
{variables.get('handler_plan', '')}

参考代码片段：
{variables.get('retrieved_context', '')}

要求：
1. 只输出纯 C 代码，不要包含 markdown 标记
2. 包含所有必要的 include、宏定义、函数声明和实现
3. 不确定的地方用 /* TODO: ... */ 标记
4. 代码风格与参考代码保持一致
5. 确保代码结构清晰，有适当的错误处理

请直接输出完整的 C 代码："""

    def _create_error_result(
        self,
        target: str,
        source_kb: str,
        top_k: int,
        model: str,
        temperature: float,
        error_message: str,
    ) -> HandlerGenerationResult:
        """创建错误结果"""
        return HandlerGenerationResult(
            code=f"/* Error: {error_message} */",
            plan=f"/* Error: {error_message} */",
            target=target,
            source_kb=source_kb,
            retrieved_chunk_count=0,
            model=model,
            temperature=temperature,
            is_valid=False,
            error_message=error_message,
        )


def generate_handler(
    target_doc_path: str,
    source_kb_dir: str,
    target_name: Optional[str] = None,
    top_k: int = 8,
    save_plan: bool = False,
    output_dir: Optional[str] = None,
) -> HandlerGenerationResult:
    """
    便捷函数：生成 Handler 代码

    Args:
        target_doc_path: 目标协议文档路径
        source_kb_dir: 源协议知识库目录
        target_name: 目标协议名称
        top_k: 检索片段数量
        save_plan: 是否保存设计计划
        output_dir: 输出目录

    Returns:
        HandlerGenerationResult 生成结果
    """
    generator = HandlerGenerator()
    return generator.generate(
        target_doc_path=Path(target_doc_path),
        source_kb_dir=Path(source_kb_dir),
        target_name=target_name,
        top_k=top_k,
        save_plan=save_plan,
        output_dir=Path(output_dir) if output_dir else None,
    )
