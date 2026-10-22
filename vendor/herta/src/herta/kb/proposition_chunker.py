"""
命题分块策略实现模块

使用 LLM 将文本段落拆解为原子事实命题，然后转为 Chunk
适合知识密集型文档，每个命题都是独立可检索的知识单元
"""

import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from herta.kb.schemas import SourceDocument, Chunk
from herta.kb.chunker import BaseChunker, ChunkConfig
from herta.llm.client import BigModelClient
from herta.logging import get_logger
from herta.utils.paths import ensure_dir

logger = get_logger(__name__)


@dataclass
class Proposition:
    """原子事实命题"""

    text: str
    source_text: str
    index: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source_text": self.source_text,
            "index": self.index,
        }


class PropositionChunker(BaseChunker):
    """
    命题分块策略

    使用 LLM 把文本拆解为原子事实命题，每个命题成为一个 Chunk
    """

    # Prompt 模板用于提取命题
    PROPOSITION_EXTRACTION_PROMPT = """请分析以下文本，提取出所有原子事实命题（atomic factual propositions）。

原子事实命题的定义：
- 一个独立、自包含的事实陈述
- 包含完整的主语、谓语、宾语（如果有）
- 可以被单独理解和验证
- 不包含指代或上下文依赖

文本：
{text}

请输出 JSON 格式的命题列表：
{{
  "propositions": [
    "命题1",
    "命题2",
    ...
  ]
}}

注意：
- 每个命题必须是完整的句子
- 不要合并多个事实
- 如果文本不包含明确事实，返回空列表
- 只输出 JSON，不要其他说明"""

    def __init__(self, config: ChunkConfig):
        super().__init__(config)
        if config.strategy != "proposition":
            raise ValueError(f"PropositionChunker 需要 strategy='proposition', 得到 '{config.strategy}'")

        # 初始化 LLM 客户端
        self.llm_client = BigModelClient()

        # 参数
        self.proposition_model = config.proposition_model
        self.max_propositions_per_chunk = config.max_propositions_per_chunk
        self.proposition_batch_size = config.proposition_batch_size
        self.proposition_cache = config.proposition_cache

        # 缓存目录
        self._cache_dir = Path(".cache/propositions")
        if self.proposition_cache:
            ensure_dir(self._cache_dir)

        # 统计
        self._stats = {
            "total_paragraphs": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_propositions": 0,
        }

    def chunk(self, doc: SourceDocument) -> List[Chunk]:
        """
        将文档分块为命题

        流程：
        1. 将文本切分为段落
        2. 对每个段落提取命题（分批处理）
        3. 将命题转换为 Chunks
        """
        text = doc.text
        if not text.strip():
            logger.warning(f"文档内容为空，跳过: {doc.file}")
            return []

        # 切分为段落
        paragraphs = self._split_into_paragraphs(text)
        logger.debug(f"文档切分为 {len(paragraphs)} 个段落")

        all_propositions: List[Proposition] = []

        # 分批处理段落
        for i in range(0, len(paragraphs), self.proposition_batch_size):
            batch = paragraphs[i:i + self.proposition_batch_size]
            batch_num = i // self.proposition_batch_size + 1
            total_batches = (len(paragraphs) + self.proposition_batch_size - 1) // self.proposition_batch_size

            logger.debug(f"处理段落批次 {batch_num}/{total_batches}: {len(batch)} 个段落")

            for para_idx, paragraph in enumerate(batch):
                global_idx = i + para_idx
                propositions = self._extract_propositions(paragraph, global_idx)
                all_propositions.extend(propositions)

        # 转换为 Chunks
        chunks = self._propositions_to_chunks(doc, all_propositions)

        # 记录统计
        self._stats["total_paragraphs"] = len(paragraphs)
        self._stats["total_propositions"] = len(all_propositions)

        logger.info(f"命题分块完成: {len(paragraphs)} 个段落 -> {len(all_propositions)} 个命题 -> {len(chunks)} 个 chunks")
        return chunks

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        将文本切分为段落

        按空行分割，同时保持段落完整性
        """
        # 按多个换行符分割
        raw_paragraphs = re.split(r'\n\s*\n', text)

        paragraphs = []
        for para in raw_paragraphs:
            para = para.strip()
            # 过滤太短的段落（少于20字符可能是格式标记）
            if len(para) >= 20:
                paragraphs.append(para)
            elif para and len(para) >= 5:
                # 较短的段落合并到前一个（如果有）
                if paragraphs:
                    paragraphs[-1] += " " + para
                else:
                    paragraphs.append(para)

        return paragraphs

    def _extract_propositions(self, paragraph: str, index: int) -> List[Proposition]:
        """
        从段落中提取命题

        使用 LLM 进行提取，支持缓存
        """
        self._stats["total_paragraphs"] += 1

        # 检查缓存
        if self.proposition_cache:
            cache_key = self._get_cache_key(paragraph)
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                self._stats["cache_hits"] += 1
                logger.debug(f"缓存命中: paragraph {index}")
                return [Proposition(p, paragraph, index) for p in cached]
            self._stats["cache_misses"] += 1

        # 调用 LLM 提取
        try:
            prompt = self.PROPOSITION_EXTRACTION_PROMPT.format(text=paragraph)

            messages = [
                {"role": "system", "content": "你是一个专业的文本分析助手，擅长提取原子事实命题。"},
                {"role": "user", "content": prompt},
            ]

            response = self.llm_client.chat(
                messages=messages,
                model=self.proposition_model,
                temperature=0.1,
            )

            # 解析响应
            propositions_text = self._parse_propositions_response(response)

            # 创建 Proposition 对象
            propositions = [
                Proposition(p, paragraph, index)
                for p in propositions_text
            ]

            self._stats["successful_extractions"] += 1

            # 保存到缓存
            if self.proposition_cache:
                self._save_to_cache(cache_key, propositions_text)

            return propositions

        except Exception as e:
            logger.warning(f"提取命题失败 (paragraph {index}): {e}")
            self._stats["failed_extractions"] += 1

            # 回退策略：将整个段落作为一个命题
            return [Proposition(paragraph, paragraph, index)]

    def _parse_propositions_response(self, response: str) -> List[str]:
        """
        解析 LLM 响应中的命题列表

        处理各种可能的格式：
        - JSON 格式
        - 纯文本列表
        - Markdown 代码块
        """
        response = response.strip()

        # 尝试从 Markdown 代码块中提取
        if "```json" in response or "```" in response:
            import re
            code_blocks = re.findall(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
            if code_blocks:
                response = code_blocks[0].strip()

        # 尝试 JSON 解析
        try:
            data = json.loads(response)
            if isinstance(data, dict) and "propositions" in data:
                props = data["propositions"]
                if isinstance(props, list):
                    return [p.strip() for p in props if p.strip()]
            elif isinstance(data, list):
                return [p.strip() for p in data if p.strip()]
        except json.JSONDecodeError:
            pass

        # 尝试按行解析（每行一个命题）
        lines = response.split('\n')
        propositions = []
        for line in lines:
            line = line.strip()
            # 移除常见的列表标记
            line = re.sub(r'^[\-\*\d]+[\.\)]?\s*', '', line)
            if line and len(line) > 10:  # 过滤太短的
                propositions.append(line)

        # 如果还是没有解析到，将整个响应作为一个命题
        if not propositions and len(response) > 20:
            propositions = [response]

        return propositions

    def _propositions_to_chunks(
        self,
        doc: SourceDocument,
        propositions: List[Proposition]
    ) -> List[Chunk]:
        """
        将命题列表转换为 Chunks

        策略：
        - 如果命题数量超过 max_propositions_per_chunk，进行组合
        - 否则每个命题一个 Chunk
        """
        chunks = []

        if not propositions:
            return chunks

        # 如果命题太多，分批组合
        if len(propositions) > self.max_propositions_per_chunk:
            # 计算需要的 chunk 数量
            num_chunks = (len(propositions) + self.max_propositions_per_chunk - 1) // self.max_propositions_per_chunk
            chunk_size = len(propositions) // num_chunks

            for i in range(num_chunks):
                start = i * chunk_size
                end = start + chunk_size if i < num_chunks - 1 else len(propositions)

                batch_props = propositions[start:end]
                combined_text = "\n".join(p.text for p in batch_props)

                chunk = Chunk(
                    chunk_id=self._generate_chunk_id(doc.source, doc.kind),
                    source=doc.source,
                    file=doc.file,
                    kind=doc.kind,
                    section=f"{doc.section}_proposition_{i:04d}",
                    chunk_strategy="proposition",
                    chunk_index=i,
                    text=combined_text,
                    token_count=self._estimate_tokens(combined_text),
                    metadata={
                        "doc_metadata": doc.metadata,
                        "proposition_count": len(batch_props),
                        "proposition_indices": [p.index for p in batch_props],
                        "original_paragraphs": list(set(p.source_text[:100] for p in batch_props)),
                    }
                )
                chunks.append(chunk)
        else:
            # 每个命题一个 chunk
            for i, prop in enumerate(propositions):
                chunk = Chunk(
                    chunk_id=self._generate_chunk_id(doc.source, doc.kind),
                    source=doc.source,
                    file=doc.file,
                    kind=doc.kind,
                    section=f"{doc.section}_proposition_{i:04d}",
                    chunk_strategy="proposition",
                    chunk_index=i,
                    text=prop.text,
                    token_count=self._estimate_tokens(prop.text),
                    metadata={
                        "doc_metadata": doc.metadata,
                        "original_paragraph": prop.source_text,
                        "paragraph_index": prop.index,
                    }
                )
                chunks.append(chunk)

        return chunks

    def _get_cache_key(self, text: str) -> str:
        """生成缓存键（全文哈希：前 200 字符相同的不同段落不得共享命题缓存）"""
        key = hashlib.md5(text.encode()).hexdigest()[:16]
        return key

    def _load_from_cache(self, cache_key: str) -> Optional[List[str]]:
        """从缓存加载"""
        cache_file = self._cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data.get("propositions")
            except Exception:
                pass
        return None

    def _save_to_cache(self, cache_key: str, propositions: List[str]) -> None:
        """保存到缓存"""
        cache_file = self._cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({"propositions": propositions}, f, ensure_ascii=False)
        except Exception as e:
            logger.debug(f"保存缓存失败: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self._stats.copy()

    def _generate_report(self, chunks: List[Chunk]) -> Any:
        """生成报告（扩展基类方法）"""
        report = super()._generate_report(chunks)
        # 添加命题特有统计
        if hasattr(report, 'params') and isinstance(report.params, dict):
            report.params["proposition_stats"] = self._stats
        return report
