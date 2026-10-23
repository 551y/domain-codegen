"""
LLM 客户端封装模块

支持两类供应商：
    - zai（默认）：智谱 AI，走 zai-sdk 原生协议
    - openai：OpenAI 兼容协议（DeepSeek、硅基流动、Moonshot 等），走 HTTP

Chat 与 Embedding 的供应商可分别配置（DeepSeek 无 embedding 接口，
可将 chat 指向 DeepSeek、embedding 保持智谱）。
"""

from typing import List, Optional, Dict, Any

import json
import re
import threading

from zai import ZhipuAiClient

from herta.config import get_settings, Settings
from herta.constants import PROVIDER_OPENAI
from herta.logging import get_logger

logger = get_logger(__name__)

# demo 集成优化：线程本地流式输出回调。
# 上层（编排服务）在生成线程里 set_stream_sink(fn) 后，该线程内的 chat 调用
# 会以 fn("begin") / fn("delta", 增量文本) 实时上报生成过程；不设置则行为与原来一致。
_stream_local = threading.local()


def set_stream_sink(fn):
    """为当前线程设置流式输出回调 fn(kind, text)；传 None 关闭"""
    _stream_local.sink = fn


def _emit(kind: str, text: str = ""):
    sink = getattr(_stream_local, "sink", None)
    if sink:
        try:
            sink(kind, text)
        except Exception:
            pass


def _normalize_base_url(url: str) -> str:
    """归一化 OpenAI 兼容接口地址：无版本后缀时补 /v1，已有 /vN 路径则保持原样"""
    url = url.rstrip("/")
    if not re.search(r"/v\d+$", url):
        url += "/v1"
    return url


def _openai_post(base_url: str, path: str, api_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """向 OpenAI 兼容接口发起 POST 请求"""
    try:
        import httpx
    except ImportError as e:
        raise RuntimeError("使用 OpenAI 兼容供应商需要 httpx：pip install httpx") from e

    url = f"{_normalize_base_url(base_url)}{path}"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    with httpx.Client(timeout=180.0) as client:
        resp = client.post(url, json=payload, headers=headers)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json()


class BigModelClient:
    """
    统一 LLM 客户端封装

    自动读取配置，提供 chat 和 embed 方法（供应商对调用方透明）

    Example:
        >>> client = BigModelClient()
        >>> response = client.chat([{"role": "user", "content": "你好"}])
        >>> embeddings = client.embed_texts(["hello", "world"])
    """

    def __init__(self, settings: Optional[Settings] = None):
        """
        初始化客户端

        Args:
            settings: 配置对象，如果为 None 则自动从环境变量加载

        Raises:
            ValueError: 当 API 密钥未设置时
        """
        self.settings = settings or get_settings()
        # chat 与 embedding 使用独立的智谱客户端（二者密钥可能不同）
        self._zai_chat_client = None
        self._zai_embed_client = None
        logger.debug(
            f"LLM 客户端初始化: chat={self.settings.chat_provider}/{self.settings.chat_model}, "
            f"embedding={self.settings.embedding_provider}/{self.settings.embedding_model}"
        )

    # ------------------------------------------------------------------ chat

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.1,
        stream: bool = False,
    ) -> str:
        """
        调用聊天模型

        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}, ...]
            model: 模型名称，默认使用配置中的 chat_model
            temperature: 采样温度，默认 0.1（更确定性）
            stream: 兼容保留参数；内部始终流式请求并聚合全文返回
                （线程内设置 set_stream_sink 后可实时收到增量回调）

        Returns:
            模型生成的纯文本字符串

        Raises:
            ValueError: 当 messages 为空或格式错误时
            RuntimeError: 当 API 调用失败时
        """
        if not messages:
            raise ValueError("messages 不能为空列表")
        if not isinstance(messages, list):
            raise ValueError("messages 必须是列表类型")
        for msg in messages:
            if not isinstance(msg, dict):
                raise ValueError("messages 中的每项必须是字典类型")
            if "role" not in msg or "content" not in msg:
                raise ValueError('messages 中的每项必须包含 "role" 和 "content" 字段')

        model_name = model or self.settings.chat_model

        if self.settings.chat_provider == PROVIDER_OPENAI:
            return self._chat_openai(messages, model_name, temperature)
        return self._chat_zai(messages, model_name, temperature)

    def _chat_zai(self, messages, model_name, temperature) -> str:
        if not self.settings.effective_chat_key():
            raise RuntimeError("未配置对话密钥：请设置 ZAI_API_KEY 或 HERTA_CHAT_API_KEY")
        if self._zai_chat_client is None:
            self._zai_chat_client = ZhipuAiClient(api_key=self.settings.effective_chat_key())
        try:
            logger.debug(f"调用智谱 chat 接口，模型: {model_name}, 消息数: {len(messages)}")
            # 始终流式请求并聚合全文：设置了线程 sink 时可实时上报增量，否则等价于原同步行为
            response = self._zai_chat_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            _emit("begin")
            parts: List[str] = []
            for chunk in response:
                delta = None
                try:
                    if chunk.choices:
                        delta = chunk.choices[0].delta.content
                except Exception:
                    delta = None
                if delta:
                    parts.append(delta)
                    _emit("delta", delta)
            content = "".join(parts)
            if not content.strip():
                # 流式聚合为空：通常是 SSE 流被网络代理/防火墙截断（chunk 全丢），
                # 静默交付空代码会让前端结果区空白且难以排查，必须显式报错
                raise RuntimeError("模型流式返回为空（0 个增量）——SSE 流可能被网络代理/防火墙截断；"
                                   "请检查网络代理设置，或在顶栏设置中更换 Base URL/模型后重试")
            logger.debug(f"chat 调用成功，返回长度: {len(content)}")
            return content
        except Exception as e:
            logger.error(f"chat 调用失败: {e}")
            raise RuntimeError(f"chat API 调用失败: {e}") from e

    def _chat_openai(self, messages, model_name, temperature) -> str:
        if not self.settings.chat_base_url:
            raise RuntimeError(
                "chat_provider=openai 但未配置 HERTA_CHAT_BASE_URL（如 https://api.deepseek.com）"
            )
        key = self.settings.effective_chat_key()
        if not key:
            raise RuntimeError("chat_provider=openai 但未配置 HERTA_CHAT_API_KEY")
        try:
            import httpx
        except ImportError as e:
            raise RuntimeError("使用 OpenAI 兼容供应商需要 httpx：pip install httpx") from e
        url = f"{_normalize_base_url(self.settings.chat_base_url)}/chat/completions"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {"model": model_name, "messages": messages, "temperature": temperature, "stream": True}
        try:
            logger.debug(f"调用 OpenAI 兼容 chat 接口 {self.settings.chat_base_url}，模型: {model_name}")
            # SSE 流式读取并聚合全文（设置线程 sink 时实时上报增量）
            parts: List[str] = []
            with httpx.Client(timeout=180.0) as client:
                with client.stream("POST", url, json=payload, headers=headers) as resp:
                    if resp.status_code != 200:
                        resp.read()
                        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                    _emit("begin")
                    for line in resp.iter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            obj = json.loads(data_str)
                            choices = obj.get("choices") or []
                            delta = (choices[0].get("delta") or {}).get("content") if choices else None
                        except Exception:
                            continue
                        if delta:
                            parts.append(delta)
                            _emit("delta", delta)
            content = "".join(parts)
            if not content.strip():
                # 流式聚合为空：通常是 SSE 流被网络代理/防火墙截断（chunk 全丢），
                # 静默交付空代码会让前端结果区空白且难以排查，必须显式报错
                raise RuntimeError("模型流式返回为空（0 个增量）——SSE 流可能被网络代理/防火墙截断；"
                                   "请检查网络代理设置，或在顶栏设置中更换 Base URL/模型后重试")
            logger.debug(f"chat 调用成功，返回长度: {len(content)}")
            return content
        except Exception as e:
            logger.error(f"chat 调用失败: {e}")
            raise RuntimeError(f"chat API 调用失败: {e}") from e

    # ------------------------------------------------------------- embedding

    def embed_texts(
        self,
        texts: List[str],
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        batch_size: int = 64,
    ) -> List[List[float]]:
        """
        批量获取文本的 Embedding 向量

        支持自动分批处理，每批不超过 batch_size（默认64，适配智谱API限制）

        Args:
            texts: 文本列表，每项是一个字符串
            model: Embedding 模型名称，默认使用配置中的 embedding_model
            dimensions: 向量维度，默认使用配置中的 embedding_dimensions
            batch_size: 每批处理的文本数量，默认64

        Returns:
            Embedding 向量列表，每个向量是一个 float 列表

        Raises:
            ValueError: 当 texts 为空或包含空字符串时
            RuntimeError: 当 API 调用失败时
        """
        if not texts:
            raise ValueError("texts 不能为空列表")
        if not isinstance(texts, list):
            raise ValueError("texts 必须是列表类型")
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(f"texts[{i}] 必须是字符串类型")
            if not text.strip():
                raise ValueError(f"texts[{i}] 不能为空或空白字符串")

        model_name = model or self.settings.embedding_model
        dims = dimensions or self.settings.embedding_dimensions
        total = len(texts)
        all_embeddings: List[List[float]] = []

        logger.info(
            f"开始生成 embeddings: {total} 个文本，批大小: {batch_size}，"
            f"供应商: {self.settings.embedding_provider}/{model_name}"
        )

        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            batch_num = i // batch_size + 1
            try:
                if self.settings.embedding_provider == PROVIDER_OPENAI:
                    batch_embeddings = self._embed_openai(batch, model_name, dims)
                else:
                    batch_embeddings = self._embed_zai(batch, model_name, dims)
                all_embeddings.extend(batch_embeddings)
                logger.debug(f"第 {batch_num} 批完成: {len(batch_embeddings)} 个向量")
            except Exception as e:
                logger.error(f"第 {batch_num} 批 embeddings 调用失败: {e}")
                raise RuntimeError(f"embeddings API 调用失败 (batch {batch_num}): {e}") from e

        logger.info(f"embeddings 生成完成: 共 {len(all_embeddings)} 个向量")
        return all_embeddings

    def _embed_zai(self, batch: List[str], model_name: str, dims: int) -> List[List[float]]:
        if not self.settings.effective_embedding_key():
            raise RuntimeError("未配置向量密钥：请设置 ZAI_API_KEY 或 HERTA_EMBEDDING_API_KEY")
        if self._zai_embed_client is None:
            self._zai_embed_client = ZhipuAiClient(api_key=self.settings.effective_embedding_key())
        response = self._zai_embed_client.embeddings.create(model=model_name, input=batch, dimensions=dims)
        if hasattr(response, "data") and len(response.data) > 0:
            return [item.embedding for item in response.data]
        raise RuntimeError("API 返回格式异常，未找到 data 字段")

    def _embed_openai(self, batch: List[str], model_name: str, dims: int) -> List[List[float]]:
        if not self.settings.embedding_base_url:
            raise RuntimeError("embedding_provider=openai 但未配置 HERTA_EMBEDDING_BASE_URL")
        key = self.settings.effective_embedding_key()
        if not key:
            raise RuntimeError("embedding_provider=openai 但未配置 HERTA_EMBEDDING_API_KEY")
        data = _openai_post(
            self.settings.embedding_base_url,
            "/embeddings",
            key,
            {"model": model_name, "input": batch},
        )
        return [item["embedding"] for item in data["data"]]

    def embed_query(
        self,
        text: str,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
    ) -> List[float]:
        """
        获取单个查询文本的 Embedding 向量

        Args:
            text: 查询文本字符串
            model: Embedding 模型名称
            dimensions: 向量维度

        Returns:
            Embedding 向量（float 列表）
        """
        if not isinstance(text, str):
            raise ValueError("text 必须是字符串类型")
        if not text.strip():
            raise ValueError("text 不能为空或空白字符串")
        embeddings = self.embed_texts([text], model=model, dimensions=dimensions)
        return embeddings[0]
