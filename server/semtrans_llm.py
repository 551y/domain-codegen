"""Minimal OpenAI-compatible adapter used by SemTrans' existing LLM hooks."""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from urllib.parse import urlsplit
from types import SimpleNamespace
from typing import Any


class ModelServiceError(RuntimeError):
    pass


def _endpoint(base_url: str) -> str:
    value = base_url.strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
        raise ModelServiceError("模型服务 Base URL 必须是有效的 HTTPS API 地址")
    if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ModelServiceError("非本机模型服务必须使用 HTTPS")
    if value.endswith("/chat/completions"):
        return value
    # DeepSeek documents its OpenAI-compatible base as the host root; unlike
    # OpenAI-compatible gateways it does not require a /v1 prefix.
    if parsed.hostname and parsed.hostname.lower() == "api.deepseek.com" and parsed.path in {"", "/"}:
        return value + "/chat/completions"
    if value.endswith("/v1") or value.endswith("/v4"):
        return value + "/chat/completions"
    return value + "/v1/chat/completions"


def _responses_endpoint(base_url: str) -> str:
    _endpoint(base_url)  # validate scheme, credentials and non-local HTTP first
    value = base_url.strip().rstrip("/")
    if value.endswith("/responses"):
        return value
    if value.endswith("/v1"):
        return value + "/responses"
    return value + "/v1/responses"


class OpenAICompatibleClient:
    """Small SDK-shaped client so the vendored SemTrans API stays in use."""

    force_non_streaming = True

    def __init__(self, *, base_url: str, api_key: str, timeout: float = 75.0, provider: str = ""):
        if not base_url or not api_key:
            raise ModelServiceError("SemTrans 模型服务地址或 API Key 尚未配置")
        self.base_url = base_url
        self.api_key = api_key
        self.provider = provider
        self.timeout = timeout
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, model: str, messages: list, temperature: float = 0.2,
                max_tokens: int = 16384, stream: bool = False, **kwargs: Any):
        if stream:
            raise ModelServiceError("当前演示适配要求模型服务使用非流式 Chat Completions")
        use_responses = self.provider == "codex"
        if use_responses:
            endpoint = _responses_endpoint(self.base_url)
            payload_data = {"model": model, "input": messages, "max_output_tokens": max_tokens, **kwargs}
        else:
            endpoint = _endpoint(self.base_url)
            payload_data = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, **kwargs}
        payload = json.dumps(payload_data, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            endpoint, data=payload, method="POST",
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read(8 * 1024 * 1024 + 1)
            if len(body) > 8 * 1024 * 1024:
                raise ModelServiceError("模型服务响应超过 8 MiB 上限")
            data = json.loads(body.decode("utf-8"))
            finish_reason = None
            if use_responses:
                content = data.get("output_text", "")
                if not content:
                    content = "".join(
                        part.get("text", "")
                        for item in data.get("output", [])
                        if isinstance(item, dict)
                        for part in item.get("content", [])
                        if isinstance(part, dict) and isinstance(part.get("text"), str)
                    )
            else:
                choice = data["choices"][0]
                finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
                content = choice["message"]["content"]
            if isinstance(content, list):
                content = "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
            if not isinstance(content, str) or not content.strip():
                if finish_reason == "length":
                    raise ModelServiceError("模型服务输出达到 token 上限，尚未返回 assistant 内容；请提高输出上限后重试")
                raise ModelServiceError("模型服务返回空的 assistant 内容")
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
        except urllib.error.HTTPError as exc:
            status = exc.code
            if status == 401 or status == 403:
                message = "模型服务鉴权失败，请检查 API Key 和账户权限"
            elif status == 404:
                message = "模型服务接口地址或模型不存在，请核对 Base URL 和模型 ID"
            elif status == 429:
                message = "模型服务限流或额度不足，请稍后重试并检查账户额度"
            else:
                message = f"模型服务返回 HTTP {status}"
            raise ModelServiceError(message) from None
        except (TimeoutError, socket.timeout):
            raise ModelServiceError("连接模型服务超时") from None
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", None)
            if isinstance(reason, TimeoutError):
                raise ModelServiceError("连接模型服务超时") from None
            raise ModelServiceError("无法连接模型服务，请检查 Base URL 和网络") from None
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            raise ModelServiceError("模型服务返回格式不符合 OpenAI Chat Completions 协议") from None

    def test_connection(self, model: str) -> None:
        self._create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            temperature=0,
            # GLM reasoning models may spend a few dozen tokens in
            # ``reasoning_content`` before emitting the short answer.  Eight
            # tokens can therefore finish with ``finish_reason=length`` and
            # an empty assistant content even when the key and endpoint are
            # valid.  Keep this probe small but leave room for that reasoning.
            max_tokens=512,
        )
