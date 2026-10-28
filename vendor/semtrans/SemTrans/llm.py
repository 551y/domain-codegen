"""Small adapters for the LLM clients used by SemTrans."""

from __future__ import annotations

from typing import Any


def sanitize_code_output(raw_output: str) -> str:
    """Remove surrounding Markdown fences from generated source code."""
    lines = raw_output.splitlines()
    while lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    while lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def create_model_completion(
    client: Any,
    model_type: str,
    system_prompt: str,
    user_prompt: str,
) -> Any:
    """Create a completion through an OpenAI-compatible or Anthropic client."""
    if client is None:
        raise EnvironmentError("The selected LLM client is not configured.")

    # Some OpenAI-compatible deployments expose Qwen/Kimi/Claude-named models
    # through the same non-streaming Chat Completions endpoint. Prefer the
    # concrete client's declared transport capability over guessing protocol
    # behavior from a model ID substring.
    if getattr(client, "force_non_streaming", False):
        return client.chat.completions.create(
            model=model_type,
            temperature=0.3,
            # DenseNet/ShuffleNet contain nested real model definitions.  An
            # 8k response budget can terminate before the closing class body,
            # which is indistinguishable from a conversion failure.  Keep a
            # bounded but sufficient code budget for the demo route.
            max_tokens=16384,
            stream=False,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

    model_name = model_type.lower()
    if "claude" in model_name:
        return client.messages.create(
            model=model_type,
            temperature=0.3,
            max_tokens=40960,
            stream=True,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

    request = {
        "model": model_type,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if "kimi" in model_name:
        request.update(max_tokens=81920, stream=True)
    elif "qwen" in model_name:
        request.update(extra_body={"enable_thinking": True}, stream=True)
    return client.chat.completions.create(**request)


def process_model_response(model_type: str, response: Any, logger: Any) -> str:
    """Extract generated text from supported streaming and non-streaming APIs."""
    model_name = model_type.lower()
    choices = getattr(response, "choices", None)
    message = getattr(choices[0], "message", None) if choices else None
    if message is not None:
        content = getattr(message, "content", "") or ""
    elif "claude" in model_name:
        chunks = []
        for event in response:
            if (
                getattr(event, "type", None) == "content_block_delta"
                and getattr(getattr(event, "delta", None), "type", None)
                == "text_delta"
            ):
                chunks.append(event.delta.text)
        content = "".join(chunks)
    elif "qwen" in model_name or "kimi" in model_name:
        chunks = []
        for chunk in response:
            choices = getattr(chunk, "choices", None)
            if not choices:
                usage = getattr(chunk, "usage", None)
                if usage is not None and logger is not None:
                    logger.debug("LLM usage: %s", usage)
                continue
            text = getattr(choices[0].delta, "content", None)
            if text:
                chunks.append(text)
        content = "".join(chunks)
    else:
        if not choices:
            raise ValueError("The LLM response contains no choices.")
        content = getattr(choices[0].message, "content", "") or ""

    return sanitize_code_output(content)
