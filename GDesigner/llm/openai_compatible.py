import asyncio
import dataclasses
from typing import Any, Dict, List, Optional, Union

import aiohttp

from GDesigner.llm.format import Message
from GDesigner.llm.price import cost_count


def normalize_messages(messages: Union[str, List[Union[Message, Dict[str, str]]]]) -> List[Dict[str, str]]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]

    normalized = []
    for message in messages:
        if dataclasses.is_dataclass(message):
            normalized.append(dataclasses.asdict(message))
        else:
            normalized.append(dict(message))
    return normalized


def completion_text(content: Union[List[str], str]) -> str:
    if isinstance(content, list):
        return "".join(content)
    return content


def prompt_text(messages: List[Dict[str, str]]) -> str:
    return "".join(item.get("content", "") for item in messages)


def usage_tokens(response_data: Dict[str, Any]) -> tuple[Optional[int], Optional[int]]:
    usage = response_data.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    return prompt_tokens, completion_tokens


async def chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: Union[str, List[Union[Message, Dict[str, str]]]],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    num_comps: Optional[int] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    zero_cost: bool = False,
) -> Union[List[str], str]:
    base = base_url.rstrip("/")
    request_url = f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"
    normalized_messages = normalize_messages(messages)
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_key,
    }
    data: Dict[str, Any] = {
        "model": model,
        "messages": normalized_messages,
    }
    if max_tokens is not None:
        data["max_tokens"] = max_tokens
    if temperature is not None:
        data["temperature"] = temperature
    if num_comps is not None:
        data["n"] = num_comps
    if extra_body:
        data.update(extra_body)

    async with aiohttp.ClientSession() as session:
        async with session.post(request_url, headers=headers, json=data) as response:
            response_data = await response.json()
            if "error" in response_data:
                raise Exception(f"API error {response.status}: {response_data['error']}")
            choices = response_data["choices"]
            if num_comps and num_comps > 1:
                content = [choice["message"]["content"] for choice in choices]
            else:
                content = choices[0]["message"]["content"]

    prompt_tokens, completion_tokens = usage_tokens(response_data)
    cost_count(
        prompt_text(normalized_messages),
        completion_text(content),
        model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        zero_cost=zero_cost,
    )
    return content


def run_chat_completion_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("Synchronous LLM generation cannot run inside an active event loop.")
