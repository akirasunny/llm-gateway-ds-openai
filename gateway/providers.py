"""Translate two wire protocols into content/usage events; SDK retries are off."""
from dataclasses import dataclass, field
from collections.abc import AsyncIterator
from typing import Protocol
from openai import AsyncOpenAI
from .models import Message, ResponseFormat, Usage


@dataclass
class Event:
    delta: str = ""
    usage: Usage | None = None


@dataclass
class Result:
    content: str
    usage: Usage = field(default_factory=Usage)


class Provider(Protocol):
    async def complete(self, model: str, messages: list[Message], fmt: ResponseFormat | None, timeout: float) -> Result: ...
    def stream(self, model: str, messages: list[Message], timeout: float) -> AsyncIterator[Event]: ...
    async def close(self) -> None: ...


def chat_usage(u) -> Usage:
    if u is None:
        return Usage()
    return Usage(input_tokens=u.prompt_tokens, output_tokens=u.completion_tokens,
                 cached_input_tokens=getattr(u, "prompt_cache_hit_tokens", None) if getattr(u, "prompt_cache_hit_tokens", None) is not None else getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", None),
                 reasoning_output_tokens=getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", None))


def responses_usage(u) -> Usage:
    if u is None:
        return Usage()
    return Usage(input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                 cached_input_tokens=getattr(getattr(u, "input_tokens_details", None), "cached_tokens", None),
                 reasoning_output_tokens=getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", None))


class ChatCompletionsProvider:
    def __init__(self, key: str, base_url: str):
        self.client = AsyncOpenAI(api_key=key, base_url=base_url, max_retries=0)

    async def close(self):
        await self.client.close()

    async def complete(self, model, messages, fmt, timeout):
        data = [m.model_dump() for m in messages]
        kwargs = {}
        if fmt:
            # DeepSeek JSON mode; the service enforces the optional schema locally.
            instruction = "Return only a valid JSON object."
            if fmt.json_schema:
                import json
                instruction += " Follow this JSON Schema: " + json.dumps(fmt.json_schema.schema_)
            data = [{"role": "system", "content": instruction}, *data]
            kwargs["response_format"] = {"type": "json_object"}
        response = await self.client.chat.completions.create(model=model, messages=data, timeout=timeout, **kwargs)
        return Result(response.choices[0].message.content or "", chat_usage(response.usage))

    async def stream(self, model, messages, timeout):
        response = await self.client.chat.completions.create(
            model=model, messages=[m.model_dump() for m in messages], timeout=timeout,
            stream=True, stream_options={"include_usage": True})
        finished = False
        async with response:
            async for chunk in response:
                if chunk.choices and chunk.choices[0].finish_reason is not None:
                    finished = True
                if chunk.usage is not None:
                    yield Event(usage=chat_usage(chunk.usage))
                if chunk.choices and chunk.choices[0].delta.content:
                    yield Event(delta=chunk.choices[0].delta.content)
        if not finished:
            raise ConnectionError("Stream ended before completion")


class ResponsesProvider:
    def __init__(self, key: str, base_url: str):
        self.client = AsyncOpenAI(api_key=key, base_url=base_url, max_retries=0)

    async def close(self):
        await self.client.close()

    def payload(self, model, messages, timeout):
        return dict(model=model, input=[m.model_dump() for m in messages], timeout=timeout, store=False)

    async def complete(self, model, messages, fmt, timeout):
        kwargs = self.payload(model, messages, timeout)
        if fmt:
            if fmt.json_schema:
                spec = fmt.json_schema
                kwargs["text"] = {"format": {"type": "json_schema", "name": spec.name, "schema": spec.schema_, "strict": spec.strict}}
            else:
                kwargs["text"] = {"format": {"type": "json_object"}}
                kwargs["input"].insert(0, {"role": "system", "content": "Return only a valid JSON object."})
        response = await self.client.responses.create(**kwargs)
        if response.status != "completed":
            raise ValueError("Incomplete upstream response")
        return Result(response.output_text, responses_usage(response.usage))

    async def stream(self, model, messages, timeout):
        response = await self.client.responses.create(**self.payload(model, messages, timeout), stream=True)
        completed = False
        async with response:
            async for event in response:
                if event.type == "response.output_text.delta":
                    yield Event(delta=event.delta)
                elif event.type == "response.completed":
                    completed = True
                    yield Event(usage=responses_usage(event.response.usage))
                elif event.type in {"error", "response.failed", "response.incomplete"}:
                    raise ValueError("Upstream stream failed")
        if not completed:
            raise ConnectionError("Stream ended before completion")
