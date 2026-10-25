import asyncio
import time
from contextlib import aclosing

from ..core.errors import GatewayError, normalize
from ..core.limits import ModelRateLimiter
from ..core.retry import backoff_seconds
from ..core.runtime import RequestContext
from ..models import LLMResponse, Usage
from .observability import TraceRecorder
from .prompts import build_messages
from .structured import parse_response


class Gateway:
    def __init__(self, models, *, backoff=0.25, trace_limit=1000):
        self.models = models
        self.limiter = ModelRateLimiter()
        self.recorder = TraceRecorder(trace_limit)
        self.traces = self.recorder.traces
        self.backoff = backoff

    async def timed_events(self, iterator, seconds):
        deadline = time.monotonic() + seconds
        async with aclosing(iterator):
            while True:
                try:
                    event = await asyncio.wait_for(anext(iterator), max(0, deadline - time.monotonic()))
                except StopAsyncIteration:
                    return
                yield event

    async def execute(self, request, ctx, messages):
        """A single event pipeline handles both complete and streaming requests."""
        if request.model not in self.models:
            raise GatewayError("unknown_model", "Model is not allowed", 400)
        candidates = [request.model]
        fallback = self.models[request.model].fallback
        if fallback and fallback != request.model:
            candidates.append(fallback)
        for name in candidates:
            config = self.models[name]
            if config.provider is None:
                raise GatewayError("gateway_misconfigured", "Provider credentials are not configured", 503)
            for retry in range(4):
                self.limiter.acquire(name, config.rpm)
                attempt = dict(model=name, number=len(ctx.attempts) + 1, status="started", usage=Usage().model_dump())
                ctx.attempts.append(attempt)
                started = time.perf_counter()
                emitted = False
                ctx.usage = Usage()
                try:
                    if request.stream:
                        async with aclosing(self.timed_events(config.provider.stream(config.upstream, messages, request.timeout_seconds), request.timeout_seconds)) as events:
                            async for event in events:
                                if event.usage is not None:
                                    attempt["usage"] = event.usage.model_dump()
                                    ctx.usage = event.usage
                                if event.delta:
                                    emitted = True
                                    ctx.actual_model = name
                                    if ctx.first_token_ms is None:
                                        ctx.first_token_ms = ctx.elapsed()
                                    yield {"type": "content.delta", "request_id": ctx.request_id, "delta": event.delta}
                    else:
                        result = await asyncio.wait_for(config.provider.complete(config.upstream, messages, request.response_format, request.timeout_seconds), request.timeout_seconds)
                        ctx.usage = result.usage
                        attempt["usage"] = result.usage.model_dump()
                        parsed = parse_response(result.content, request.response_format)
                    ctx.actual_model = name
                    attempt["status"] = "success"
                    attempt["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
                    if request.stream:
                        yield {"type": "response.completed", "request_id": ctx.request_id, "model": name,
                               "usage": ctx.usage.model_dump(), "first_token_latency_ms": ctx.first_token_ms}
                    else:
                        yield LLMResponse(request_id=ctx.request_id, model=name, content=result.content, parsed=parsed,
                                          usage=ctx.usage, latency_ms=ctx.elapsed(), attempts=len(ctx.attempts))
                    return
                except (asyncio.CancelledError, GeneratorExit):
                    if attempt["status"] != "success":
                        attempt.update(status="failed", error_code="cancelled")
                    raise
                except Exception as exc:
                    error = normalize(exc)
                    attempt.update(status="failed", error_code=error.code)
                    attempt["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
                    if emitted or not error.retryable:
                        raise error from None
                    if retry == 3:
                        break
                    delay = backoff_seconds(self.backoff, retry)
                    await asyncio.sleep(max(delay, error.retry_after or 0))
                finally:
                    attempt.setdefault("latency_ms", round((time.perf_counter() - started) * 1000, 3))
            ctx.usage = Usage()
        raise GatewayError("model_unavailable", "Primary and backup attempts exhausted")

    async def complete(self, request):
        ctx = RequestContext()
        code = None
        try:
            messages = build_messages(request)
            async with asyncio.timeout(request.deadline_seconds):
                async with aclosing(self.execute(request, ctx, messages)) as results:
                    return await anext(results)
        except TimeoutError:
            code = "deadline_exceeded"
            raise GatewayError(code, "Overall request deadline exceeded", 504) from None
        except GatewayError as exc:
            code = exc.code
            exc.request_id = ctx.request_id
            raise
        except asyncio.CancelledError:
            code = "cancelled"
            raise
        finally:
            self.recorder.record(request, ctx, code)

    async def start_stream(self, request):
        ctx = RequestContext()
        iterator = self.stream(request, ctx)
        # Prefetch so local 429/configuration errors can retain HTTP status.
        first = await anext(iterator)
        return first, iterator

    async def stream(self, request, ctx):
        code = None
        sent = False
        try:
            messages = build_messages(request)
            async with aclosing(self.timed_events(self.execute(request, ctx, messages), request.deadline_seconds)) as events:
                async for event in events:
                    sent = True
                    yield event
        except (GatewayError, TimeoutError) as exc:
            error = exc if isinstance(exc, GatewayError) else GatewayError("deadline_exceeded", "Overall request deadline exceeded", 504)
            code = error.code
            if not sent:
                error.request_id = ctx.request_id
                raise error from None
            yield {"type": "response.failed", "request_id": ctx.request_id, "error": {"code": error.code, "message": error.message}}
        except (asyncio.CancelledError, GeneratorExit):
            code = "cancelled"
            raise
        finally:
            self.recorder.record(request, ctx, code)
