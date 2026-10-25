import json
import logging
from collections import deque

from ..core.runtime import now_utc
from ..models import Usage

logger = logging.getLogger("llm_gateway")


class TraceRecorder:
    def __init__(self, limit=1000):
        self.traces = deque(maxlen=limit)

    def record(self, request, context, code=None):
        trace = {
            "request_id": context.request_id,
            "timestamp": now_utc(),
            "requested_model": request.model if request.model else None,
            "actual_model": context.actual_model,
            "latency_ms": context.elapsed(),
            "first_token_latency_ms": context.first_token_ms,
            "usage": context.usage.model_dump(),
            "attempts": context.attempts,
            "status": "failed" if code else "success",
            "error_code": code,
        }
        if request.prompt:
            trace.update(prompt_name=request.prompt.name, prompt_version=request.prompt.version)
        trace["known_usage_totals"] = {
            key: sum(a["usage"][key] for a in context.attempts if a["usage"][key] is not None)
            if any(a["usage"][key] is not None for a in context.attempts) else None
            for key in Usage.model_fields
        }
        trace["usage_complete"] = bool(context.attempts) and all(
            a["usage"]["input_tokens"] is not None and a["usage"]["output_tokens"] is not None
            for a in context.attempts
        )
        self.traces.append(trace)
        logger.info("llm_call_trace=%s", json.dumps(trace))
