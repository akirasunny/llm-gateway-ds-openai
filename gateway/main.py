import logging
import os
import secrets
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .api.routes import router
from .core.errors import GatewayError
from .core.runtime import Model, RequestContext
from .providers import ChatCompletionsProvider, ResponsesProvider
from .services.gateway import Gateway


def configured_gateway():
    primary = os.getenv("PRIMARY_MODEL", "deepseek-v4-flash")
    backup = os.getenv("BACKUP_MODEL", "gpt-6-astra")

    def provider(cls, key, url):
        secret = os.getenv(key)
        return cls(secret, url) if secret else None

    models = {
        primary: Model(
            primary,
            provider(ChatCompletionsProvider, "DEEPSEEK_API_KEY", os.getenv("PRIMARY_BASE_URL", "https://api.deepseek.com")),
            int(os.getenv("PRIMARY_RPM", "60")),
            backup,
        ),
        backup: Model(
            backup,
            provider(ResponsesProvider, "OPENAI_API_KEY", os.getenv("BACKUP_BASE_URL", "https://api.openai.com/v1")),
            int(os.getenv("BACKUP_RPM", "60")),
        ),
    }
    if primary == backup or any(model.rpm < 1 for model in models.values()):
        raise ValueError("Models must be distinct and RPM limits positive")
    return Gateway(models)


def create_app(gateway=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.gateway = gateway or configured_gateway()
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("openai").setLevel(logging.CRITICAL)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        try:
            yield
        finally:
            for model in app.state.gateway.models.values():
                if model.provider:
                    await model.provider.close()

    async def authorize(request: Request):
        expected = os.getenv("GATEWAY_API_KEY")
        if expected and not secrets.compare_digest(
            request.headers.get("authorization", ""), "Bearer " + expected
        ):
            raise GatewayError("unauthorized", "Gateway authentication required", 401)

    app = FastAPI(title="LLM Gateway", lifespan=lifespan, dependencies=[Depends(authorize)])

    @app.middleware("http")
    async def measure_request(request, call_next):
        request.state.started = time.perf_counter()
        return await call_next(request)

    @app.exception_handler(GatewayError)
    async def gateway_error(_, exc):
        headers = {}
        if exc.retry_after is not None:
            headers["Retry-After"] = str(max(1, int(exc.retry_after)))
        return JSONResponse(
            status_code=exc.status,
            content={"error": {"code": exc.code, "message": exc.message}, "request_id": exc.request_id},
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, _):
        context = RequestContext()
        trace = {
            "request_id": context.request_id,
            "status": "failed",
            "error_code": "invalid_request",
            "attempts": [],
            "usage": {},
            "latency_ms": round((time.perf_counter() - request.state.started) * 1000, 3),
            "first_token_latency_ms": None,
        }
        request.app.state.gateway.recorder.traces.append(trace)
        return JSONResponse(
            status_code=422,
            content={
                "error": {"code": "invalid_request", "message": "Request validation failed"},
                "request_id": context.request_id,
            },
        )

    app.include_router(router)
    return app


app = create_app()
