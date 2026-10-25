import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ..models import LLMRequest

router = APIRouter()


@router.post("/v1/llm")
async def invoke(body: LLMRequest, request: Request):
    service = request.app.state.gateway
    if not body.stream:
        return await service.complete(body)
    first, iterator = await service.start_stream(body)

    async def sse():
        try:
            yield "data: " + json.dumps(first) + "\n\n"
            async for event in iterator:
                yield "data: " + json.dumps(event) + "\n\n"
        finally:
            await iterator.aclose()

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
        },
    )


@router.get("/v1/traces")
async def traces(request: Request):
    return list(request.app.state.gateway.recorder.traces)


@router.get("/healthz")
async def healthz():
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request):
    return {"status": "ready", "models": sorted(request.app.state.gateway.models)}
