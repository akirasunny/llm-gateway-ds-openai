import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from ..models import Usage
from ..providers import Provider


@dataclass
class Model:
    upstream: str
    provider: Provider | None
    rpm: int = 60
    fallback: str | None = None


@dataclass
class RequestContext:
    request_id: str = field(default_factory=lambda: str(uuid4()))
    started: float = field(default_factory=time.perf_counter)
    attempts: list = field(default_factory=list)
    actual_model: str | None = None
    first_token_ms: float | None = None
    usage: Usage = field(default_factory=Usage)

    def elapsed(self):
        return round((time.perf_counter() - self.started) * 1000, 3)


def now_utc():
    return datetime.now(timezone.utc).isoformat()
