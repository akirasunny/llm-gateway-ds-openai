import math
import time
from collections import deque

from .errors import GatewayError


class ModelRateLimiter:
    """Single-process rolling request limits, tracked independently by model."""

    def __init__(self):
        self.windows = {}

    def acquire(self, model, limit):
        now = time.monotonic()
        window = self.windows.setdefault(model, deque())
        while window and window[0] <= now - 60:
            window.popleft()
        if len(window) >= limit:
            raise GatewayError(
                "rate_limit_exceeded",
                "Model request limit exceeded",
                429,
                retry_after=max(1, math.ceil(60 - (now - window[0]))),
            )
        window.append(now)
