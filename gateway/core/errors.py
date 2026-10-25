from openai import APIConnectionError, APIStatusError


class GatewayError(Exception):
    def __init__(self, code, message, status=502, retryable=False, retry_after=None):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status
        self.retryable, self.retry_after = retryable, retry_after
        self.request_id = None


def normalize(exc):
    if isinstance(exc, GatewayError):
        return exc
    if isinstance(exc, (APIConnectionError, TimeoutError, ConnectionError)):
        return GatewayError("upstream_unavailable", "Provider connection failed or timed out", retryable=True)
    if isinstance(exc, APIStatusError):
        retryable = exc.status_code in {408, 429} or exc.status_code >= 500
        delay = None
        try:
            delay = max(0, float(exc.response.headers.get("retry-after", "")))
        except (ValueError, TypeError):
            pass
        code = "upstream_rate_limited" if exc.status_code == 429 else "upstream_error"
        return GatewayError(code, "Provider rejected the request", retryable=retryable, retry_after=delay)
    return GatewayError("upstream_error", "Provider returned an invalid response")
