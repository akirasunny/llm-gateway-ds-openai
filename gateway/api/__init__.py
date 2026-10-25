"""HTTP API routes for the gateway."""


def __getattr__(name):
    # Compatibility for the former gateway.api:app entrypoint.
    if name in {"app", "create_app"}:
        from ..main import app, create_app
        return {"app": app, "create_app": create_app}[name]
    raise AttributeError(name)
