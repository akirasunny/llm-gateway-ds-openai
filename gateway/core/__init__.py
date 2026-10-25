"""Runtime policies and shared gateway primitives."""


def __getattr__(name):
    # Compatibility for callers that imported these symbols from the former core.py.
    if name == "Gateway":
        from ..services.gateway import Gateway
        return Gateway
    if name == "GatewayError":
        from .errors import GatewayError
        return GatewayError
    if name == "Model":
        from .runtime import Model
        return Model
    raise AttributeError(name)
