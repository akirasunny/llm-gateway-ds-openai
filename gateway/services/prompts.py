from string import Template

from ..core.errors import GatewayError
from ..models import Message

TEMPLATES = {
    ("knowledge_decision", "v1"): "You are ${product_name}'s knowledge assistant. Search when evidence is insufficient; never invent policies."
}


def build_messages(request):
    messages = list(request.messages)
    if request.prompt is None:
        return messages
    template = TEMPLATES.get((request.prompt.name, request.prompt.version))
    if template is None:
        raise GatewayError("unknown_prompt", "Prompt name/version not found", 400)
    try:
        rendered = Template(template).substitute(request.prompt.variables)
    except (KeyError, ValueError):
        raise GatewayError("invalid_prompt_variables", "Prompt variables are missing or invalid", 400) from None
    messages.insert(0, Message(role="system", content=rendered))
    return messages
