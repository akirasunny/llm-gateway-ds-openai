import json

from jsonschema import Draft202012Validator, ValidationError

from ..core.errors import GatewayError


def parse_response(content, response_format):
    if response_format is None:
        return None
    try:
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError
        if response_format.json_schema:
            Draft202012Validator(response_format.json_schema.schema_).validate(parsed)
        return parsed
    except (ValueError, ValidationError):
        raise GatewayError(
            "invalid_structured_output",
            "Provider output is not a valid JSON object matching the requested format",
        ) from None
