from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from jsonschema import Draft202012Validator, SchemaError


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Message(StrictModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=20_000)


class PromptSelection(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    version: str = Field(min_length=1, max_length=50)
    variables: dict[str, str] = Field(default_factory=dict)


class SchemaSpec(StrictModel):
    name: str = "gateway_response"
    schema_: dict[str, Any] = Field(alias="schema")
    strict: bool = True

    @model_validator(mode="after")
    def valid_schema(self):
        try:
            Draft202012Validator.check_schema(self.schema_)
        except SchemaError:
            raise ValueError("Invalid JSON Schema") from None
        def check_references(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in {"$ref", "$dynamicRef"} and isinstance(item, str) and not item.startswith("#"):
                        raise ValueError("Only local JSON Schema references are supported")
                    check_references(item)
            elif isinstance(value, list):
                for item in value:
                    check_references(item)
        check_references(self.schema_)
        if self.schema_.get("type") != "object":
            raise ValueError("Root JSON Schema must have type object")
        return self


class ResponseFormat(StrictModel):
    type: Literal["json_object", "json_schema"]
    json_schema: SchemaSpec | None = None

    @model_validator(mode="after")
    def combination(self):
        if (self.type == "json_schema") != (self.json_schema is not None):
            raise ValueError("json_schema is required only for type json_schema")
        return self


class LLMRequest(StrictModel):
    model: str = Field(min_length=1, max_length=100)
    messages: list[Message] = Field(min_length=1, max_length=100)
    stream: bool = False
    response_format: ResponseFormat | None = None
    prompt: PromptSelection | None = None
    timeout_seconds: float = Field(default=30, gt=0, le=120)
    deadline_seconds: float = Field(default=120, gt=0, le=600)

    @model_validator(mode="after")
    def supported(self):
        if self.stream and self.response_format:
            raise ValueError("Streaming and structured output cannot be combined")
        return self


class Usage(StrictModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    reasoning_output_tokens: int | None = Field(default=None, ge=0)


class LLMResponse(StrictModel):
    request_id: str
    model: str
    content: str
    parsed: dict[str, Any] | None = None
    usage: Usage
    latency_ms: float
    first_token_latency_ms: float | None = None
    attempts: int
