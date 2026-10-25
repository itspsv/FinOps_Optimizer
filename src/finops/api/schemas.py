"""OpenAI-compatible request schemas.

We validate only the fields the proxy relies on and pass everything else
(temperature, tools, response_format, ...) through untouched.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    # The client's requested model. The proxy picks the real upstream model,
    # so this is recorded but not forwarded as-is.
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False

    def upstream_payload(self) -> dict[str, Any]:
        """Fields the client actually sent, minus the ones the proxy controls."""
        return self.model_dump(mode="json", exclude_unset=True, exclude={"model", "stream"})
