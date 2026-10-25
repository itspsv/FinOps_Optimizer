from collections.abc import AsyncIterator
from typing import Any, Protocol


class ProviderError(Exception):
    """An upstream call failed.

    `status_code` is what the proxy should return to its client, and `body`
    is an OpenAI-shaped error payload.
    """

    def __init__(self, status_code: int, body: dict[str, Any]):
        super().__init__(f"provider error {status_code}: {body}")
        self.status_code = status_code
        self.body = body

    @classmethod
    def from_message(cls, status_code: int, message: str, error_type: str) -> "ProviderError":
        return cls(status_code, {"error": {"message": message, "type": error_type}})


class Provider(Protocol):
    name: str
    model: str

    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a non-streaming chat completion and return the JSON response."""
        ...

    async def stream(self, payload: dict[str, Any]) -> AsyncIterator[bytes]:
        """Start a streaming chat completion.

        Raises ProviderError before returning if the upstream rejects the request,
        so callers can still send a proper error status. The returned iterator
        yields raw SSE bytes.
        """
        ...

    async def aclose(self) -> None: ...
