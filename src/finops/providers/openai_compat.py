"""Client for any OpenAI-compatible chat API (Groq, OpenAI, Ollama's /v1, ...)."""

from collections.abc import AsyncIterator
from typing import Any

import httpx

from finops.core.config import ProviderSettings
from finops.providers.base import ProviderError


class OpenAICompatibleProvider:
    def __init__(self, settings: ProviderSettings):
        self.name = settings.name
        self.model = settings.model
        headers = {}
        if settings.api_key is not None:
            headers["Authorization"] = f"Bearer {settings.api_key.get_secret_value()}"
        self._client = httpx.AsyncClient(
            base_url=settings.base_url.rstrip("/") + "/",
            headers=headers,
            timeout=settings.timeout_s,
        )

    def _prepare(self, payload: dict[str, Any], stream: bool) -> dict[str, Any]:
        return {**payload, "model": self.model, "stream": stream}

    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._client.post(
                "chat/completions", json=self._prepare(payload, stream=False)
            )
        except httpx.HTTPError as exc:
            raise _transport_error(self.name, exc) from exc
        if response.is_error:
            raise _upstream_error(response)
        return response.json()

    async def stream(self, payload: dict[str, Any]) -> AsyncIterator[bytes]:
        request = self._client.build_request(
            "POST", "chat/completions", json=self._prepare(payload, stream=True)
        )
        try:
            response = await self._client.send(request, stream=True)
        except httpx.HTTPError as exc:
            raise _transport_error(self.name, exc) from exc
        if response.is_error:
            await response.aread()
            await response.aclose()
            raise _upstream_error(response)
        return _iter_and_close(response)

    async def aclose(self) -> None:
        await self._client.aclose()


async def _iter_and_close(response: httpx.Response) -> AsyncIterator[bytes]:
    try:
        async for chunk in response.aiter_raw():
            yield chunk
    finally:
        await response.aclose()


def _upstream_error(response: httpx.Response) -> ProviderError:
    try:
        body = response.json()
    except ValueError:
        body = {"error": {"message": response.text, "type": "upstream_error"}}
    return ProviderError(response.status_code, body)


def _transport_error(provider: str, exc: httpx.HTTPError) -> ProviderError:
    if isinstance(exc, httpx.TimeoutException):
        return ProviderError.from_message(504, f"{provider} timed out", "upstream_timeout")
    return ProviderError.from_message(502, f"{provider} unreachable: {exc}", "upstream_unreachable")
