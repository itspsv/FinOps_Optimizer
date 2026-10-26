"""SSE helpers: replay a stored completion as a stream, and observe a passthrough stream."""

import json
from collections.abc import AsyncIterator
from typing import Any


async def completion_to_sse(completion: dict[str, Any]) -> AsyncIterator[bytes]:
    choice = completion["choices"][0]
    base = {
        "id": completion.get("id"),
        "object": "chat.completion.chunk",
        "created": completion.get("created"),
        "model": completion.get("model"),
    }
    content_chunk = {
        **base,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": choice["message"]["content"]},
                "finish_reason": None,
            }
        ],
    }
    final_chunk = {
        **base,
        "choices": [{"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason")}],
    }
    for chunk in (content_chunk, final_chunk):
        yield f"data: {json.dumps(chunk)}\n\n".encode()
    yield b"data: [DONE]\n\n"


class StreamTap:
    """Passes SSE bytes through unchanged while collecting text, model and usage.

    Bytes are forwarded as they arrive; parsing never delays or alters the stream.
    """

    def __init__(self, source: AsyncIterator[bytes]):
        self._source = source
        self._buffer = b""
        self.text_parts: list[str] = []
        self.model: str | None = None
        self.usage: dict[str, Any] | None = None

    @property
    def text(self) -> str:
        return "".join(self.text_parts)

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._source:
            yield chunk
            self._observe(chunk)

    def _observe(self, chunk: bytes) -> None:
        self._buffer += chunk
        *lines, self._buffer = self._buffer.split(b"\n")
        for line in lines:
            if not line.startswith(b"data: {"):
                continue
            try:
                event = json.loads(line[6:])
            except ValueError:
                continue
            self.model = event.get("model") or self.model
            # OpenAI puts usage on the last chunk; Groq also uses `x_groq.usage`.
            usage = event.get("usage") or (event.get("x_groq") or {}).get("usage")
            if usage:
                self.usage = usage
            for choice in event.get("choices") or []:
                if content := (choice.get("delta") or {}).get("content"):
                    self.text_parts.append(content)
