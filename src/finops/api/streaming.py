"""Replay a stored (non-streaming) completion as an OpenAI-style SSE stream."""

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
