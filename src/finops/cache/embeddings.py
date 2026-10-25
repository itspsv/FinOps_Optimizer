from typing import Protocol

import anyio


class Embedder(Protocol):
    model_name: str

    async def embed(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    """Local ONNX embeddings. The model downloads once on first use (~70 MB)."""

    def __init__(self, model_name: str, cache_dir: str):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    async def embed(self, text: str) -> list[float]:
        # Inference is CPU-bound; keep it off the event loop.
        return await anyio.to_thread.run_sync(self._embed_sync, text)

    def _embed_sync(self, text: str) -> list[float]:
        return next(iter(self._model.embed([text]))).tolist()
