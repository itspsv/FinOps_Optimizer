from pathlib import Path

import yaml
from pydantic import BaseModel, model_validator


class ModelPrice(BaseModel):
    input_per_1m: float
    output_per_1m: float


class PriceTable(BaseModel):
    version: str
    baseline_model: str
    premium_priced_as: str
    models: dict[str, ModelPrice]

    @model_validator(mode="after")
    def _reference_models_are_priced(self) -> "PriceTable":
        for name in (self.baseline_model, self.premium_priced_as):
            if name not in self.models:
                raise ValueError(f"pricing has no entry for {name!r}")
        return self

    @classmethod
    def load(cls, path: Path) -> "PriceTable":
        return cls.model_validate(yaml.safe_load(path.read_text()))

    def cost(self, model: str | None, prompt_tokens: int, completion_tokens: int) -> float | None:
        """USD for one request, or None if the model isn't in the table."""
        price = self.models.get(model or "")
        if price is None:
            return None
        return (
            prompt_tokens * price.input_per_1m + completion_tokens * price.output_per_1m
        ) / 1_000_000
