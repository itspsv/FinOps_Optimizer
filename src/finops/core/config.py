"""Application settings.

Precedence (highest first): init kwargs > environment variables > .env > config/settings.yaml.
Environment variables use the FINOPS_ prefix and "__" for nesting,
e.g. FINOPS_PREMIUM__API_KEY.
"""

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_FILE = Path(os.environ.get("FINOPS_CONFIG_FILE", PROJECT_ROOT / "config" / "settings.yaml"))


class ProviderSettings(BaseModel):
    """Connection details for one OpenAI-compatible upstream."""

    name: str
    base_url: str
    model: str
    api_key: SecretStr | None = None
    timeout_s: float = 60.0


class CacheSettings(BaseModel):
    enabled: bool = True
    similarity_threshold: float = Field(0.92, ge=0, le=1)
    ttl_s: int = 86400
    max_temperature: float = 1.0
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    model_cache_dir: Path = Path("~/.cache/finops/fastembed")


class RoutingSettings(BaseModel):
    enabled: bool = True
    policy_version: str = "v1"
    local_max_prompt_tokens: int = 1500
    local_max_completion_tokens: int = 512
    complexity_threshold: int = 2
    breaker_failure_threshold: int = 3
    breaker_cooldown_s: float = 30.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FINOPS_",
        env_nested_delimiter="__",
        env_file=PROJECT_ROOT / ".env",
        extra="ignore",
        yaml_file=CONFIG_FILE,
    )

    log_level: str = "INFO"
    database_url: str = "postgresql:///finops"
    premium: ProviderSettings
    local: ProviderSettings | None = None
    routing: RoutingSettings = RoutingSettings()
    cache: CacheSettings = CacheSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
