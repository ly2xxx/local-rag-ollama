"""Typed settings for the API tier (DESIGN.md §5.6).

Sourced from environment variables / `.env`. Cluster deployments supply the
non-secret values via ConfigMap and the secrets via Secret.

The legacy module-level constants in `config.py` are still used by the agent
core; unifying the two is deferred (see PHASED.md, Phase 1 deviations).
"""

from functools import lru_cache
from typing import Dict, List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .config import OLLAMA_BASE_URL, OLLAMA_MODEL, REDIS_URL


class Settings(BaseSettings):
    """Application settings. Env names are the field names, case-insensitive."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Identity -----------------------------------------------------------
    app_name: str = "agentic-rag-api"
    app_version: str = "0.1.0"
    environment: str = "local"

    # --- Auth ---------------------------------------------------------------
    # "key1:tenant_a,key2:tenant_b". Auth fails closed: an empty map means every
    # authenticated endpoint returns 401.
    api_keys: str = ""
    auth_enabled: bool = True
    default_tenant: str = "public"

    # --- Upstreams ----------------------------------------------------------
    redis_url: str = REDIS_URL
    ollama_model: str = OLLAMA_MODEL
    ollama_base_url: str = OLLAMA_BASE_URL

    # --- Request handling ---------------------------------------------------
    max_upload_bytes: int = 25 * 1024 * 1024

    # --- SSE ----------------------------------------------------------------
    sse_heartbeat_seconds: float = 15.0
    sse_token_chunk_chars: int = 24
    sse_observation_preview_bytes: int = 2048

    # --- Readiness ----------------------------------------------------------
    # DESIGN §5.5 treats Redis loss as degradation, but the Phase 1 DoD requires
    # /readyz to fail when Redis is down. Configurable so Phase 5 can revisit
    # without another code change.
    readyz_require_redis: bool = True
    redis_ping_timeout_seconds: float = 1.5

    # --- Startup ------------------------------------------------------------
    warm_embeddings_on_startup: bool = True

    # --- Misc ---------------------------------------------------------------
    cors_allow_origins: str = "*"
    log_level: str = "INFO"

    @field_validator("api_keys", "cors_allow_origins", mode="before")
    @classmethod
    def _coerce_to_str(cls, value: object) -> str:
        return "" if value is None else str(value)

    @property
    def api_key_map(self) -> Dict[str, str]:
        """Maps API key -> tenant_id. Malformed pairs are dropped, not guessed at."""
        mapping: Dict[str, str] = {}
        for pair in self.api_keys.split(","):
            key, separator, tenant = pair.partition(":")
            if not separator:
                continue
            key, tenant = key.strip(), tenant.strip()
            if key and tenant:
                mapping[key] = tenant
        return mapping

    @property
    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()
