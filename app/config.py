"""
Central application configuration.
All values sourced from environment / .env file.
Pydantic BaseSettings validates every field at startup.
"""
from functools import lru_cache
from urllib.parse import urlsplit, urlunsplit

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── OpenAI ────────────────────────────────────────────────────────────
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_chat_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-large"
    openai_embedding_dimension: int = 3072

    # ── PostgreSQL ────────────────────────────────────────────────────────
    # If DATABASE_URL is set (e.g. a managed Postgres like Neon), it takes
    # precedence over the discrete postgres_* fields below.
    database_url: str | None = None
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "studymind"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"

    def _dsn_with_driver(self, driver: str) -> str:
        if self.database_url:
            parts = urlsplit(self.database_url)
            scheme = f"postgresql+{driver}"
            # asyncpg takes SSL via connect_args (see engine.py), not sslmode/
            # channel_binding query params — psycopg2 (sync) handles them fine.
            query = "" if driver == "asyncpg" else parts.query
            return urlunsplit((scheme, parts.netloc, parts.path, query, parts.fragment))
        return (
            f"postgresql+{driver}://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def postgres_dsn(self) -> str:
        return self._dsn_with_driver("asyncpg")

    @property
    def postgres_dsn_sync(self) -> str:
        return self._dsn_with_driver("psycopg2")

    # ── Typesense ─────────────────────────────────────────────────────────
    typesense_host: str = "localhost"
    typesense_port: int = 8108
    typesense_protocol: str = "http"
    typesense_api_key: str = "studymind-typesense-key"

    # ── Launch gating ─────────────────────────────────────────────────────
    # Comma-separated roles that may self-register. Existing accounts of any role keep working.
    enabled_roles: str = "self_learner"

    @property
    def enabled_role_list(self) -> list[str]:
        return [r.strip() for r in self.enabled_roles.split(",") if r.strip()]

    # ── JWT Auth ──────────────────────────────────────────────────────────
    jwt_secret_key: str = Field(default="change-me-in-production-use-32-char-min")
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60
    jwt_refresh_token_expire_days: int = 7

    # ── App ───────────────────────────────────────────────────────────────
    app_env: str = "development"
    app_log_level: str = "INFO"

    # ── RAG Pipeline ──────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_retrieval: int = 6
    similarity_threshold: float = 0.25

    @field_validator("app_env")
    @classmethod
    def validate_env(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"app_env must be one of {allowed}")
        return v

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
