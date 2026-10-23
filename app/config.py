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
    openai_chat_model: str = "gpt-4.1-mini"
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
    # If TYPESENSE_URL is set (e.g. a managed instance on Render), it takes
    # precedence over the discrete typesense_* fields below.
    typesense_url: str | None = None
    typesense_host: str = "localhost"
    typesense_port: int = 8108
    typesense_protocol: str = "http"
    typesense_api_key: str = "studymind-typesense-key"

    def get_typesense_node(self) -> dict:
        if self.typesense_url:
            from urllib.parse import urlparse
            parsed = urlparse(self.typesense_url)
            return {
                "host":     parsed.hostname,
                "port":     parsed.port or (443 if parsed.scheme == "https" else 8108),
                "protocol": parsed.scheme or "http",
            }
        return {
            "host":     self.typesense_host,
            "port":     int(self.typesense_port),
            "protocol": self.typesense_protocol,
        }

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

    # ── Cloudinary (platform document storage) ───────────────────────────
    cloudinary_cloud_name: str = Field(default="")
    cloudinary_api_key: str = Field(default="")
    cloudinary_api_secret: str = Field(default="")
    cloudinary_folder: str = Field(default="studymind/documents")

    @property
    def cloudinary_configured(self) -> bool:
        return bool(self.cloudinary_cloud_name and self.cloudinary_api_key and self.cloudinary_api_secret)

    # ── App ───────────────────────────────────────────────────────────────
    app_env: str = "development"
    app_log_level: str = "INFO"

    # ── RAG Pipeline ──────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_retrieval: int = 6
    similarity_threshold: float = 0.25

    # ── Usage policies (opt-in quotas for pass sessions; see app/services/usage_policy.py) ──
    # Only API keys whose `platform` is listed may attach a usage policy to a session token.
    # Sessions without a policy are never limited.
    usage_policy_platforms: str = "learnify"
    usage_policy_timezone: str = "Europe/London"
    # learnify_free_course_pass — frozen onto each policy when it is created
    pass_daily_chat: int = 60
    pass_daily_quiz: int = 5
    pass_daily_flashcards: int = 5
    pass_daily_summary: int = 3
    pass_budget_usd: float = 4.50
    pass_max_days: int = 30

    @property
    def usage_policy_platform_set(self) -> set[str]:
        return {p.strip() for p in self.usage_policy_platforms.split(",") if p.strip()}

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
