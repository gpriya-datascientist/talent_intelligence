from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Literal


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Talent Intelligence Platform"
    ENV: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = True

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/talent_db"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20

    # Anthropic
    ANTHROPIC_API_KEY: str
    ANTHROPIC_MODEL: str = "claude-sonnet-4-6"
    ANTHROPIC_MAX_TOKENS: int = 2048

    # OpenAI (embeddings)
    OPENAI_API_KEY: str
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536

    # Vector store
    VECTOR_STORE: Literal["faiss", "pinecone"] = "faiss"
    FAISS_INDEX_PATH: str = "./data/faiss_index"
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = "talent-intelligence"
    PINECONE_ENVIRONMENT: str = ""

    # GitHub
    GITHUB_TOKEN: str = ""

    # Langfuse observability
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

    # n8n automation
    N8N_WEBHOOK_URL: str = "http://localhost:5678"
    N8N_API_KEY: str = ""

    # RAG
    RAG_TOP_K: int = 10
    RAG_SCORE_THRESHOLD: float = 0.65

    # Ranking weights — must sum to 1.0
    WEIGHT_SKILL_MATCH: float = 0.45
    WEIGHT_RECENCY: float = 0.20
    WEIGHT_GITHUB_ACTIVITY: float = 0.15
    WEIGHT_AVAILABILITY: float = 0.20

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
