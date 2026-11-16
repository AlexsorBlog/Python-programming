from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Config only from the environment: INDEX_PATH, DEFAULT_K, MAX_K."""

    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)

    index_path: Path = Path("data/demo.pkl")
    # optional: without it only mode=lexical works
    embeddings_path: Path | None = None
    default_k: int = 10
    max_k: int = 50
    title: str = "findex"
