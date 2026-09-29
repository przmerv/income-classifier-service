from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env")

    model_path: Path = Path("artifacts/model.joblib")
    model_version: str = "0.1.0"
    log_level: str = "INFO"
    random_seed: int = 42
    test_size: float = 0.2
