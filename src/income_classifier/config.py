from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All settings the service needs, in one place.

    Each value has a default in code and can be overridden by an environment
    variable with the APP_ prefix (e.g. APP_LOG_LEVEL=DEBUG) or by a .env
    file. Values are type-checked at startup, so a bad setting fails fast.
    """

    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env")

    model_path: Path = Path("artifacts/model.joblib")  # where the trained model is saved/loaded
    model_version: str = "0.1.0"  # reported by /predict and /version
    log_level: str = "INFO"  # how much detail logs show
    random_seed: int = 42  # makes the train/test split repeatable
    test_size: float = 0.2  # share of data held back for testing
    git_sha: str = "local"  # code commit, set by CI at build time
