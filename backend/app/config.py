"""Application configuration."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "plant_disease_mobilenetv3.pt"
LABELS_PATH = PROJECT_ROOT / "models" / "class_labels.json"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

IMAGE_SIZE = 224
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_prefix="PLANT_DEMO_", protected_namespaces=("settings_",))

    host: str = "0.0.0.0"
    port: int = 7860
    model_path: Path = MODEL_PATH
    labels_path: Path = LABELS_PATH
    frontend_dir: Path = FRONTEND_DIR
    max_upload_bytes: int = MAX_UPLOAD_BYTES
    top_k: int = 3


settings = Settings()
