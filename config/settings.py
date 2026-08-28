import os
import json
from pathlib import Path
from pydantic import BaseModel
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = BASE_DIR / "enterprise_config.json"

class AppConfig(BaseModel):
    active_provider: str = "ollama"
    model_name: str = "llama3.2:1b"
    api_key: Optional[str] = ""
    min_relevance_threshold: float = 0.15
    pii_masking: bool = True
    admin_secret_key: str = "admin-enterprise-key-2026"
    chroma_host: str = os.getenv("CHROMA_HOST", "localhost")
    chroma_port: int = int(os.getenv("CHROMA_PORT", 8000))
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

def load_config() -> AppConfig:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                return AppConfig(**data)
        except Exception:
            return AppConfig()
    return AppConfig()

def save_config(cfg: AppConfig):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg.dict(), f, indent=4)

settings = load_config()