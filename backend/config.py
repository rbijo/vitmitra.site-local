from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = os.getenv("VIT_BASE_MODEL", "Qwen/Qwen3-1.7B")
ADAPTER_DIR = Path(os.getenv("VIT_ADAPTER_DIR", str(PROJECT_ROOT / "models" / "vit-faq-qwen3-qlora" / "final_adapter")))
MAX_INPUT_LENGTH = int(os.getenv("VIT_MAX_INPUT_LENGTH", "256"))
MAX_NEW_TOKENS = int(os.getenv("VIT_MAX_NEW_TOKENS", "96"))
HOST = os.getenv("VIT_HOST", "127.0.0.1")
PORT = int(os.getenv("VIT_PORT", "8000"))
