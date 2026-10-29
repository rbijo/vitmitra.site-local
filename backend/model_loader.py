import logging
import re
import threading
from typing import Any
from functools import lru_cache

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import ADAPTER_DIR, MODEL_NAME

logger = logging.getLogger(__name__)

_status_lock = threading.Lock()
_status: dict[str, Any] = {"ready": False, "phase": "starting", "message": "Starting VITMITRA inference service", "progress": 0}


def set_status(phase: str, message: str, progress: int, error: str | None = None) -> None:
    with _status_lock:
        _status.update({"ready": phase == "ready", "phase": phase, "message": message, "progress": max(0, min(100, progress))})
        if error:
            _status["error"] = error
        elif "error" in _status:
            _status.pop("error")


def get_status() -> dict[str, Any]:
    with _status_lock:
        return dict(_status)


def _dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32


@lru_cache(maxsize=1)
def get_model_and_tokenizer():
    if not (ADAPTER_DIR / "adapter_config.json").exists():
        raise FileNotFoundError(f"LoRA adapter not found: {ADAPTER_DIR}")
    dtype = _dtype()
    logger.info("Loading %s with adapter %s on %s", MODEL_NAME, ADAPTER_DIR, "cuda" if torch.cuda.is_available() else "cpu")
    set_status("loading_tokenizer", "Loading tokenizer", 15)
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"torch_dtype": dtype}
    if torch.cuda.is_available():
        kwargs.update({"device_map": "auto", "quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)})
    set_status("loading_base_model", f"Loading {MODEL_NAME}", 35)
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **kwargs)
    if not torch.cuda.is_available():
        base.to("cpu")
    set_status("loading_adapter", "Loading LoRA adapter", 75)
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR), is_trainable=False)
    model.eval()
    set_status("initializing_inference", "Initializing inference engine", 90)
    model.config.vitmitra_context_capacity = getattr(model.config, "max_position_embeddings", 32768)
    logger.info("Model loaded successfully")
    set_status("ready", "VITMITRA is ready", 100)
    return model, tokenizer


_generation_lock = threading.Lock()


def clean_response(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.I | re.S)
    return re.sub(r"</?think>", "", text, flags=re.I).strip()
