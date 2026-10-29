import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import HOST, PORT
from .inference import answer_question
from .model_loader import get_model_and_tokenizer, get_status, set_status

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    def initialize():
        try:
            set_status("starting", "Starting VITMITRA inference service", 5)
            get_model_and_tokenizer()
        except Exception as exc:
            logger.exception("Model failed to load during startup")
            set_status("error", "Model initialization failed", 0, str(exc))
    threading.Thread(target=initialize, name="vitmitra-model-loader", daemon=True).start()
    yield


app = FastAPI(title="VIT FAQ SLM API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@app.get("/health")
def health():
    status = get_status()
    if not status["ready"]:
        raise HTTPException(status_code=503, detail=status)
    return {"status": "ok", "model": "Qwen3-1.7B + LoRA"}


@app.get("/status")
def status():
    current = get_status()
    if current["ready"]:
        _, tokenizer = get_model_and_tokenizer()
        current["context_capacity"] = int(getattr(get_model_and_tokenizer()[0].config, "vitmitra_context_capacity", getattr(get_model_and_tokenizer()[0].config, "max_position_embeddings", 32768)))
        current["tokenizer"] = tokenizer.__class__.__name__
    return current


@app.post("/api/tokenize")
def tokenize(payload: ChatRequest):
    if not get_status()["ready"]:
        raise HTTPException(status_code=503, detail="VITMITRA is still initializing.")
    model, tokenizer = get_model_and_tokenizer()
    tokens = tokenizer.apply_chat_template([{"role": "user", "content": payload.question.strip()}], tokenize=True, add_generation_prompt=True, enable_thinking=False)
    capacity = int(getattr(model.config, "vitmitra_context_capacity", getattr(model.config, "max_position_embeddings", 32768)))
    return {"question_tokens": len(tokens), "context_capacity": capacity}


@app.post("/api/chat")
def chat(payload: ChatRequest):
    if not get_status()["ready"]:
        raise HTTPException(status_code=503, detail="VITMITRA is still initializing.")
    try:
        return answer_question(payload.question.strip())
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="The model could not answer right now.") from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=False)
