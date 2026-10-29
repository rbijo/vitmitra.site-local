import time
import torch

from .config import MAX_INPUT_LENGTH, MAX_NEW_TOKENS
from .model_loader import clean_response, get_model_and_tokenizer, _generation_lock


def answer_question(question: str) -> dict:
    model, tokenizer = get_model_and_tokenizer()
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LENGTH)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    started = time.perf_counter()
    with _generation_lock, torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    generated = outputs[0, inputs["input_ids"].shape[1]:]
    answer = clean_response(tokenizer.decode(generated, skip_special_tokens=True))
    return {"answer": answer or "I’m sorry, I couldn’t find an answer to that question.", "question_tokens": int(inputs["input_ids"].shape[1]), "response_tokens": int(generated.shape[0]), "context_capacity": int(getattr(model.config, "vitmitra_context_capacity", getattr(model.config, "max_position_embeddings", 32768))), "inference_seconds": round(time.perf_counter() - started, 3)}
