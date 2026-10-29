# VIT FAQ Support SLM

Local FastAPI + React chatbot for the supplied Qwen3-1.7B LoRA adapter.

## Run

```powershell
.\myvenv\Scripts\pip install -r requirements.txt
Start-Process powershell -ArgumentList '-NoExit','-Command','.\myvenv\Scripts\uvicorn backend.main:app --host 127.0.0.1 --port 8000'
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The first backend start downloads the base model if it is not already cached.

Fine-tuned Small Language Model for answering VIT Vellore domain-specific FAQ queries.

## Project Structure

- `data/raw/` — Original datasets
- `data/processed/` — Prepared training/validation/test datasets
- `src/` — Training, evaluation, preprocessing and inference code
- `configs/` — Training configuration
- `models/` — Local model artifacts
- `evaluation/` — Evaluation results
- `scripts/` — Utility scripts
- `notebooks/` — Exploratory analysis

## Model

The base model has not yet been fine-tuned.

Model selection and training configuration will be finalized after dataset preparation and benchmarking.

## Training

Training instructions will be added after the dataset preparation pipeline is implemented.

## Deployment

The final model will be packaged for external inference hosting so that the website can consume it through an API.
