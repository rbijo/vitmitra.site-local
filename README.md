# VIT FAQ Support SLM

Local FastAPI + React chatbot for the supplied Qwen3-1.7B LoRA adapter.

<!-- VITMITRA Project Status -->
![MVP](https://img.shields.io/badge/MVP-Complete-brightgreen?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Archived-lightgrey?style=for-the-badge)
![Maintenance](https://img.shields.io/badge/Maintenance-Inactive-red?style=for-the-badge)
![Development](https://img.shields.io/badge/Development-Complete-success?style=for-the-badge)

<!-- Repository -->
![Last Commit](https://img.shields.io/github/last-commit/rbijo/vitmitra.site-local/main?style=for-the-badge&logo=github)
![Language](https://img.shields.io/github/languages/top/rbijo/vitmitra.site-local?style=for-the-badge)
![Repo Size](https://img.shields.io/github/repo-size/rbijo/vitmitra.site-local?style=for-the-badge)

<!-- AI Model -->
![Qwen3](https://img.shields.io/badge/Model-Qwen3--1.7B-7E57C2?style=for-the-badge)
![QLoRA](https://img.shields.io/badge/Fine--Tuning-QLoRA-blueviolet?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch)

<!-- Hugging Face Models -->
[![Base Model](https://img.shields.io/badge/Hugging%20Face-Base%20Model-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co/Qwen/Qwen3-1.7B-Base)

[![Fine-tuned Model](https://img.shields.io/badge/Hugging%20Face-VIT%20FAQ%20Adapter-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co/robinbijo/vit-faq-qwen3-1.7b)

[![Model Downloads](https://img.shields.io/badge/Model-Available%20on%20Hugging%20Face-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co/robinbijo/vit-faq-qwen3-1.7b)


<!-- Tech Stack -->
![React](https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi)
![Vite](https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite)
![PEFT](https://img.shields.io/badge/Model%20Integration-PEFT-blueviolet?style=for-the-badge)
![Transformers](https://img.shields.io/badge/Library-Transformers-yellow?style=for-the-badge&logo=huggingface)
![REST API](https://img.shields.io/badge/API-REST-009688?style=for-the-badge)
![Inference](https://img.shields.io/badge/Inference-Local-green?style=for-the-badge)

<!-- Security -->
![Security](https://img.shields.io/badge/Security-Not%20Audited-lightgrey?style=for-the-badge)

## Run
Clone **this** repo and download the [base model](https://huggingface.co/Qwen/Qwen3-1.7B-Base) and the [Model Adapter](https://huggingface.co/robinbijo/vit-faq-qwen3-1.7b) into the same directory <br>
In the IDE that is at the root folder run:
```powershell
.\myvenv\Scripts\pip install -r requirements.txt
Start-Process powershell -ArgumentList '-NoExit','-Command','.\myvenv\Scripts\uvicorn backend.main:app --host 127.0.0.1 --port 8000'
```
In another terminal pointing to the same directory run:
```powershell
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

# SECURITY AUDIT

<!-- Security and Quality -->
<!-- Security -->
![Security](https://img.shields.io/badge/Security-Not%20Audited-red?style=for-the-badge)
![Security Level](https://img.shields.io/badge/Security%20Level-Unverified-orange?style=for-the-badge)
![Secrets](https://img.shields.io/badge/Secrets-Environment%20Based-blue?style=for-the-badge)
![Dependency Audit](https://img.shields.io/badge/Dependency%20Audit-Not%20Verified-lightgrey?style=for-the-badge)
![Documentation](https://img.shields.io/badge/Documentation-Not%20Available-red?style=for-the-badge)

