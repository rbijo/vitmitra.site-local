"""
VIT FAQ Support Chatbot
QLoRA Fine-Tuning Script for Qwen3-1.7B

Training strategy:
    - Base model: Qwen/Qwen3-1.7B
    - 4-bit NF4 quantization
    - Double quantization
    - BF16 computation
    - LoRA adapters
    - Prompt/completion dataset
    - Completion-only loss

Hardware target:
    NVIDIA RTX 4050 Laptop GPU, 6 GB VRAM

Usage:
    Smoke test:
        python src/train.py --smoke-test

    Full training:
        python src/train.py
"""

# ---------------------------------------------------------------------------
# IMPORTANT: Set CUDA allocator configuration before importing torch.
# ---------------------------------------------------------------------------

import os

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import argparse
import math
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import SFTConfig, SFTTrainer


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SFT_DIR = PROJECT_ROOT / "data" / "sft"
OUTPUT_DIR = PROJECT_ROOT / "models" / "vit-faq-qwen3-qlora"

TRAIN_FILE = SFT_DIR / "train.jsonl"
VALIDATION_FILE = SFT_DIR / "validation.jsonl"


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-1.7B"

MAX_LENGTH = 128


# ---------------------------------------------------------------------------
# LoRA configuration
# ---------------------------------------------------------------------------

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

NUM_EPOCHS = 3

PER_DEVICE_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4

EFFECTIVE_BATCH_SIZE = (
    PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
)

LEARNING_RATE = 1e-4

# Originally intended warmup ratio:
WARMUP_RATIO = 0.05

WEIGHT_DECAY = 0.01

SEED = 42

LOGGING_STEPS = 10
EVAL_STEPS = 50
SAVE_STEPS = 50
SAVE_TOTAL_LIMIT = 2


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def print_header(title: str) -> None:
    """Print a consistent section header."""

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_gpu_memory(label: str) -> None:
    """Print current CUDA memory usage."""

    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)

    print(
        f"{label}: "
        f"allocated={allocated:.2f} GB, "
        f"reserved={reserved:.2f} GB"
    )


def get_device_dtype() -> torch.dtype:
    """Select the appropriate compute dtype."""

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16

    return torch.float16


def calculate_warmup_steps(
    dataset_size: int,
    epochs: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    warmup_ratio: float,
) -> tuple[int, int]:
    """
    Calculate total optimizer steps and warmup steps.

    Example for this project:

        800 samples
        batch size = 4
        gradient accumulation = 4
        3 epochs

        800 / 4 = 200 dataloader batches
        200 / 4 = 50 optimizer steps per epoch
        50 * 3 = 150 total optimizer steps

        150 * 0.05 = 7.5
        => 8 warmup steps
    """

    batches_per_epoch = math.ceil(
        dataset_size / batch_size
    )

    optimizer_steps_per_epoch = math.ceil(
        batches_per_epoch / gradient_accumulation_steps
    )

    total_optimizer_steps = math.ceil(
        optimizer_steps_per_epoch * epochs
    )

    warmup_steps = max(
        0,
        round(total_optimizer_steps * warmup_ratio),
    )

    return total_optimizer_steps, warmup_steps


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def print_environment() -> None:
    """Print environment information."""

    print_header("ENVIRONMENT")

    print(f"PyTorch       : {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version  : {torch.version.cuda}")
        print(f"GPU           : {torch.cuda.get_device_name(0)}")

        total_memory = torch.cuda.get_device_properties(0).total_memory
        total_memory_gb = total_memory / (1024 ** 3)

        print(f"GPU VRAM      : {total_memory_gb:.2f} GB")
        print(
            f"BF16 supported: "
            f"{torch.cuda.is_bf16_supported()}"
        )
    else:
        raise RuntimeError(
            "CUDA is not available. "
            "This project is configured for GPU QLoRA training."
        )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_training_datasets():
    """Load the prepared prompt/completion datasets."""

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training dataset not found:\n{TRAIN_FILE}"
        )

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            f"Validation dataset not found:\n{VALIDATION_FILE}"
        )

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(TRAIN_FILE),
            "validation": str(VALIDATION_FILE),
        },
    )

    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]

    print_header("DATASET")

    print(
        f"Training examples   : {len(train_dataset)}"
    )

    print(
        f"Validation examples : {len(validation_dataset)}"
    )

    validate_dataset_structure(
        train_dataset,
        "training",
    )

    validate_dataset_structure(
        validation_dataset,
        "validation",
    )

    print("[PASS] Dataset structure validated.")

    return train_dataset, validation_dataset


def validate_dataset_structure(dataset, split_name: str) -> None:
    """
    Validate the prompt/completion structure expected by TRL.

    Expected format:

        {
            "prompt": [
                {
                    "role": "user",
                    "content": "..."
                }
            ],
            "completion": [
                {
                    "role": "assistant",
                    "content": "..."
                }
            ]
        }
    """

    required_fields = {"prompt", "completion"}

    actual_fields = set(dataset.column_names)

    missing_fields = required_fields - actual_fields

    if missing_fields:
        raise ValueError(
            f"{split_name} dataset is missing fields: "
            f"{sorted(missing_fields)}"
        )

    for index, example in enumerate(dataset):
        prompt = example["prompt"]
        completion = example["completion"]

        if not isinstance(prompt, list):
            raise TypeError(
                f"{split_name} example {index}: "
                "'prompt' must be a list."
            )

        if not isinstance(completion, list):
            raise TypeError(
                f"{split_name} example {index}: "
                "'completion' must be a list."
            )

        if len(prompt) == 0:
            raise ValueError(
                f"{split_name} example {index}: "
                "'prompt' cannot be empty."
            )

        if len(completion) == 0:
            raise ValueError(
                f"{split_name} example {index}: "
                "'completion' cannot be empty."
            )

        for message in prompt:
            if not isinstance(message, dict):
                raise TypeError(
                    f"{split_name} example {index}: "
                    "prompt message must be a dictionary."
                )

            if message.get("role") != "user":
                raise ValueError(
                    f"{split_name} example {index}: "
                    "prompt role must be 'user'."
                )

            if not isinstance(message.get("content"), str):
                raise TypeError(
                    f"{split_name} example {index}: "
                    "prompt content must be a string."
                )

        for message in completion:
            if not isinstance(message, dict):
                raise TypeError(
                    f"{split_name} example {index}: "
                    "completion message must be a dictionary."
                )

            if message.get("role") != "assistant":
                raise ValueError(
                    f"{split_name} example {index}: "
                    "completion role must be 'assistant'."
                )

            if not isinstance(message.get("content"), str):
                raise TypeError(
                    f"{split_name} example {index}: "
                    "completion content must be a string."
                )


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def load_project_tokenizer():
    """Load the Qwen3 tokenizer."""

    print_header("TOKENIZER")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(
        f"Tokenizer class : "
        f"{tokenizer.__class__.__name__}"
    )

    print(
        f"Vocabulary size : "
        f"{len(tokenizer)}"
    )

    print(
        f"Chat template   : "
        f"{tokenizer.chat_template is not None}"
    )

    print(
        f"PAD token       : "
        f"{tokenizer.pad_token!r}"
    )

    print(
        f"EOS token       : "
        f"{tokenizer.eos_token!r}"
    )

    return tokenizer


# ---------------------------------------------------------------------------
# QLoRA configuration
# ---------------------------------------------------------------------------

def create_quantization_config(
    compute_dtype: torch.dtype,
) -> BitsAndBytesConfig:
    """Create the 4-bit NF4 QLoRA configuration."""

    print_header("QLORA QUANTIZATION")

    print("4-bit quantization       : enabled")
    print("Quantization type        : NF4")
    print("Double quantization     : enabled")
    print(f"Compute dtype            : {compute_dtype}")

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(
    quantization_config: BitsAndBytesConfig,
    compute_dtype: torch.dtype,
):
    """Load the Qwen3 model using 4-bit quantization."""

    print_header("MODEL")

    print(f"Loading model: {MODEL_NAME}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        dtype=compute_dtype,
    )

    # KV caching is incompatible with gradient checkpointing
    # during training.
    model.config.use_cache = False

    print(
        f"Model class : "
        f"{model.__class__.__name__}"
    )

    print(
        f"Model device: "
        f"{model.device}"
    )

    print_gpu_memory("After model loading")

    return model


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

def create_lora_config() -> LoraConfig:
    """Create the LoRA adapter configuration."""

    print_header("LORA")

    print(f"Rank              : {LORA_R}")
    print(f"Alpha             : {LORA_ALPHA}")
    print(f"Dropout           : {LORA_DROPOUT}")
    print("Target modules:")

    for module in LORA_TARGET_MODULES:
        print(f"  - {module}")

    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

def create_training_config(
    train_dataset_size: int,
    smoke_test: bool,
) -> SFTConfig:
    """Create TRL SFTConfig compatible with installed TRL."""

    if smoke_test:
        epochs = 1
        max_steps = 2

        # The smoke test should not spend time warming up.
        warmup_steps = 0

    else:
        epochs = NUM_EPOCHS
        max_steps = -1

        _, warmup_steps = calculate_warmup_steps(
            dataset_size=train_dataset_size,
            epochs=epochs,
            batch_size=PER_DEVICE_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_ratio=WARMUP_RATIO,
        )

    print_header("TRAINING CONFIGURATION")

    print(
        f"Epochs                    : {epochs}"
    )

    if smoke_test:
        print(
            f"Smoke-test max steps     : {max_steps}"
        )
    else:
        print(
            f"Max steps                : {max_steps}"
        )

    print(
        f"Per-device batch size     : "
        f"{PER_DEVICE_BATCH_SIZE}"
    )

    print(
        f"Gradient accumulation    : "
        f"{GRADIENT_ACCUMULATION_STEPS}"
    )

    print(
        f"Effective batch size      : "
        f"{EFFECTIVE_BATCH_SIZE}"
    )

    print(
        f"Learning rate              : "
        f"{LEARNING_RATE}"
    )

    print(
        f"Maximum sequence length   : "
        f"{MAX_LENGTH}"
    )

    print(
        f"Warmup ratio              : "
        f"{WARMUP_RATIO:.2%}"
    )

    print(
        f"Warmup steps              : "
        f"{warmup_steps}"
    )

    print(
        f"Weight decay              : "
        f"{WEIGHT_DECAY}"
    )

    print(
        f"Seed                      : "
        f"{SEED}"
    )

    return SFTConfig(
        output_dir=str(OUTPUT_DIR),

        # Training length
        num_train_epochs=epochs,
        max_steps=max_steps,

        # Batch configuration
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

        # Optimizer
        learning_rate=LEARNING_RATE,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=WEIGHT_DECAY,

        # Precision
        bf16=torch.cuda.is_available()
        and torch.cuda.is_bf16_supported(),
        fp16=(
            torch.cuda.is_available()
            and not torch.cuda.is_bf16_supported()
        ),

        # Sequence handling
        max_length=MAX_LENGTH,
        completion_only_loss=True,

        # Evaluation
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,

        # Logging
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        logging_first_step=True,

        # Checkpointing
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,

        # Reproducibility
        seed=SEED,

        # Memory optimization
        gradient_checkpointing=True,

        # Data loading
        dataloader_pin_memory=True,

        # Dataset columns
        remove_unused_columns=False,

        # No external experiment tracking
        report_to="none",

        # No Hugging Face Hub upload
        push_to_hub=False,
    )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def create_trainer(
    model,
    tokenizer,
    train_dataset,
    validation_dataset,
    training_config,
    lora_config,
):
    """Create the TRL SFTTrainer."""

    print_header("TRAINER")

    trainer = SFTTrainer(
        model=model,
        args=training_config,

        train_dataset=train_dataset,
        eval_dataset=validation_dataset,

        processing_class=tokenizer,

        peft_config=lora_config,
    )

    print("[PASS] SFTTrainer created successfully.")

    return trainer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run QLoRA training."""

    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-1.7B with QLoRA."
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run a minimal 2-step training test "
            "instead of full training."
        ),
    )

    args = parser.parse_args()

    smoke_test = args.smoke_test

    # -----------------------------------------------------------------------
    # Startup
    # -----------------------------------------------------------------------

    print("=" * 80)
    print("VIT FAQ QLORA TRAINING")
    print("=" * 80)

    if smoke_test:
        print()
        print("MODE: SMOKE TEST")
        print("Only 2 optimizer steps will be performed.")
    else:
        print()
        print("MODE: FULL TRAINING")
        print(
            f"{NUM_EPOCHS} epochs will be performed."
        )

    # -----------------------------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------------------------

    set_seed(SEED)

    # -----------------------------------------------------------------------
    # Environment
    # -----------------------------------------------------------------------

    print_environment()

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------

    train_dataset, validation_dataset = (
        load_training_datasets()
    )

    # -----------------------------------------------------------------------
    # Tokenizer
    # -----------------------------------------------------------------------

    tokenizer = load_project_tokenizer()

    # -----------------------------------------------------------------------
    # Quantization
    # -----------------------------------------------------------------------

    compute_dtype = get_device_dtype()

    quantization_config = create_quantization_config(
        compute_dtype,
    )

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------

    model = load_model(
        quantization_config=quantization_config,
        compute_dtype=compute_dtype,
    )

    # -----------------------------------------------------------------------
    # LoRA
    # -----------------------------------------------------------------------

    lora_config = create_lora_config()

    # -----------------------------------------------------------------------
    # Training configuration
    # -----------------------------------------------------------------------

    training_config = create_training_config(
        train_dataset_size=len(train_dataset),
        smoke_test=smoke_test,
    )

    # -----------------------------------------------------------------------
    # Trainer
    # -----------------------------------------------------------------------

    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        training_config=training_config,
        lora_config=lora_config,
    )

    print_gpu_memory("Before training")

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    print_header("TRAINING")

    if smoke_test:
        print(
            "Starting smoke test..."
        )
    else:
        print(
            "Starting full QLoRA training..."
        )

    train_result = trainer.train()

    print()
    print(
        "[PASS] Training completed successfully."
    )

    # -----------------------------------------------------------------------
    # Training metrics
    # -----------------------------------------------------------------------

    print_header("TRAINING RESULTS")

    metrics = train_result.metrics

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print_gpu_memory("After training")

    # -----------------------------------------------------------------------
    # Save adapter
    # -----------------------------------------------------------------------

    if smoke_test:
        smoke_output_dir = (
            OUTPUT_DIR / "smoke_test"
        )

        print_header("SMOKE TEST OUTPUT")

        print(
            f"Saving smoke-test adapter to:\n"
            f"{smoke_output_dir}"
        )

        trainer.save_model(
            str(smoke_output_dir)
        )

        tokenizer.save_pretrained(
            str(smoke_output_dir)
        )

        print(
            "[PASS] Smoke-test adapter saved."
        )

    else:
        final_output_dir = (
            OUTPUT_DIR / "final_adapter"
        )

        print_header("FINAL MODEL OUTPUT")

        print(
            f"Saving final adapter to:\n"
            f"{final_output_dir}"
        )

        trainer.save_model(
            str(final_output_dir)
        )

        tokenizer.save_pretrained(
            str(final_output_dir)
        )

        print(
            "[PASS] Final LoRA adapter saved."
        )

    # -----------------------------------------------------------------------
    # Final status
    # -----------------------------------------------------------------------

    print()
    print("=" * 80)

    if smoke_test:
        print(
            "SMOKE TEST PASSED"
        )
        print(
            "The QLoRA training pipeline is operational."
        )
        print(
            "Full training can now be started."
        )
    else:
        print(
            "QLORA TRAINING COMPLETED"
        )
        print(
            "The final LoRA adapter has been saved."
        )

    print("=" * 80)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
