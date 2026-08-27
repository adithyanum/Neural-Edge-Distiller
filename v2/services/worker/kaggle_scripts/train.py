"""
Neural Edge Distiller V2 - real training script.
Runs INSIDE the Kaggle kernel (T4 GPU). Not imported by the worker container -
this file is pushed as the kernel's code_file by KaggleBackend.submit().

Mirrors V1's train.sh intent (LoRA, 100 iters-equivalent, batch 2, lr 1e-4)
translated from mlx_lm.lora to HF transformers + peft.

Expects:
  - /kaggle/input/<dataset-slug>/all.jsonl   (attached Kaggle Dataset, 986 records)
  - env var HF_TOKEN                          (Kaggle Secret, for gated Llama-3.2 checkpoint)

Produces (read back by KaggleBackend.download()):
  - /kaggle/working/adapter_output/           (peft adapter dir: adapter_model.safetensors, adapter_config.json)
  - /kaggle/working/metrics.json              ({"final_loss": <float>})
"""

import os
import subprocess
import sys

subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q",
        "torch==2.4.1", "torchvision==0.19.1", "torchaudio==2.4.1",
        "--index-url", "https://download.pytorch.org/whl/cu121",
    ],
    check=True,
)
subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q",
        "transformers==4.45.2", "peft==0.13.2", "accelerate==0.34.2",
    ],
    check=True,
)

import json
import glob

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

# ============================================================
# Configuration - mirrors V1 train.sh where a direct equivalent exists
# ============================================================

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

DATA_PATH = glob.glob("/kaggle/input/**/all.jsonl", recursive=True)
DATA_PATH = DATA_PATH[0] if DATA_PATH else "all.jsonl"  # local fallback for testing

OUTPUT_DIR = "/kaggle/working"
ADAPTER_DIR = os.path.join(OUTPUT_DIR, "adapter_output")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")

# V1: --num-layers 8 -> mlx_lm.lora applies LoRA to the last 8 transformer layers.

NUM_LORA_LAYERS = 8
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

LEARNING_RATE = 1e-4       
PER_DEVICE_BATCH_SIZE = 2    
NUM_TRAIN_EPOCHS = None       
MAX_STEPS = 100                
MAX_SEQ_LEN = 1024

# Kaggle's API can't attach interactively-configured Secrets to a kernel
# pushed programmatically, so the HF token is shipped as the sole file in a
# small private Kaggle Dataset instead.

def load_hf_token():
    token_paths = glob.glob("/kaggle/input/**/hf_token.txt", recursive=True)
    if not token_paths:
        raise RuntimeError(
            "hf_token.txt not found anywhere under /kaggle/input/ - is the HF "
            "token dataset attached as a dataset_source in kernel-metadata.json?"
        )
    with open(token_paths[0], "r", encoding="utf-8") as f:
        return f.read().strip()


HF_TOKEN = load_hf_token()


# ============================================================
# Data loading
# ============================================================

def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {path}")
    return records


def to_chat_text(record, tokenizer):
    """
    Build a single training string per record using the model's chat template.
    context (when present, e.g. classification full-example records) is folded
    into the user turn ahead of the prompt so the model sees the same input
    shape the judge validated against.
    """
    user_parts = []
    if record.get("context"):
        user_parts.append(f"Context: {record['context']}")
    user_parts.append(record["prompt"])
    user_content = "\n\n".join(user_parts)

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": record["response"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def build_dataset(records, tokenizer):
    texts = [to_chat_text(r, tokenizer) for r in records]

    def tokenize_fn(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
        )
        out["labels"] = out["input_ids"].copy()
        return out

    ds = Dataset.from_dict({"text": texts})
    ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    return ds


# ============================================================
# Model + LoRA setup
# ============================================================

def build_target_module_names(model, num_last_layers, base_target_modules):
    """
    Resolve TARGET_MODULES into fully-qualified module names restricted to the
    last `num_last_layers` decoder layers, reproducing mlx_lm.lora's --num-layers.
    """
    layer_indices = set()
    total_layers = model.config.num_hidden_layers
    start = max(0, total_layers - num_last_layers)
    for i in range(start, total_layers):
        layer_indices.add(i)

    target_names = []
    for name, _ in model.named_modules():
        for idx in layer_indices:
            prefix = f"layers.{idx}."
            if prefix in name and any(name.endswith(t) for t in base_target_modules):
                target_names.append(name)
    return target_names


def get_supported_dtype():
    """
    bf16 requires Ampere+ (compute capability >= 8.0). Kaggle sometimes
    assigns a P100 (Pascal, 6.0), which has no hardware bf16 support -
    running bf16 there either silently emulates (very slow) or hits the
    same class of kernel-image error we just fixed for fp32 casts.
    Use fp16 on anything older than Ampere.
    """
    if not torch.cuda.is_available():
        return torch.float32, False, False
    major, _ = torch.cuda.get_device_capability(0)
    if major >= 8:
        return torch.bfloat16, True, False
    return torch.float16, False, True


def main():
    print("=" * 60)
    print("Neural Edge Distiller V2 - Real Training Run")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    compute_dtype, use_bf16, use_fp16 = get_supported_dtype()
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} (dtype: {compute_dtype})")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        torch_dtype=compute_dtype,
        device_map="auto",
    )

    target_modules = build_target_module_names(model, NUM_LORA_LAYERS, TARGET_MODULES)
    print(f"LoRA target modules ({len(target_modules)}): last {NUM_LORA_LAYERS} layers")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    records = load_records(DATA_PATH)
    train_ds = build_dataset(records, tokenizer)

    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "checkpoints"),
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
    )

    print("Starting training...")
    train_result = trainer.train()
    final_loss = train_result.training_loss
    print(f"Training complete. final_loss={final_loss}")

    os.makedirs(ADAPTER_DIR, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    with open(METRICS_PATH, "w") as f:
        json.dump({"final_loss": final_loss}, f)

    print(f"Adapter saved to {ADAPTER_DIR}")
    print(f"Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()