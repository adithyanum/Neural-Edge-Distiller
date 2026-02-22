# Neural Edge Distiller

A synthetic Chain-of-Thought distillation pipeline that transfers structured reasoning from a large teacher model to a smaller student model, running entirely on Apple Silicon via MLX.

## What it does

Distills structured reasoning behaviour from **Llama 3.2 8B** into **Llama 3.2 3B** using LoRA fine-tuning on 25 hand-crafted distributed systems scenarios. The student model learns to follow a strict 4-step reasoning format — FAILURE MODE → NAIVE FIXES → MECHANISM → TRADE-OFF — that the vanilla model produces inconsistently.

**Result: structured response rate improves from 1/3 to 3/3 on benchmark prompts, with identical memory footprint.**

---

## Project Structure

```
Neural_Edge_Distiller/
├── ui/
│   └── app.py                     # Streamlit UI — side-by-side inference
├── benchmarks/
│   └── latency_check.py           # Before/after benchmark script
├── data/
│   ├── raw/
│   │   └── curriculum_goals.json  # 25 distillation scenarios
│   └── training/
│       ├── train.jsonl            # Training split (20 entries)  --| generated with 
│       └── valid.jsonl            # Validation split (5 entries) --|           prepare_data_.py
├── models/
│   ├── adapters/                  # LoRA adapter weights
│   └── neural-edge-3b/            # Fused fine-tuned model
├── results/
│   ├── benchmark_results.txt      # Saved benchmark output
│   └── *.png                      # UI screenshots
├── scripts/
│   ├── __init__.py
│   ├── utils.py                   # Shared prompt logic and helpers
│   ├── generate_curriculum.py     # CoT generation with 8B teacher
│   ├── data_factory.py            # Generation loop over curriculum
│   ├── prepare_data.py            # Convert JSONL to MLX chat format
│   ├── train.sh                   # LoRA fine-tuning command
│   └── fuse.sh                    # Merge adapters into base model
└── README.md
```

---

## Setup

```bash
# Create and activate virtual environment
python -m venv distiller_env
source distiller_env/bin/activate

# Install dependencies
pip install mlx-lm streamlit
```

---

## Pipeline

### 1. Generate training data
Uses the 8B teacher model to produce structured CoT responses across 25 distributed systems scenarios.

```bash
python scripts/data_factory.py
```

### 2. Prepare data for training
Converts to MLX chat format and creates train/valid split.

```bash
python scripts/prepare_data.py
```

### 3. Fine-tune with LoRA
Trains only 0.1% of model parameters (3.47M / 3.2B) via LoRA on Apple Silicon.

```bash
./scripts/train.sh
```

### 4. Fuse adapters
Merges LoRA weights into the base model to create the final deployable model.

```bash
./scripts/fuse.sh
```

### 5. Run benchmark
Compares vanilla 3B vs fine-tuned Neural Edge 3B across 3 test prompts.

```bash
python benchmarks/latency_check.py
```

### 6. Launch UI
Side-by-side inference interface with live metrics.

```bash
streamlit run ui/app.py
```

---

## Benchmark Results

| Metric | Vanilla 3B | Neural Edge 3B |
|---|---|---|
| Structured responses | 1/3 | 3/3 |
| Avg tokens/sec | 33.2 | 35.9 |
| Peak memory | 12.71 GB | 12.71 GB |

---

## Key Design Decisions

**Why LoRA over full fine-tuning** — Trains in minutes on M4 MacBook instead of days. Only adapter weights are updated; base model stays frozen.

**Why 4-bit quantization** — Reduces 8B model from ~16GB to ~5GB, fitting comfortably in 16GB unified memory alongside training overhead.

**Why synthetic CoT data** — No labelled dataset exists for this reasoning format. The 8B teacher generates structured examples that the 3B student learns to replicate.

**Why no automated validation** — At 25 samples, manual review is more reliable than automated checks that share the same blind spots as the generator. Three factually incorrect entries were identified and corrected manually.

---

## Tech Stack

- [MLX](https://github.com/ml-explore/mlx) — Apple Silicon ML framework
- [mlx-lm](https://github.com/ml-explore/mlx-lm) — LLM inference and LoRA fine-tuning
- [Streamlit](https://streamlit.io) — UI framework
- Models: `mlx-community/Meta-Llama-3-8B-Instruct-4bit` (teacher), `mlx-community/Llama-3.2-3B-Instruct-4bit` (student)
