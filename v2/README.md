# Neural Edge Distiller V2

A production-shaped LLM distillation platform: train a LoRA adapter that teaches Llama-3.2-3B to follow a chain-of-thought format, run it through a real job-orchestration pipeline on free infrastructure, and serve it with vLLM under measured concurrency.

V1 proved the idea by training locally on Apple Silicon with MLX. V2 proves the *platform*: the same idea pushed through a distributed pipeline — queue, worker, cloud training, cloud storage, cloud serving — all on zero paid infrastructure.

## What this is

- **Base model:** `meta-llama/Llama-3.2-3B-Instruct`
- **Method:** LoRA (r=16, alpha=32, dropout=0.05) on q/k/v/o projections, restricted to the last 8 of 28 decoder layers
- **Dataset:** 986 records across 10 task categories (math reasoning, logical reasoning, code debugging, classification, planning, instruction following, code generation, data transformation, information extraction, question answering)
- **Target behavior:** structured CoT output (`THOUGHT:` → `ANSWER:`) for reasoning tasks, bare output for pure-instruction tasks
- **Training infra:** Kaggle T4 (free tier), orchestrated through a Docker Compose stack
- **Serving infra:** a separate Kaggle T4 running vLLM with native LoRA support, exposed via a Cloudflare quick tunnel

## Architecture

```
 ┌─────────┐    ┌───────┐    ┌──────────────┐    ┌─────────────┐
 │ gateway │───▶│ redis │───▶│  ray worker  │───▶│ Kaggle T4   │
 └─────────┘    └───────┘    │ (job queue)  │    │ train.py    │
                              └──────┬───────┘    └──────┬──────┘
                                     │                    │
                               metrics logged        adapter saved
                                     ▼                    ▼
                              ┌───────────┐        ┌──────────────┐
                              │  MLflow   │        │ Azure Blob   │
                              │ (Postgres)│        │ Storage      │
                              └───────────┘        └──────┬───────┘
                                                           │
                                                   manual deploy
                                                           ▼
                                                 ┌──────────────────┐
                                                 │ Kaggle T4         │
                                                 │ vLLM + LoRA       │
                                                 │ + Cloudflare tunnel│
                                                 └──────────────────┘
```

Training is queue-driven (gateway → Redis → Ray worker → Kaggle), because a training job is finite and fits a submit/poll/download state machine. Serving deliberately lives outside that machinery — a long-running kernel never reaches "complete," so it's a separate, manually-triggered script (`deploy_serve.py`) rather than a queued job.

## Real bugs found and fixed


1. **EOS masking bug.** `tokenizer.pad_token` was set to `tokenizer.eos_token`, and `DataCollatorForLanguageModeling` masks loss by matching `pad_token_id` — so every genuine end-of-response EOS signal got masked out alongside real padding. The model never learned to stop generating. Fixed by masking via `attention_mask` position instead of token-ID equality, and switching to `default_data_collator`.
2. **Full-sequence loss bug.** `labels = input_ids.copy()` trained the model to predict the prompt and chat-template boilerplate, not just the response, diluting the loss signal. Fixed by masking the prompt+template prefix to `-100`.

Plus a chain of Kaggle environment bugs (documented inline in `train.py`): a Pascal-generation P100 GPU being silently assigned and incompatible with the preinstalled CUDA 12.8 torch build, `dataset_sources` mounting three directories deep instead of one, and `kernel_secrets` not being a real pushable field (worked around with a private Kaggle Dataset holding plaintext secret files).

## Results

### Training length (100 steps vs. 1 epoch / 493 steps)

| category | 100 steps | 500 steps |
|---|---|---|
| math_reasoning | 100% | 100% |
| logical_reasoning | 80% | 100% |
| code_debugging | 40% | 100% |
| classification | 60% | 100% |
| planning | 20% | 60% |
| instruction_following | 80% | 80% |
| **overall (30 prompts)** | **63%** | **90%** |

"Compliant" = correct `THOUGHT:`/`ANSWER:` structure (or bare output for instruction_following) AND a clean stop (not cut off at the token cap). Final training loss: 1.87 (buggy run) → 1.4 (100 steps, bugs fixed) → 1.30 (500 steps).

### Concurrency (500-step adapter, vLLM on a single T4)

| concurrency | total tok/s | per-stream tok/s | p50 latency | p95 latency |
|---|---|---|---|---|
| 1 | 11.5 | 10.7 | 8.1s | 15.6s |
| 4 | 41.7 | 10.0 | 9.1s | 15.1s |
| 8 | 81.9 | 9.9 | 9.1s | 15.7s |
| 16 | 124.3 | 9.0 | 10.0s | 16.9s |

~10.8x total throughput at 16x concurrency, zero errors across 128 requests. Per-stream speed only drops ~16%, consistent with vLLM's continuous batching doing its job.

## Known limitations

- **Planning is the weakest category** (60% compliant): still loops on long, itemized answers. Likely needs either more planning-specific training data or a wider LoRA target (more layers), not yet tried.
- **instruction_following leaks reasoning 20% of the time**, unchanged between 100 and 500 steps — points at a data/format issue in that category's training examples rather than undertraining.
- **Eval sample is small** (30 hand-written prompts, 5 per category), so percentages carry meaningful noise.
- **Not a true held-out set** — all prompt categories were represented in training data, though the eval prompts themselves are new and hand-written, not drawn from the training file.
- A dataset dedup-rate discrepancy (information_extraction/question_answering keeping fewer records than other categories) was flagged early and never root-caused.
- `kaggle_backend.py` has a known-harmless leftover debug block (noisy but non-fatal `AttributeError`s on every submit) not yet cleaned up.

## Stack

PyTorch, PEFT, Transformers (pinned to a Pascal-GPU-compatible build), Docker Compose, Redis, Ray, MLflow + Postgres, Azure Blob Storage, vLLM, `pycloudflared`, Kaggle API.

## Repo layout

```
v2/
├── datasets/           # seeds → generated → judged → final (986 records)
├── models/adapter_output/   # trained LoRA adapter
├── scripts/             # dataset generation/judging pipeline
├── services/
│   ├── gateway/          # job intake API
│   └── worker/
│       ├── kaggle_scripts/   # train.py, app.py (run ON Kaggle)
│       └── services/         # kaggle_backend.py, training orchestration (run in the worker container)
├── benchmarks/           # eval_bench.py + results/
└── docker-compose.yml
```