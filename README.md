# Neural Edge Distiller

A chain-of-thought knowledge distillation project for Llama-3.2-3B, built twice: once as a fast local proof of concept, once as a real distributed ML platform.

The goal in both versions is the same — fine-tune a small model with LoRA so it reliably produces structured `THOUGHT:` → `ANSWER:` reasoning on tasks that need it, and bare output on tasks that don't. What changed between versions is everything around that goal: how training runs, where it runs, how results are stored, and how the model gets served.

## Versions

### [`v1/`](./v1) — local proof of concept

Trained and served entirely on Apple Silicon using MLX. Proves the distillation idea works: a quantized 3B model learns the CoT format via LoRA, demoed through a local Streamlit app that runs the base model and the fine-tuned adapter side by side on the same prompt, with live latency/throughput/structure-compliance metrics.

Fast to iterate on, zero cloud infrastructure, but single-machine and not representative of how a model would actually get trained or served in production.

### [`v2/`](./v2) — distributed platform

Same underlying idea, rebuilt as an actual pipeline: a job-queue architecture (gateway → Redis → Ray worker) submits real training jobs to Kaggle's free GPU tier, logs metrics to MLflow, stores the resulting adapter in Azure Blob Storage, and serves it from a separate Kaggle kernel running vLLM with native LoRA support, tunneled out via Cloudflare — all without any paid infrastructure.

V2 is where the real engineering happened: two genuine training-math bugs were found and fixed by reading actual model output rather than trusting the loss curve, a chain of Kaggle/CUDA environment incompatibilities got debugged and documented, and the final model was evaluated with a real compliance benchmark (90% structural compliance vs. 63% at 1/5 the training length) and a concurrency benchmark (~10.8x throughput at 16x concurrent requests on a single free-tier GPU).

See [`v2/README.md`](./v2/README.md) for the full architecture, the bugs found, and the results.

## Why two versions

V1 answers "does this work at all." V2 answers "can I build the infrastructure a real team would use to train, store, and serve this" — on a student budget, with no credit card and no paid compute. The gap between the two is most of the actual learning: job orchestration, environment debugging on someone else's GPU, separating finite training jobs from long-running serving processes, and verifying a model's behavior with evidence instead of a single number.

## Stack summary

| | V1 | V2 |
|---|---|---|
| Training | MLX, local Apple Silicon | PyTorch + PEFT, Kaggle T4 (cloud) |
| Orchestration | none (manual) | Docker Compose: gateway, Redis, Ray worker |
| Experiment tracking | none | MLflow + Postgres |
| Storage | local filesystem | Azure Blob Storage |
| Serving | local Streamlit demo | vLLM + LoRA on Kaggle, Cloudflare tunnel |
| Evaluation | manual side-by-side comparison | scripted eval (compliance scoring) + concurrency benchmark |