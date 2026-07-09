import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import mlx.core as mx
from mlx_lm import load, generate
from scripts.utils import has_structure, format_prompt, clean_response, get_sampler

TEST_PROMPTS = [
    "Kafka consumer lag spiking to 48 hours under peak load with strict per-user event ordering. Recover throughput without violating ordering constraints.",
    "P99 latency on LLM inference endpoint spiking to 30s under concurrent load. Reduce latency without scaling GPU count.",
    "A single malformed event repeatedly crashing Kafka consumers and blocking the entire partition from progressing. Isolate and handle poison pill messages without manual intervention or partition stall."
]


def run_benchmark(model_path, label):
    print(f"\n{'='*60}")
    print(f"  MODEL: {label}")
    print(f"  PATH:  {model_path}")
    print(f"{'='*60}")

    model, tokenizer = load(model_path)
    sampler = get_sampler()

    total_tokens = 0
    total_time = 0
    responses = []

    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"\n── Prompt {i+1} ──────────────────────────────────────────")
        print(f"Q: {prompt}\n")

        start = time.time()
        response = generate(model, tokenizer, prompt=format_prompt(tokenizer, prompt), max_tokens=500, sampler=sampler)
        elapsed = time.time() - start

        response = clean_response(response)
        token_count = len(response.split())
        tps = token_count / elapsed
        total_tokens += token_count
        total_time += elapsed
        responses.append(response)

        print(response)
        structured = "✅ Structured" if has_structure(response) else "❌ Unstructured"
        print(f"\n⏱  Time: {elapsed:.2f}s | Tokens/sec: {tps:.1f} | Format: {structured}")

    peak_mem = mx.device_info()["max_recommended_working_set_size"] / 1e9
    avg_tps = total_tokens / total_time
    structured_count = sum(has_structure(r) for r in responses)

    print(f"\n{'='*60}")
    print(f"  SUMMARY — {label}")
    print(f"  Avg Tokens/sec      : {avg_tps:.1f}")
    print(f"  Peak Memory         : {peak_mem:.2f} GB")
    print(f"  Structured responses: {structured_count}/{len(responses)}")
    print(f"{'='*60}\n")

    return avg_tps, peak_mem, structured_count, responses


if __name__ == "__main__":
    print("\n🔬 NEURAL EDGE DISTILLER — Before/After Benchmark")
    print("Comparing vanilla 3B vs fine-tuned Neural Edge 3B\n")

    vanilla_tps, vanilla_mem, vanilla_struct, vanilla_responses = run_benchmark(
        "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "Vanilla Llama 3.2 3B"
    )

    finetuned_tps, finetuned_mem, finetuned_struct, finetuned_responses = run_benchmark(
        "models/neural-edge-3b",
        "Neural Edge 3B (Fine-tuned)"
    )

    print("\n" + "="*60)
    print("  FINAL COMPARISON")
    print("="*60)
    print(f"  {'Metric':<28} {'Vanilla 3B':>12} {'Neural Edge':>12}")
    print(f"  {'-'*54}")
    print(f"  {'Avg Tokens/sec':<28} {vanilla_tps:>11.1f} {finetuned_tps:>11.1f}")
    print(f"  {'Peak Memory (GB)':<28} {vanilla_mem:>11.2f} {finetuned_mem:>11.2f}")
    print(f"  {'Structured responses':<28} {f'{vanilla_struct}/3':>12} {f'{finetuned_struct}/3':>12}")
    print("="*60)

    print("\n📊 FORMAT QUALITY PER PROMPT")
    print("-"*60)
    for i in range(len(TEST_PROMPTS)):
        v = "✅" if has_structure(vanilla_responses[i]) else "❌"
        f = "✅" if has_structure(finetuned_responses[i]) else "❌"
        print(f"  Prompt {i+1}: Vanilla={v}  Neural Edge={f}")
    print("-"*60)
    print()