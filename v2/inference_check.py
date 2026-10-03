"""
Neural Edge Distiller V2 - inference spot-check.

Loads the base model + the trained LoRA adapter WITHOUT merging (PeftModel
wraps the base at inference time), and runs a handful of prompts spanning
different categories to eyeball whether the model actually learned the
THOUGHT:/ANSWER: CoT structure the dataset was built around.

Run locally (Apple Silicon: uses MPS if available, falls back to CPU) or on
any CUDA machine - device is auto-detected.

Usage:
    python inference_check.py --adapter-path /path/to/adapter_output

Requires: transformers, peft, torch (whatever's already in distiller_env
from earlier work should cover this - no Kaggle-specific pins needed here,
those were only for the P100's Pascal architecture on Kaggle's kernel image).
"""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

# One or two hand-picked prompts per category, deliberately NOT verbatim
# copies of training records - close enough to the domain to be a fair test,
# different enough to check generalization rather than memorization.
SPOT_CHECK_PROMPTS = [
    ("math_reasoning", "A train travels 240 miles in 4 hours, then 180 miles in 3 hours. What is its average speed for the whole trip?"),
    ("logical_reasoning", "All engineers at this company know Python. Maria works at this company and knows Python. Can we conclude Maria is an engineer? Explain."),
    ("code_debugging", "This function is supposed to return the sum of a list but always returns 0:\ndef total(nums):\n    s = 0\n    for n in nums:\n        s = n\n    return s"),
    ("classification", "Classify the sentiment of this review: 'The battery life is decent but the screen is disappointingly dim outdoors.'"),
    ("planning", "I need to migrate a production database with zero downtime. Outline the key steps."),
    ("instruction_following", "Output only the word DONE and nothing else."),
]


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", required=True, help="Path to the adapter_output directory")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="HF access token for the gated Llama-3.2 checkpoint. Falls back to "
             "HF_TOKEN env var. If neither is set, falls back to a cached "
             "`huggingface-cli login` session if you've already run that.",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # bf16 is fine on MPS/CUDA for inference (no training-time numerical
    # stability concerns here); fp32 fallback keeps CPU-only runs correct.
    dtype = torch.bfloat16 if device != "cpu" else torch.float32

    print(f"Loading base model: {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)  # adapter dir has the saved tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=args.hf_token,
        torch_dtype=dtype,
    ).to(device)

    print(f"Loading adapter from: {args.adapter_path} ...")
    model = PeftModel.from_pretrained(base_model, args.adapter_path).to(device)
    model.eval()

    print("\n" + "=" * 70)
    print("INFERENCE SPOT-CHECK")
    print("=" * 70)

    for category, prompt in SPOT_CHECK_PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,  # deterministic, easier to eyeball/compare across runs
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        print(f"\n--- [{category}] ---")
        print(f"PROMPT: {prompt}")
        print(f"RESPONSE:\n{response}")
        print("-" * 70)

    print("\nDone. Check above: does each response (except instruction_following,")
    print("which should be a bare compliant answer) show a THOUGHT: section before")
    print("ANSWER:, matching the dataset's CoT structure?")


if __name__ == "__main__":
    main()