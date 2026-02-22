import json
from mlx_lm import load
from generate_curriculum import generate_architecture
from tqdm import tqdm

# 1. Load Model ONCE
model_path = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
model, tokenizer = load(model_path)

# 2. Load Scenarios
with open("data/raw/curriculum_goals.json", "r") as f:
    scenarios = json.load(f)["scenarios"]

print(f"Loaded {len(scenarios)} scenarios. Starting generation...\n")

# 3. The Factory Loop
generated = 0
skipped = 0

with open("data/training/synthetic_distillation.jsonl", "w") as out_f:
    for item in tqdm(scenarios, desc="Generating"):

        wisdom = generate_architecture(
            model, tokenizer,
            item['title'],
            item['description']
        )

        # Only structural check — is the output complete?
        if not wisdom or "END_OF_ARCH" not in wisdom:
            print(f"\n[SKIP] Malformed output for: {item['title']}")
            skipped += 1
            continue

        entry = {
            "instruction": item['description'],
            "response": wisdom
        }

        out_f.write(json.dumps(entry) + "\n")
        generated += 1

print(f"\nDone. Generated: {generated} | Skipped: {skipped}")
print(f"Output: data/training/synthetic_distillation.jsonl")