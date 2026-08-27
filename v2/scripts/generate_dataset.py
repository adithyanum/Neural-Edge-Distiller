import json
from pathlib import Path

from mlx_lm import load

from generation_engine import run_generation, ALL_CATEGORIES


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"

SEEDS_DIR = Path("v2/datasets/seeds")

GENERATED_DIR = Path("v2/datasets/generated")


# ============================================================
# Seed Loader
# ============================================================

def load_seeds_for_category(category: str):
    """
    Seeds live as one JSONL file per category
    (one JSON object per line), not a single seeds.json.
    """

    path = SEEDS_DIR / f"{category}.jsonl"

    seeds = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            seeds.append(json.loads(line))

    print(f"Loaded {len(seeds)} seed prompts for '{category}'.")

    return seeds


# ============================================================
# Model Loader
# ============================================================

def load_teacher():

    print()

    print("=" * 60)
    print("Loading Teacher Model...")
    print("=" * 60)

    model, tokenizer = load(MODEL_NAME)

    print("Teacher Loaded.")

    print()

    return model, tokenizer


# ============================================================
# Main
# ============================================================

def main():

    model, tokenizer = load_teacher()

    GENERATED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for category in sorted(ALL_CATEGORIES):
 
        print()
        print("=" * 60)
        print(f"Category: {category}")
        print("=" * 60)
 
        seeds = load_seeds_for_category(category)
 
        output_path = GENERATED_DIR / f"{category}.jsonl"
 
        run_generation(
            model=model,
            tokenizer=tokenizer,
            seeds=seeds,
            output_path=output_path,
        )
        
    print()

    print("=" * 60)
    print("Dataset generation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()