"""
assemble_final.py

Builds the final training dataset from two trusted sources:
  1. datasets/judged/{category}.jsonl records where judge_verdict.verdict
     == "pass" (teacher-generated, judge-approved)
  2. datasets/seeds/{category}.jsonl records (handwritten prompts, answered
     with the CoT few-shot rewrite — trusted without going through the
     Gemini judge, since these were generated/reviewed directly rather
     than by the teacher model)

Within each category, near-duplicate prompts are removed using semantic
similarity (sentence-transformers embeddings + FAISS), keeping seeds over
generated records when they collide, since seeds are the higher-trust
source. judge_verdict is stripped from the output — it did its job during
filtering, no reason to carry judge metadata into training data.

Output: datasets/final/{category}.jsonl, plus datasets/final/all.jsonl
(all categories concatenated, for a single-file training run).

Usage:
    python assemble_final.py                      # all categories
    python assemble_final.py --category planning
    python assemble_final.py --threshold 0.90      # looser dedup
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("assemble_final")

ROOT = Path(__file__).resolve().parent.parent
JUDGED_DIR = ROOT / "datasets" / "judged"
SEEDS_DIR = ROOT / "datasets" / "seeds"
FINAL_DIR = ROOT / "datasets" / "final"

ALL_CATEGORIES = [
    "math_reasoning",
    "logical_reasoning",
    "code_generation",
    "code_debugging",
    "data_transformation",
    "information_extraction",
    "classification",
    "planning",
    "question_answering",
    "instruction_following",
]

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.92  # cosine similarity above this = treated as duplicate


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_category_sources(category: str) -> list:
    """
    Returns the merged, pre-dedup list for one category: all answered
    seeds first (higher trust, kept on collision), then judged records
    that passed. judge_verdict is stripped here since it's not part of
    the final training schema.
    """
    seed_records = load_jsonl(SEEDS_DIR / f"{category}.jsonl")

    judged_records = load_jsonl(JUDGED_DIR / f"{category}.jsonl")
    passed = []
    for r in judged_records:
        verdict = r.get("judge_verdict", {})
        if verdict.get("verdict") == "pass":
            clean = dict(r)
            clean.pop("judge_verdict", None)
            passed.append(clean)

    logger.info(
        "[%s] %d seeds + %d judge-passed generated = %d candidates before dedup",
        category, len(seed_records), len(passed), len(seed_records) + len(passed),
    )

    return seed_records + passed


# ---------------------------------------------------------------------------
# Semantic dedup
# ---------------------------------------------------------------------------

def get_embedder(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error(
            "sentence-transformers not installed. Run: "
            "pip install sentence-transformers faiss-cpu --break-system-packages"
        )
        raise
    return SentenceTransformer(model_name)


def record_text(record: dict) -> str:
    """What gets embedded for similarity comparison — context + prompt,
    matching what the judge and generator both treat as 'the task'."""
    parts = [record.get("context"), record.get("prompt")]
    return " ".join(p for p in parts if p)


def dedup_semantic(records: list, embedder, threshold: float) -> tuple:
    """
    Incremental dedup: for each record in order, embed it and search
    against an index of everything already kept. If the closest existing
    match is above `threshold` cosine similarity, drop this record as a
    duplicate. Otherwise keep it and add it to the index.

    Order matters: records list is seeds-first, so when a generated
    record duplicates a seed, the seed (added first) wins and the
    generated duplicate is the one dropped.

    Returns (kept_records, dropped_count).
    """
    import faiss
    import numpy as np

    if not records:
        return [], 0

    texts = [record_text(r) for r in records]
    embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.asarray(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine similarity

    kept = []
    dropped = 0

    for i, record in enumerate(records):
        vec = embeddings[i:i + 1]

        if index.ntotal > 0:
            similarities, _ = index.search(vec, 1)
            top_similarity = float(similarities[0][0])
        else:
            top_similarity = -1.0  # nothing to compare against yet

        if top_similarity >= threshold:
            dropped += 1
            continue

        index.add(vec)
        kept.append(record)

    return kept, dropped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def assemble_category(category: str, embedder, threshold: float) -> list:
    candidates = load_category_sources(category)
    kept, dropped = dedup_semantic(candidates, embedder, threshold)

    logger.info(
        "[%s] %d candidates -> %d kept, %d dropped as near-duplicates",
        category, len(candidates), len(kept), dropped,
    )

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FINAL_DIR / f"{category}.jsonl"
    with open(out_path, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    return kept


def main():
    parser = argparse.ArgumentParser(description="Assemble the final training dataset.")
    parser.add_argument("--category", choices=ALL_CATEGORIES, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    embedder = get_embedder(args.model)

    categories = [args.category] if args.category else ALL_CATEGORIES

    all_records = []
    for category in categories:
        kept = assemble_category(category, embedder, args.threshold)
        all_records.extend(kept)

    if not args.category:
        all_path = FINAL_DIR / "all.jsonl"
        with open(all_path, "w") as f:
            for r in all_records:
                f.write(json.dumps(r) + "\n")
        logger.info("Wrote combined final/all.jsonl: %d total records", len(all_records))

    logger.info("Final assembly complete.")


if __name__ == "__main__":
    main()