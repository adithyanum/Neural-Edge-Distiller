"""
run_judge.py

LLM-judge pass over generated synthetic dataset records.

For each record in v2/datasets/generated/{category}.jsonl, calls a Gemini
judge model to produce a pass/fail verdict, then writes the original record
plus the verdict to v2/datasets/judged/{category}.jsonl.

Resumable: records already judged (by id) in the output file are skipped.
Raw generated files are never modified.

Usage:
    export GEMINI_API_KEY=...
    python run_judge.py                     # judge all categories
    python run_judge.py --category math_reasoning
    python run_judge.py --model gemini-3.5-flash-lite --rpm 12
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from pydantic_settings import BaseSettings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("judge")

# This file lives at <repo_root>/scripts/run_judge.py, so parent.parent is
# the repo root regardless of the directory this script is invoked from.
ROOT = Path(__file__).resolve().parent.parent
GENERATED_DIR = ROOT / "datasets" / "generated"
JUDGED_DIR = ROOT / "datasets" / "judged"


class JudgeSettings(BaseSettings):
    """Standalone settings for this script only. Deliberately does NOT reuse
    services/gateway/config.py's Settings: that class requires postgres/redis/
    kaggle fields this script has no business depending on, and its env_file
    path is resolved relative to the *working directory* the process is
    launched from (fine inside Docker's fixed workdir, not fine for a script
    you might run from repo root, scripts/, or anywhere else).

    Pointing env_file at an absolute path derived from this file's own
    location means `python run_judge.py` works the same regardless of cwd.
    """

    gemini_api_key: str

    class Config:
        env_file = str(ROOT / ".env")
        extra = "ignore"  # don't choke on postgres/kaggle/etc keys sharing the file

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

CODE_CATEGORIES = {"code_generation", "code_debugging"}

DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_RPM = 10         
MAX_RETRIES = 4
BASE_BACKOFF_SECONDS = 5

# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_INSTRUCTIONS = """You are a strict but fair data-quality judge for a supervised \
fine-tuning dataset. You will be shown one training example: a category, a \
difficulty level, a prompt, and a model-generated response. Decide whether \
this example is GOOD ENOUGH to keep in a training set.

Judge on these things:
1. ON-TASK: does the response actually address what the prompt asks, rather \
than answering a different or generic question?
2. CORRECT / COHERENT: is the response factually/logically correct for the \
prompt, well-formed, and free of nonsense, truncation, or placeholder text \
(e.g. a bare label like "medium" instead of real content)?
3. COMPLETE: does the response look finished, not cut off mid-sentence or \
mid-structure?
4. REASONING: most responses in this dataset are expected to contain a \
THOUGHT section (numbered reasoning steps) followed by an ANSWER section. \
If the response has this structure, judge whether the THOUGHT steps are \
genuine reasoning that actually leads to the ANSWER — not filler, not a \
restatement of the prompt, not disconnected from the final answer. A \
THOUGHT section that exists but doesn't meaningfully reason toward the \
answer should FAIL this check even though the format looks right.
   EXCEPTION: if the prompt explicitly demands an exact, unpolluted output \
(phrases like "output only the word X", "respond with only", "no other \
text", "no explanation", "nothing else"), then the response should NOT \
contain a THOUGHT section at all — it should be the bare compliant answer \
only. In that case, a leaked THOUGHT section is a FAIL, and a clean bare \
answer is correct (do not penalize it for lacking reasoning).
5. If the response has NO THOUGHT section at all, and the prompt does NOT \
demand exact/bare output, that is also a FAIL — this dataset should not \
contain bare, unreasoned answers except where the prompt specifically \
requires them.

{code_note}

Respond with ONLY a single JSON object, no markdown fences, no preamble:
{{"verdict": "pass" or "fail", "reason": "<one short sentence, under 20 words>"}}

Category: {category}
Difficulty: {difficulty}
{context_block}
Prompt:
{prompt}

Response:
{response}
"""

CODE_NOTE = """For code, you cannot execute it. Judge plausibility instead: does \
the code's logic match what was asked, is it syntactically well-formed, and \
free of obvious errors (undefined names, mismatched brackets, wrong operator)? \
Do not require perfect style, only correctness of approach."""


def build_judge_prompt(record: dict) -> str:
    category = record.get("category", "unknown")
    code_note = CODE_NOTE if category in CODE_CATEGORIES else ""

    context = record.get("context")
    context_block = f"\nContext:\n{context}\n" if context else ""

    return JUDGE_INSTRUCTIONS.format(
        code_note=code_note,
        category=category,
        difficulty=record.get("difficulty", "unknown"),
        context_block=context_block,
        prompt=record.get("prompt", ""),
        response=record.get("response", ""),
    )


# ---------------------------------------------------------------------------
# Gemini call
# ---------------------------------------------------------------------------

def get_gemini_client():
    """Lazily imports and configures the Gemini SDK. Raises a clear error
    if the package or API key is missing, instead of failing deep in a loop."""
    try:
        import google.generativeai as genai
    except ImportError:
        logger.error(
            "google-generativeai not installed. Run: "
            "pip install google-generativeai --break-system-packages"
        )
        sys.exit(1)

    try:
        settings = JudgeSettings()
    except Exception as e:
        logger.error(
            "Could not load GEMINI_API_KEY (checked %s and real env vars): %s",
            ROOT / ".env", e,
        )
        sys.exit(1)

    genai.configure(api_key=settings.gemini_api_key)
    return genai


def parse_verdict_json(text: str) -> dict:
    """Parses exactly one JSON object out of the model's raw text response,
    same raw_decode approach used in generation_engine.py to survive any
    trailing junk the model tacks on."""
    text = text.strip()
    # strip accidental markdown fences defensively
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(text)
    return obj


def call_judge(genai_module, model_name: str, prompt: str) -> dict:
    model = genai_module.GenerativeModel(model_name)
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = model.generate_content(
                prompt,
                generation_config={"temperature": 0.0, "max_output_tokens": 200},
            )
            raw_text = resp.text
            verdict = parse_verdict_json(raw_text)
            if verdict.get("verdict") not in ("pass", "fail"):
                raise ValueError(f"bad verdict field: {verdict}")
            return verdict
        except Exception as e:
            last_err = e
            is_rate_limit = "429" in str(e) or "quota" in str(e).lower()
            wait = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "Judge call failed (attempt %d/%d)%s: %s. Retrying in %ds.",
                attempt, MAX_RETRIES,
                " [rate limit]" if is_rate_limit else "",
                e, wait,
            )
            time.sleep(wait)
    # All retries exhausted: fail closed with a visible marker rather than
    # silently dropping the record or crashing the whole run.
    logger.error("Judge call permanently failed: %s", last_err)
    return {"verdict": "fail", "reason": f"judge_error: {last_err}"}


# ---------------------------------------------------------------------------
# Per-category driver
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


def load_already_judged_ids(judged_path: Path) -> set:
    return {r["id"] for r in load_jsonl(judged_path) if "id" in r}


def judge_category(genai_module, category: str, model_name: str, rpm: int):
    src_path = GENERATED_DIR / f"{category}.jsonl"
    out_path = JUDGED_DIR / f"{category}.jsonl"
    JUDGED_DIR.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(src_path)
    if not records:
        logger.warning("No generated records found for %s at %s", category, src_path)
        return

    already_done = load_already_judged_ids(out_path)
    pending = [r for r in records if r.get("id") not in already_done]

    logger.info(
        "[%s] %d total records, %d already judged, %d pending",
        category, len(records), len(already_done), len(pending),
    )

    if not pending:
        return

    min_interval = 60.0 / rpm
    pass_count = 0
    fail_count = 0

    with open(out_path, "a") as out_f:
        for i, record in enumerate(pending, 1):
            start = time.time()
            prompt = build_judge_prompt(record)
            verdict = call_judge(genai_module, model_name, prompt)

            judged_record = dict(record)
            judged_record["judge_verdict"] = verdict
            out_f.write(json.dumps(judged_record) + "\n")
            out_f.flush()

            if verdict["verdict"] == "pass":
                pass_count += 1
            else:
                fail_count += 1
                logger.info(
                    "[%s] FAIL %s: %s", category, record.get("id"), verdict.get("reason")
                )

            if i % 10 == 0 or i == len(pending):
                logger.info(
                    "[%s] %d/%d judged (pass=%d fail=%d)",
                    category, i, len(pending), pass_count, fail_count,
                )

            elapsed = time.time() - start
            sleep_for = max(0.0, min_interval - elapsed)
            if sleep_for > 0 and i < len(pending):
                time.sleep(sleep_for)

    logger.info(
        "[%s] DONE. pass=%d fail=%d rate=%.1f%%",
        category, pass_count, fail_count,
        100.0 * pass_count / max(1, pass_count + fail_count),
    )


def main():
    parser = argparse.ArgumentParser(description="Run LLM judge over generated dataset.")
    parser.add_argument(
        "--category", choices=ALL_CATEGORIES, default=None,
        help="Judge a single category. Default: all categories.",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Gemini model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--rpm", type=int, default=DEFAULT_RPM,
        help=f"Requests per minute cap, self-throttled (default: {DEFAULT_RPM})",
    )
    args = parser.parse_args()

    genai_module = get_gemini_client()

    categories = [args.category] if args.category else ALL_CATEGORIES
    for category in categories:
        judge_category(genai_module, category, args.model, args.rpm)


if __name__ == "__main__":
    main()