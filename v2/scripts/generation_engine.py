import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("dataset-generator")


# ============================================================
# Category Definitions
# ============================================================

PROMPT_ONLY_CATEGORIES = {
    "logical_reasoning",
    "math_reasoning",
    "code_generation",
    "code_debugging",
    "planning",
}

FULL_EXAMPLE_CATEGORIES = {
    "information_extraction",
    "instruction_following",
    "question_answering",
    "data_transformation",
    "classification",
}

ALL_CATEGORIES = (
    PROMPT_ONLY_CATEGORIES |
    FULL_EXAMPLE_CATEGORIES
)


# ============================================================
# Generation Parameters
# ============================================================

TEMPERATURE = 0.8
TOP_P = 0.95
MAX_TOKENS = 768
MAX_RETRIES = 3


# ============================================================
# Difficulty Progression
# ============================================================

DIFFICULTY_MAP = {
    "easy": "medium",
    "medium": "hard",
    "hard": "hard"
}


# ============================================================
# Duplicate Cache
# ============================================================

_seen_prompts: Set[str] = set()


# ============================================================
# Cleaning Helpers
# ============================================================

SPECIAL_TOKENS = [
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|eot_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
]


def clean_generation(text: str) -> str:
    """
    Remove chat tokens and markdown fences.
    """

    if not text:
        return ""

    for token in SPECIAL_TOKENS:
        text = text.replace(token, "")

    text = text.replace("```json", "")
    text = text.replace("```", "")

    return text.strip()


# ============================================================
# JSON Extraction
# ============================================================

def extract_json(text: str) -> Optional[str]:
    """
    Try JSON_START marker first.
    Fallback to regex.
    """

    text = clean_generation(text)

    if "<JSON_START>" in text and "<JSON_END>" in text:

        start = text.index("<JSON_START>") + len("<JSON_START>")
        end = text.index("<JSON_END>")

        return text[start:end].strip()

    match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL
    )

    if match:
        return match.group()

    return None


def parse_json(text: str):

    blob = extract_json(text)

    if blob is None:
        raise ValueError("No JSON found.")

    # raw_decode parses exactly one JSON value starting at position 0
    # and ignores anything after it — unlike json.loads, it won't
    # fail with "Extra data" if the model trails off with a second
    # JSON block, a markdown fence, or leftover hallucinated text.
    decoder = json.JSONDecoder()

    obj, _ = decoder.raw_decode(blob.strip())

    return obj


# ============================================================
# Duplicate Detection
# ============================================================

def is_duplicate(prompt: str) -> bool:

    normalized = prompt.lower().strip()

    if normalized in _seen_prompts:
        return True

    _seen_prompts.add(normalized)

    return False


# ============================================================
# Difficulty Helpers
# ============================================================

def next_difficulty(level: str) -> str:

    level = level.lower()

    return DIFFICULTY_MAP.get(level, "medium")


# ============================================================
# Validation
# ============================================================

REQUIRED_FIELDS = [
    "instruction",
    "answer",
]


def validate_entry(entry: Dict) -> bool:
    """
    Basic validation.
    """

    for field in REQUIRED_FIELDS:

        if field not in entry:
            return False

        if not entry[field]:
            return False

    return True


def validate_prompt(prompt: str) -> bool:

    if len(prompt.strip()) < 15:
        return False

    if len(prompt.split()) < 5:
        return False

    return True


# ============================================================
# Writer
# ============================================================

def write_jsonl(path: Path, item: Dict):

    with open(path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                item,
                ensure_ascii=False
            )
        )
        f.write("\n")


# ============================================================
# Chat Wrapper
# ============================================================

def chat(
    model,
    tokenizer,
    prompt: str,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
):

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    sampler = make_sampler(
        temp=temperature,
        top_p=TOP_P,
    )

    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=formatted,
        max_tokens=max_tokens,
        sampler=sampler,
    )

    # Truncate at the first turn boundary BEFORE stripping tokens.
    # generate() sometimes keeps going past the real answer and
    # hallucinates a fake follow-up turn. clean_generation() alone
    # would just delete the <|eot_id|> marker and glue that fake
    # turn directly onto the real answer with no boundary at all.
    response = response.split("<|eot_id|>")[0]

    return clean_generation(response)


# ============================================================
# Retry Wrapper
# ============================================================

def chat_json(
    model,
    tokenizer,
    prompt: str,
):

    for attempt in range(MAX_RETRIES):

        try:

            output = chat(
                model,
                tokenizer,
                prompt,
            )

            return parse_json(output)

        except Exception as e:

            logger.warning(
                f"Attempt {attempt+1}/{MAX_RETRIES} failed: {e}"
            )

    raise RuntimeError("JSON generation failed.")

# ============================================================
# Prompt Variation Definitions
# ============================================================

VARIATION_TYPES = {
    "difficulty": (
        "Increase the reasoning complexity while preserving the original task."
    ),
    "context": (
        "Place the task into a completely different realistic scenario."
    ),
    "constraint": (
        "Add one or more meaningful constraints that require additional reasoning."
    ),
    "perspective": (
        "Rewrite the task from another person's perspective while preserving the objective."
    ),
}


# ============================================================
# Prompt Builder (Prompt-only Categories)
# ============================================================

def build_prompt_only_prompt(seed):
    """
    Categories:
        - Code Generation
        - Debugging
        - Logical Reasoning
        - Mathematical Reasoning
        - Planning
    """

    category = seed["category"]

    is_code = category in {"code_generation", "code_debugging"}

    if is_code:
        example = """{
    "difficulty": "Debug the following Python function. It is meant to return True for even numbers and False for odd numbers, but currently returns the wrong result for negative inputs.\\n\\ndef is_even(n):\\n    return n % 2 == 1\\n\\nFind and fix the bug.",
    "context": "A logging system calls this function on every incoming request. Debug it so it correctly classifies numbers as even or odd, including negative numbers and zero.\\n\\ndef is_even(n):\\n    return n % 2 == 1",
    "constraint": "Debug the following function without using the modulo operator anywhere in your fix.\\n\\ndef is_even(n):\\n    return n % 2 == 1",
    "perspective": "You are reviewing a junior developer's pull request. Explain in plain language what is wrong with this function, then provide the corrected version.\\n\\ndef is_even(n):\\n    return n % 2 == 1"
}"""
    else:
        example = """{
    "difficulty": "A train travels 240 km in 3 hours, then increases its speed by 20 km/h for the remaining 180 km of the journey. Calculate the total travel time, then determine the average speed for the entire trip.",
    "context": "A warehouse manager needs to know how many boxes fit on a shelf. Each box is 40 cm wide and the shelf is 3.2 meters long. How many boxes fit, and how much space is left over?",
    "constraint": "Solve the following using only mental math, without writing anything down or using a calculator: what is 15 percent of 240?",
    "perspective": "As a teacher explaining a concept to a student for the first time, walk through how to determine whether 91 is a prime number, showing every step of your reasoning."
}"""

    return f"""
You are creating benchmark prompts for training a reasoning language model.

Seed Prompt

Category:
{category}

Difficulty:
{seed["difficulty"]}

Instruction:
{seed["prompt"]}

Generate EXACTLY FOUR variations of the seed task above.

Variation 1 (key "difficulty"): a harder version of the same task.
Variation 2 (key "context"): the same task, moved into a new realistic scenario.
Variation 3 (key "constraint"): the same task, with a meaningful added restriction.
Variation 4 (key "perspective"): the same task, framed from a different viewpoint.

CRITICAL RULES

- Each of the four JSON values must be a COMPLETE, STANDALONE task — something
  a model could receive with NO other context and immediately attempt to solve.
- Do NOT output single words, labels, or short descriptive phrases such as
  "medium", "a junior developer", or "the function should work for positive
  integers". Those describe what changed; they are NOT the task itself.
- The key names ("difficulty", "context", "constraint", "perspective")
  describe HOW that variation differs from the seed — they are NOT
  instructions to output a label for that concept.
- Every value must be several full sentences. For code tasks, embed the
  actual code snippet inside the string (use \\n for newlines) plus a
  clear instruction of what to do with it.
- Preserve the original task's core subject matter.
- Make every one of the four variations meaningfully different from each other.
- Never explain your reasoning. Do NOT generate answers here.
- Output ONLY JSON, matching the exact shape and keys shown below.

Example of the CORRECT shape and completeness level (write NEW content,
do not reuse this example's subject matter):

<JSON_START>

{example}

<JSON_END>

Now generate the JSON for the seed task above.
"""


# ============================================================
# Full Example Prompt
# ============================================================

def build_full_example_prompt(seed):
    """
    Used for:

    Information Extraction
    QA
    Creative Writing
    Data Transformation
    Instruction Following

    Teacher creates BOTH
    context + instruction + answer
    """

    return f"""
You are creating synthetic benchmark examples.

Seed Category

{seed["category"]}

Seed Instruction

{seed["prompt"]}

Generate FOUR completely new examples.

Example 1
Increase difficulty.

Example 2
Different realistic context.

Example 3
Extra constraints.

Example 4
Different perspective.

Each example MUST contain

- context
- instruction
- answer

The "answer" field must show reasoning before the final answer, using
EXACTLY this structure inside the string:

THOUGHT:
1. <first reasoning step>
2. <continue reasoning toward the correct answer>
(as many steps as genuinely needed — usually 2-5)
ANSWER: <the final, clean answer — nothing after this line>

Rules

- Everything must be self-contained.
- Never rely on hidden information.
- Never hallucinate missing context.
- Produce realistic data.
- "answer" MUST contain both a THOUGHT section and an ANSWER section, in
  that order. A bare answer with no THOUGHT section is not acceptable.
- If a genuinely tempting wrong answer, shortcut, or misconception exists
  for this specific example, name it in THOUGHT and explain concretely why
  it's wrong. Do NOT invent or force a wrong alternative when none is
  natural — a fabricated "tempting mistake" for a trivial example is worse
  than not including one. Only include it when it's real.
- THOUGHT steps must be genuine reasoning toward the answer, not a
  restatement of the instruction.
- If the instruction demands an exact-format or no-extra-output final
  answer, THOUGHT still applies, but ANSWER must obey that constraint
  precisely — reasoning goes in THOUGHT, never bleeds into ANSWER.
- Return ONLY JSON.

<JSON_START>

{{
    "difficulty": {{
        "context":"...",
        "instruction":"...",
        "answer":"THOUGHT:\\n1. ...\\n2. ...\\nANSWER: ..."
    }},

    "context": {{
        "context":"...",
        "instruction":"...",
        "answer":"THOUGHT:\\n1. ...\\n2. ...\\nANSWER: ..."
    }},

    "constraint": {{
        "context":"...",
        "instruction":"...",
        "answer":"THOUGHT:\\n1. ...\\n2. ...\\nANSWER: ..."
    }},

    "perspective": {{
        "context":"...",
        "instruction":"...",
        "answer":"THOUGHT:\\n1. ...\\n2. ...\\nANSWER: ..."
    }}
}}

<JSON_END>
"""


# ============================================================
# Metadata Builder
# ============================================================

def build_metadata(seed, variation_name):
    """
    Shared metadata for every generated example.
    Matches the locked dataset schema: id, category, source,
    difficulty, parent_id, variation_role.
    """

    return {
        "id": f'{seed["id"]}_gen_{variation_name}',
        "category": seed["category"],
        "source": "generated",
        "difficulty": next_difficulty(seed["difficulty"]),
        "parent_id": seed["id"],
        "variation_role": variation_name,
    }


# ============================================================
# Duplicate-safe Prompt Selection
# ============================================================

def choose_unique_prompt(candidate):
    """
    Reject duplicates before writing.
    """

    if not validate_prompt(candidate):
        return None

    if is_duplicate(candidate):
        return None

    return candidate


# ============================================================
# Strict-output Enforcement
#
# Some instructions demand an exact, unpolluted final output (e.g.
# "output only the word DONE", "return only valid JSON, no explanation").
# The teacher still reasons via THOUGHT/ANSWER during generation — that
# scaffolding helps it land on a correct answer — but storing the THOUGHT
# prefix as the trained `response` for these instructions would teach the
# student model to violate the very constraint the example is meant to
# test. This is decided deterministically in code from the prompt text,
# not left to the teacher's judgment at inference time, so it's a fixed,
# testable rule rather than inconsistent per-generation behavior.
# ============================================================

STRICT_OUTPUT_PATTERNS = [
    r"output only",
    r"respond only",
    r"respond with only",
    r"return only",
    r"reply with only",
    r"do not include (any )?(other|extra|additional) (text|output|formatting)",
    r"no (other|extra|additional) (text|output|formatting|explanation)",
    r"no explanation",
    r"nothing else",
    r"exactly the word",
    r"single word (answer|response|output)",
    r"do not include formatting",
]

_STRICT_OUTPUT_RE = re.compile(
    "|".join(STRICT_OUTPUT_PATTERNS),
    flags=re.IGNORECASE,
)


def is_strict_output_instruction(prompt: str) -> bool:
    """
    Deterministic, testable check — not a model judgment call.
    """

    if not prompt:
        return False

    return bool(_STRICT_OUTPUT_RE.search(prompt))


def extract_final_answer(response: str) -> str:
    """
    Strips everything up to and including the last 'ANSWER:' marker,
    returning just the final answer content. If no marker is found,
    returns the response unchanged (fail-safe: never silently blank
    out a response we can't confidently parse).
    """

    if not response:
        return response

    marker_pos = response.rfind("ANSWER:")

    if marker_pos == -1:
        logger.warning(
            "extract_final_answer: no 'ANSWER:' marker found, "
            "returning response unchanged."
        )
        return response

    return response[marker_pos + len("ANSWER:"):].strip()


# ============================================================
# Output Record Builders
# ============================================================

def build_prompt_record(seed, variation_name, prompt, answer):
    meta = build_metadata(seed, variation_name)

    if isinstance(answer, (dict, list)):
        answer = json.dumps(answer, ensure_ascii=False)
    elif not isinstance(answer, str):
        answer = str(answer)

    if is_strict_output_instruction(prompt):
        answer = extract_final_answer(answer)

    return {
        **meta,
        "prompt": prompt,
        "response": answer,
    }


def build_full_record(seed, variation_name, example):
    meta = build_metadata(seed, variation_name)

    answer = example["answer"]
    instruction = example["instruction"]

    if isinstance(answer, (dict, list)):
        answer = json.dumps(answer, ensure_ascii=False)
    elif not isinstance(answer, str):
        answer = str(answer)

    if is_strict_output_instruction(instruction):
        answer = extract_final_answer(answer)

    return {
        **meta,
        "context": example["context"],
        "prompt": instruction,
        "response": answer,
    }


# ============================================================
# Simple Progress Logger
# ============================================================

def log_generation(seed_id, variation):
    logger.info(
        f"[{seed_id}] Generated {variation}"
    )

# ============================================================
# Batch Answer Prompt
# ============================================================

def build_batch_answer_prompt(prompts):
    """
    Generate answers for all prompt variations in a single inference.
    """

    formatted = []

    for i, p in enumerate(prompts, start=1):

        formatted.append(
            f"""
Prompt {i}

{p}
"""
        )

    joined = "\n".join(formatted)

    return f"""
You are an expert AI assistant training a smaller model to reason like you.

Solve every instruction below completely. Do NOT just give a final answer —
show your reasoning first, then the answer, using EXACTLY this structure
inside each string value:

THOUGHT:
1. <first reasoning step>
2. <continue reasoning toward the correct answer>
(as many steps as genuinely needed — usually 2-5)
ANSWER: <the final, clean answer — nothing after this line>

CRITICAL RULES

- Every one of the four values MUST contain both a THOUGHT section and an
  ANSWER section in that exact order. A bare answer with no THOUGHT section
  is not acceptable, even for simple-looking tasks.
- If a genuinely tempting wrong answer, shortcut, or misconception exists
  for this specific task, name it in THOUGHT and explain concretely why
  it's wrong. Do NOT invent or force a wrong alternative when none is
  natural — a fabricated "tempting mistake" for a trivial task is worse
  than not including one. Only include it when it's real.
- THOUGHT steps must be genuine reasoning toward the answer — not a
  restatement of the question, not a single vague step like "figure it out".
- ANSWER must be the clean final output only — no hedging, no "I think",
  no repeating the reasoning.
- If the instruction itself demands an exact-format or no-extra-output final
  answer (e.g. "output only the word X"), the THOUGHT section still applies,
  but ANSWER must obey that exact-output constraint precisely — reasoning
  goes in THOUGHT, never bleeds into ANSWER.
- If a value contains code, embed it inside THOUGHT and/or ANSWER as a plain
  string with \\n for newlines.

Return your answer using EXACTLY this JSON shape, with EXACTLY these
four keys — do not rename, nest, wrap, or add any other keys such as
"result" or "answer".

Example of the correct shape for one value (write NEW content for the
actual task, this is only showing the structure — note step 2 uses a real
tempting shortcut for this specific problem, not a fabricated one):

<JSON_START>

{{
    "1":"THOUGHT:\\n1. The train covers 240 km in 3 hours, so its initial speed is 240/3 = 80 km/h.\\n2. A tempting shortcut is to average the two speeds (80 and 100) and multiply by total distance, but that's wrong because the time spent at each speed differs, not the distance — averaging speeds directly ignores that.\\n3. Instead, compute time for each leg separately: remaining 180 km at 100 km/h takes 180/100 = 1.8 hours.\\n4. Total time is 3 + 1.8 = 4.8 hours; total distance is 240 + 180 = 420 km.\\n5. Average speed is total distance over total time: 420 / 4.8 = 87.5 km/h, not the naive average of 90.\\nANSWER: Total travel time is 4.8 hours; average speed is 87.5 km/h.",
    "2":"...",
    "3":"...",
    "4":"..."
}}

<JSON_END>

Instructions

{joined}
"""


# ============================================================
# Prompt-only Generator
# ============================================================

def generate_prompt_only_batch(
    model,
    tokenizer,
    seed,
):
    """
    Pipeline

    Seed
        ↓
    Generate 4 prompts
        ↓
    Generate 4 answers together
        ↓
    Return records
    """

    variation_json = chat_json(
        model,
        tokenizer,
        build_prompt_only_prompt(seed)
    )

    prompt_map = {}

    for variation_name in [
        "difficulty",
        "context",
        "constraint",
        "perspective",
    ]:

        if variation_name not in variation_json:
            logger.warning(
                f"[{seed['id']}] Missing key '{variation_name}' in "
                f"variation JSON (got keys: {list(variation_json.keys())})."
            )
            continue

        raw_prompt = variation_json[variation_name]

        if not isinstance(raw_prompt, str) or not raw_prompt.strip():
            # Model returned a nested dict, null, or empty string
            # for this variation instead of plain text — not usable
            # as a prompt, so skip it rather than crash the whole seed.
            logger.warning(
                f"[{seed['id']}] '{variation_name}' was {type(raw_prompt).__name__}, "
                f"not usable text — skipped."
            )
            continue

        prompt = raw_prompt.strip()

        raw_len_chars = len(prompt)
        raw_len_words = len(prompt.split())

        prompt = choose_unique_prompt(prompt)

        if prompt is None:
            logger.warning(
                f"[{seed['id']}] '{variation_name}' rejected by "
                f"choose_unique_prompt (len={raw_len_chars} chars, "
                f"{raw_len_words} words — either too short or a duplicate)."
            )
            continue

        prompt_map[variation_name] = prompt

    if not prompt_map:
        return []

    answers = chat_json(
        model,
        tokenizer,
        build_batch_answer_prompt(
            list(prompt_map.values())
        )
    )

    records = []

    prompt_names = list(prompt_map.keys())

    for idx, variation_name in enumerate(prompt_names):

        raw_answer = answers.get(str(idx + 1)) if isinstance(answers, dict) else None

        if raw_answer is None:
            # For training data, silently mislabeling an answer to the
            # wrong prompt (via a positional guess) is worse than dropping
            # the example outright. If the model didn't return the exact
            # key we asked for, skip this variation visibly rather than
            # guess which of the other answers might belong to it.
            logger.warning(
                f"[{seed['id']}] Missing exact key '{idx + 1}' in answers "
                f"(got keys: {list(answers.keys()) if isinstance(answers, dict) else type(answers)}); "
                f"skipping '{variation_name}' rather than guessing positionally."
            )
            continue

        if isinstance(raw_answer, (dict, list)):
            answer = json.dumps(raw_answer, ensure_ascii=False)
        elif not isinstance(raw_answer, str):
            answer = str(raw_answer)
        else:
            answer = raw_answer.strip()

        if not prompt_map[variation_name] or not answer:
            continue

        record = build_prompt_record(
            seed,
            variation_name,
            prompt_map[variation_name],
            answer,
        )

        records.append(record)

        log_generation(
            seed["id"],
            variation_name
        )

    return records


# ============================================================
# Full-example Generator
# ============================================================

def generate_full_example_batch(
    model,
    tokenizer,
    seed,
):
    """
    Teacher generates

    context
    instruction
    answer

    in one inference.
    """

    examples = chat_json(
        model,
        tokenizer,
        build_full_example_prompt(seed)
    )

    records = []

    for variation_name in [
        "difficulty",
        "context",
        "constraint",
        "perspective",
    ]:

        if variation_name not in examples:
            continue

        example = examples[variation_name]

        if not validate_entry(example):
            continue

        if is_duplicate(
            example["instruction"]
        ):
            continue

        record = build_full_record(
            seed,
            variation_name,
            example,
        )

        records.append(record)

        log_generation(
            seed["id"],
            variation_name,
        )

    return records

from tqdm import tqdm


# ============================================================
# Resume Support
# ============================================================

def load_completed_seed_ids(output_path: Path):
    """
    Resume generation without regenerating completed seeds.
    """

    completed = set()

    if not output_path.exists():
        return completed

    with open(output_path, "r", encoding="utf-8") as f:

        for line in f:

            try:
                obj = json.loads(line)

                completed.add(obj["parent_id"])

            except Exception:
                continue

    logger.info(
        f"Found {len(completed)} completed seeds."
    )

    return completed


# ============================================================
# Category Dispatcher
# ============================================================

def process_seed(
    model,
    tokenizer,
    seed,
):
    """
    Automatically select the correct generation pipeline.
    """

    category = seed["category"]

    if category in PROMPT_ONLY_CATEGORIES:

        return generate_prompt_only_batch(
            model,
            tokenizer,
            seed,
        )

    elif category in FULL_EXAMPLE_CATEGORIES:

        return generate_full_example_batch(
            model,
            tokenizer,
            seed,
        )

    else:

        logger.warning(
            f"Unknown category: {category}"
        )

        return []


# ============================================================
# Save Records
# ============================================================

def save_records(
    output_path: Path,
    records,
):

    if not records:
        return

    for record in records:

        write_jsonl(
            output_path,
            record,
        )


# ============================================================
# Process One Category
# ============================================================

def process_category(
    model,
    tokenizer,
    seeds,
    output_path: Path,
):

    completed = load_completed_seed_ids(
        output_path
    )

    generated = 0

    skipped = 0

    failed = 0

    iterator = tqdm(
        seeds,
        desc="Generating"
    )

    for seed in iterator:

        if seed["id"] in completed:

            skipped += 1
            continue

        try:

            records = process_seed(
                model,
                tokenizer,
                seed,
            )

            save_records(
                output_path,
                records,
            )

            generated += len(records)

            iterator.set_postfix(
                generated=generated,
                skipped=skipped,
            )

        except Exception as e:

            logger.exception(e)

            failed += 1

    logger.info(
        "=" * 60
    )

    logger.info(
        f"Finished generation."
    )

    logger.info(
        f"Generated : {generated}"
    )

    logger.info(
        f"Skipped  : {skipped}"
    )

    logger.info(
        f"Failed   : {failed}"
    )

    logger.info(
        "=" * 60
    )


# ============================================================
# Public API
# ============================================================

def run_generation(
    model,
    tokenizer,
    seeds,
    output_path,
):
    """
    Main entrypoint.

    This is the only function that
    generate_dataset.py needs.
    """

    process_category(
        model,
        tokenizer,
        seeds,
        Path(output_path),
    )