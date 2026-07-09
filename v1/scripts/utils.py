# scripts/utils.py
from mlx_lm.sample_utils import make_sampler

SYSTEM_PROMPT = """You are a distributed systems architect.
Respond ONLY in this exact format with no preamble, greetings, filler, or advice:

QUESTION: <restate the question exactly>
THOUGHT:
1. FAILURE MODE: What is the precise root cause or bottleneck?
2. NAIVE FIXES: What are two obvious solutions and why do they fail?
3. MECHANISM: What is the precise solution and how does it work?
4. TRADE-OFF: What does this solution cost?
ARCHITECTURE: <component1> -> <component2> -> ...
END_OF_ARCH"""

STRUCTURE_KEYS = ["FAILURE MODE", "NAIVE FIXES", "MECHANISM", "TRADE-OFF", "ARCHITECTURE", "END_OF_ARCH"]

def has_structure(r):
    return all(k in r for k in STRUCTURE_KEYS)

def format_prompt(tokenizer, question):
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True
    )

def clean_response(r):
    if "END_OF_ARCH" in r:
        return r.split("END_OF_ARCH")[0].strip() + "\nEND_OF_ARCH"
    return r.strip()

def get_sampler():
    return make_sampler(temp=0.4, top_p=0.9, min_p=0.05)