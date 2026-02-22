import json
import random

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

random.seed(42)

with open("data/training/synthetic_distillation.jsonl", "r") as f:
    entries = [json.loads(line) for line in f if line.strip()]

# Shuffle and split 80/20 train/valid
random.shuffle(entries)
split = int(len(entries) * 0.8)
train_entries = entries[:split]
valid_entries = entries[split:]

def to_chat_format(entry):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": entry["instruction"]},
            {"role": "assistant", "content": entry["response"]}
        ]
    }

with open("data/training/train.jsonl", "w") as f:
    for entry in train_entries:
        f.write(json.dumps(to_chat_format(entry)) + "\n")

with open("data/training/valid.jsonl", "w") as f:
    for entry in valid_entries:
        f.write(json.dumps(to_chat_format(entry)) + "\n")

print(f"Train: {len(train_entries)} entries -> data/training/train.jsonl")
print(f"Valid: {len(valid_entries)} entries -> data/training/valid.jsonl")