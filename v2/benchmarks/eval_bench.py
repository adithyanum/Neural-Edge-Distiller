#!/usr/bin/env python3
"""
Neural Edge Distiller V2 - eval + concurrency benchmark against the live vLLM endpoint.

Part 1 (eval): the same hand-written prompts go to the base model and to
distiller-v2, both at temperature 0. Each output is scored on:
  - structure_ok : CoT categories need exactly one THOUGHT: then exactly one ANSWER:
                   instruction_following needs NO THOUGHT: block (bare output)
  - clean_stop   : finish_reason == "stop" (not "length")
  - compliant    : structure_ok AND clean_stop
  - correct      : rough regex check on the final answer, math_reasoning only

Part 2 (bench): fires requests at distiller-v2 at several concurrency levels and
records latency p50/p95, total tokens/sec and per-stream tokens/sec.

Usage:
  python eval_bench.py --url https://<tunnel>.trycloudflare.com
  python eval_bench.py --url ... --skip-bench --per-category 1     # quick smoke test
"""

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import requests

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_MODEL = "distiller-v2"

# Hand-written prompts. "expect" is an optional regex for a rough correctness check.
PROMPTS = {
    "math_reasoning": [
        {"prompt": "A shop sells pens at 12 rupees each. Riya buys 7 pens and pays with a 100 rupee note. How much change does she get?", "expect": r"\b16\b"},
        {"prompt": "A tank fills at 15 litres per minute and drains at 6 litres per minute. Starting empty, how many minutes until it holds 180 litres?", "expect": r"\b20\b"},
        {"prompt": "What is '15%' of 240, plus '30%' of 80?", "expect": r"\b60\b"},
        {"prompt": "Two cyclists start 130 km apart and ride toward each other at 12 km/h and 14 km/h. After how many minutes do they meet?", "expect": r"\b300\b|\b5\s*(hours|hrs|h)\b"},
        {"prompt": "The sum of three consecutive integers is 96. What are they?", "expect": r"\b31\b"},
    ],
    "logical_reasoning": [
        {"prompt": "All bloops are razzies. All razzies are lazzies. Is every bloop a lazzie?"},
        {"prompt": "If it rains, the match is cancelled. The match was not cancelled. What can we conclude about the rain?"},
        {"prompt": "Ana is taller than Ben. Ben is taller than Chitra. Dev is shorter than Chitra. Who is the tallest and who is the shortest?"},
        {"prompt": "Three boxes are labelled Apples, Oranges and Mixed, and every label is wrong. You pick one fruit from the box labelled Mixed and it is an apple. What does that box contain?"},
        {"prompt": "Every student either passed or failed, and nobody who passed got a prize. Meera got a prize. Did Meera pass?"},
    ],
    "code_debugging": [
        {"prompt": "This Python function should return the largest number in a list, but it returns 0 for [-5, -2, -9]. Why, and how do you fix it?\n\ndef largest(nums):\n    best = 0\n    for n in nums:\n        if n > best:\n            best = n\n    return best"},
        {"prompt": "This loop is supposed to print every item but it skips the first one. Fix it.\n\nfor i in range(1, len(items)):\n    print(items[i])"},
        {"prompt": "Calling this function twice shares items between calls. Why?\n\ndef add_item(item, bucket=[]):\n    bucket.append(item)\n    return bucket"},
        {"prompt": "This code raises IndexError. Explain why and fix it.\n\nfor i in range(len(a) + 1):\n    print(a[i])"},
        {"prompt": "Python says SyntaxError on this line. What is wrong?\n\nif x = 5:\n    print('five')"},
    ],
    "classification": [
        {"prompt": "Classify the sentiment as positive, negative or neutral: 'I waited 40 minutes and the waiter was rude.'"},
        {"prompt": "Is this email spam or not spam? 'Congratulations! You won a free cruise. Click here to claim now.'"},
        {"prompt": "Which category fits best: sports, politics, technology or cooking? 'The new chip doubles battery life on laptops.'"},
        {"prompt": "Is this review positive or negative? 'Best purchase I have made all year, works perfectly.'"},
        {"prompt": "Classify the intent as booking, cancellation, rescheduling or complaint: 'Can you move my appointment to Thursday?'"},
    ],
    "planning": [
        {"prompt": "Plan a 3-day study schedule to prepare for a data structures exam covering 10 topics."},
        {"prompt": "Make a step-by-step plan to move a small website from shared hosting to a cloud server with minimal downtime."},
        {"prompt": "Give me a plan to learn basic SQL in two weeks, one hour a day."},
        {"prompt": "Plan how to run the registration desk for a college tech fest with 300 attendees."},
        {"prompt": "Plan a weekend trip for two people on a budget of 8000 rupees."},
    ],
    "instruction_following": [
        {"prompt": "Say hello in exactly one sentence."},
        {"prompt": "List three fruits, comma separated, nothing else."},
        {"prompt": "Reply with only the number 42."},
        {"prompt": "Write the word 'ready' in capital letters and nothing else."},
        {"prompt": "Translate 'good morning' to Spanish. Output only the translation."},
    ],
}


# ------------------------------------------------------------------ helpers

def chat(url, model, prompt, max_tokens, timeout=95, retries=3):
    """One chat completion. Never raises; errors come back in the dict.

    Retries on connection-level failures (tunnel drops, SSL EOF, resets) with
    a short backoff, since those are transport flakiness, not model failures.
    Does NOT retry on a successful-but-non-JSON response (e.g. a Cloudflare
    100s timeout page) or an HTTP error with a JSON body - those are real
    and get reported as-is.
    """
    t0 = time.perf_counter()
    out = {"text": "", "finish_reason": None, "completion_tokens": 0, "error": None}
    last_exc = None
    for attempt in range(retries):
        try:
            r = requests.post(
                f"{url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0,
                },
                timeout=timeout,
            )
            ctype = r.headers.get("content-type", "")
            if "json" not in ctype:  # cloudflare timeout pages come back as HTML
                out["error"] = f"non-json response: HTTP {r.status_code} ({ctype})"
            else:
                d = r.json()
                if "choices" not in d:
                    out["error"] = f"HTTP {r.status_code}: {str(d)[:200]}"
                else:
                    ch = d["choices"][0]
                    out["text"] = ch["message"]["content"] or ""
                    out["finish_reason"] = ch.get("finish_reason")
                    out["completion_tokens"] = d.get("usage", {}).get("completion_tokens", 0)
            last_exc = None
            break
        except (requests.exceptions.ConnectionError, requests.exceptions.SSLError, requests.exceptions.Timeout) as e:
            last_exc = e
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
            continue
        except Exception as e:
            out["error"] = f"{type(e).__name__}: {e}"
            break
    if last_exc is not None:
        out["error"] = f"{type(last_exc).__name__} after {retries} attempts: {last_exc}"
    out["latency"] = time.perf_counter() - t0
    return out


def pctl(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def score(cat, text, finish, expect):
    n_t = text.count("THOUGHT:")
    n_a = text.count("ANSWER:")
    s = {"n_thought": n_t, "n_answer": n_a, "clean_stop": finish == "stop"}
    if cat == "instruction_following":
        s["structure_ok"] = n_t == 0
    else:
        s["structure_ok"] = n_t == 1 and n_a == 1 and text.index("THOUGHT:") < text.index("ANSWER:")
    s["compliant"] = s["structure_ok"] and s["clean_stop"]
    if expect:
        seg = text.split("ANSWER:")[-1] if "ANSWER:" in text else text[-200:]
        s["correct"] = bool(re.search(expect, seg))
    return s


# ------------------------------------------------------------------ eval

def run_eval(url, per_category, max_tokens, workers):
    jobs = []
    for model in (BASE_MODEL, ADAPTER_MODEL):
        for cat, items in PROMPTS.items():
            for i, item in enumerate(items[:per_category]):
                jobs.append((model, cat, i, item["prompt"], item.get("expect")))

    def run(job):
        model, cat, i, prompt, expect = job
        res = chat(url, model, prompt, max_tokens)
        rec = {"model": model, "category": cat, "idx": i, "prompt": prompt, **res}
        if res["error"]:
            rec["scores"] = {"structure_ok": False, "clean_stop": False, "compliant": False}
        else:
            rec["scores"] = score(cat, res["text"], res["finish_reason"], expect)
        return rec

    print(f"Running eval: {len(jobs)} requests, {workers} at a time...")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        records = list(ex.map(run, jobs))

    summary = {}
    for model in (BASE_MODEL, ADAPTER_MODEL):
        summary[model] = {}
        groups = {cat: [r for r in records if r["model"] == model and r["category"] == cat] for cat in PROMPTS}
        groups["ALL"] = [r for r in records if r["model"] == model]
        for cat, rs in groups.items():
            if not rs:
                continue
            n = len(rs)
            with_expect = [r for r in rs if "correct" in r["scores"]]
            summary[model][cat] = {
                "n": n,
                "structure_ok": sum(r["scores"]["structure_ok"] for r in rs) / n,
                "clean_stop": sum(r["scores"]["clean_stop"] for r in rs) / n,
                "compliant": sum(r["scores"]["compliant"] for r in rs) / n,
                "correct": (sum(r["scores"]["correct"] for r in with_expect) / len(with_expect)) if with_expect else None,
                "errors": sum(1 for r in rs if r["error"]),
            }
    return records, summary


def print_eval(summary):
    for model, cats in summary.items():
        print(f"\n== {model} ==")
        print(f"{'category':24}{'n':>4}{'structure':>11}{'clean_stop':>12}{'compliant':>11}{'correct':>9}{'errors':>8}")
        for cat, s in cats.items():
            corr = "-" if s["correct"] is None else f"{s['correct']:.0%}"
            print(f"{cat:24}{s['n']:>4}{s['structure_ok']:>11.0%}{s['clean_stop']:>12.0%}{s['compliant']:>11.0%}{corr:>9}{s['errors']:>8}")


# ------------------------------------------------------------------ bench

def run_bench(url, model, levels, n_requests, max_tokens):
    pool = [item["prompt"] for items in PROMPTS.values() for item in items]
    rows = []
    for level in levels:
        jobs = [pool[i % len(pool)] for i in range(n_requests)]
        print(f"Bench: concurrency {level}, {n_requests} requests...")
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=level) as ex:
            res = list(ex.map(lambda p: chat(url, model, p, max_tokens), jobs))
        wall = time.perf_counter() - t0
        ok = [r for r in res if not r["error"]]
        lat = [r["latency"] for r in ok]
        toks = sum(r["completion_tokens"] for r in ok)
        per_stream = [r["completion_tokens"] / r["latency"] for r in ok if r["latency"] > 0]
        rows.append({
            "concurrency": level,
            "requests": n_requests,
            "errors": len(res) - len(ok),
            "wall_s": round(wall, 2),
            "req_per_s": round(len(ok) / wall, 3),
            "total_tok_per_s": round(toks / wall, 1),
            "per_stream_tok_per_s": round(sum(per_stream) / len(per_stream), 1) if per_stream else None,
            "lat_p50_s": round(pctl(lat, 0.5), 2) if lat else None,
            "lat_p95_s": round(pctl(lat, 0.95), 2) if lat else None,
            "avg_completion_tokens": round(toks / len(ok), 1) if ok else None,
        })
    return rows


def print_bench(rows):
    print(f"\n{'conc':>5}{'ok/req':>9}{'wall_s':>9}{'tok/s':>9}{'per-stream':>12}{'p50_s':>8}{'p95_s':>8}{'errors':>8}")
    for r in rows:
        ok = r["requests"] - r["errors"]
        print(f"{r['concurrency']:>5}{f'{ok}/{r['requests']}':>9}{r['wall_s']:>9}{r['total_tok_per_s']:>9}"
              f"{str(r['per_stream_tok_per_s']):>12}{str(r['lat_p50_s']):>8}{str(r['lat_p95_s']):>8}{r['errors']:>8}")


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="tunnel base url, no trailing path")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--per-category", type=int, default=5, help="prompts per category (max 5)")
    ap.add_argument("--max-tokens", type=int, default=512, help="eval max_tokens")
    ap.add_argument("--eval-workers", type=int, default=4)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--requests", type=int, default=32, help="requests per concurrency level")
    ap.add_argument("--bench-max-tokens", type=int, default=200)
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-bench", action="store_true")
    args = ap.parse_args()
    url = args.url.rstrip("/")

    # sanity: both models must be registered
    try:
        models = [m["id"] for m in requests.get(f"{url}/v1/models", timeout=30).json()["data"]]
    except Exception as e:
        sys.exit(f"Could not reach {url}/v1/models: {e}")
    for m in (BASE_MODEL, ADAPTER_MODEL):
        if m not in models:
            sys.exit(f"Model {m!r} not served. Server lists: {models}")

    print("Warm-up...")
    for m in (BASE_MODEL, ADAPTER_MODEL):
        w = chat(url, m, "Say hi.", 16)
        if w["error"]:
            sys.exit(f"Warm-up failed on {m}: {w['error']}")

    result = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "url": url,
            "base_model": BASE_MODEL,
            "adapter": ADAPTER_MODEL,
            "temperature": 0,
            "eval_max_tokens": args.max_tokens,
            "bench_max_tokens": args.bench_max_tokens,
            "gpu": "Kaggle T4 (fp16), vLLM 0.29.0, single instance",
        }
    }

    if not args.skip_eval:
        records, summary = run_eval(url, args.per_category, args.max_tokens, args.eval_workers)
        print_eval(summary)
        result["eval"] = {"summary": summary, "samples": records}

    if not args.skip_bench:
        rows = run_bench(url, ADAPTER_MODEL, args.levels, args.requests, args.bench_max_tokens)
        print_bench(rows)
        result["bench"] = rows

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval_bench_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()