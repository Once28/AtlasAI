"""
eval.py — lightweight local eval for the family travel agent.

Loops through a handful of test prompts, calls Ollama directly (bypassing
Streamlit), and reports latency + tokens/sec per call. Also runs a cheap
keyword check confirming responses lean toward the high-priority unvisited
regions and don't treat language/translation as a blocking factor.

This is a sanity check, not a rigorous eval harness — good enough to catch
a regression after a prompt or model change before you fire up the UI.

Run: python eval.py
"""

import json
import time

from prompts import build_qwen_system_prompt
from router import call_ollama, MODEL_QWEN

DB_PATH = "family_db.json"

TEST_PROMPTS = [
    "Suggest a 10-day trip for the family this fall, budget around $3000.",
    "We haven't done Western Europe yet — what would you recommend there?",
    "Is Singapore worth going back to, or should we look somewhere new?",
    "The grandparents don't speak English — will that be a problem in the Baltics?",
    "What's a good option under $4500 that isn't another cruise?",
]

# Words that should show up reasonably often if the model is honoring the
# high-priority/unvisited guidance from family_db.json.
HIGH_PRIORITY_KEYWORDS = ["western europe", "western mediterranean", "baltic"]
# Phrases that would indicate the model is treating language as a hard
# blocker, which item 2 of the prompt update explicitly says it should not.
LANGUAGE_BLOCKING_PHRASES = [
    "not recommended due to language",
    "because they don't speak english",
    "language barrier makes this destination unsuitable",
]


def load_db() -> dict:
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_eval():
    db = load_db()
    system_prompt = build_qwen_system_prompt(db)

    results = []
    for prompt in TEST_PROMPTS:
        messages = [{"role": "user", "content": prompt}]

        start = time.perf_counter()
        response = call_ollama(MODEL_QWEN, system_prompt, messages)
        elapsed_s = time.perf_counter() - start

        content = response.get("message", {}).get("content", "")
        eval_count = response.get("eval_count")  # tokens generated, if Ollama reports it
        eval_duration_ns = response.get("eval_duration")  # generation time in ns
        tokens_per_sec = (
            eval_count / (eval_duration_ns / 1e9)
            if eval_count and eval_duration_ns
            else None
        )

        lower_content = content.lower()
        mentions_high_priority = any(kw in lower_content for kw in HIGH_PRIORITY_KEYWORDS)
        blocks_on_language = any(p in lower_content for p in LANGUAGE_BLOCKING_PHRASES)

        results.append({
            "prompt": prompt,
            "latency_s": round(elapsed_s, 2),
            "tokens_per_sec": round(tokens_per_sec, 1) if tokens_per_sec else None,
            "mentions_high_priority_region": mentions_high_priority,
            "blocks_on_language": blocks_on_language,
            "response_preview": content[:200],
        })

    return results


def print_report(results: list[dict]):
    print(f"\n{'=' * 60}\nFamily Travel Agent — eval report ({MODEL_QWEN})\n{'=' * 60}")
    for r in results:
        print(f"\nPrompt: {r['prompt']}")
        print(f"  Latency: {r['latency_s']}s   Tokens/sec: {r['tokens_per_sec']}")
        print(f"  Mentions high-priority region: {r['mentions_high_priority_region']}")
        print(f"  Blocks on language (should be False): {r['blocks_on_language']}")
        print(f"  Preview: {r['response_preview']}...")

    avg_latency = sum(r["latency_s"] for r in results) / len(results)
    flagged = [r for r in results if r["blocks_on_language"]]
    print(f"\n{'-' * 60}")
    print(f"Average latency: {avg_latency:.2f}s")
    print(f"Responses treating language as blocking (should be 0): {len(flagged)}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    print_report(run_eval())
