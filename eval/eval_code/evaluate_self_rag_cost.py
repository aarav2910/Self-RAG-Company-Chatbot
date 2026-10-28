"""Estimate chat-model cost for full Self-RAG runs using reported token usage.

Run from the repository root with:
    python eval/eval_code/evaluate_self_rag_cost.py --limit 5 --repeats 1

The estimate covers chat-model tokens reported by LangChain. It does not include
embedding API usage from the vector retriever.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from langchain_core.callbacks import UsageMetadataCallbackHandler

PROJECT_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_DIR
GOLDEN_PATH = PROJECT_DIR / "eval" / "golden" / "correctness_golden.jsonl"
SELF_RAG_PATH = PROJECT_DIR / "src" / "self-rag.py"
RESULTS_DIR = PROJECT_DIR / "eval" / "results"
RESULTS_PATH = RESULTS_DIR / "self_rag_cost_results.json"

# Defaults mirror the rates in the supplied example. Verify and adjust these
# constants against the provider's current pricing before using for budgeting.
MODEL_PRICING_PER_MILLION = {
    "gpt-4o-mini": {
        "input": 0.15,
        "cached_input": 0.075,
        "output": 0.60,
    }
}


def load_goldens(path: Path) -> list[dict]:
    """Read JSONL or a JSON array stored with a .jsonl suffix."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "goldens" in data:
        return data["goldens"]
    return [data]


def load_self_rag():
    spec = importlib.util.spec_from_file_location("self_rag_cost_eval", SELF_RAG_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Self-RAG module: {SELF_RAG_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def model_family(name: str) -> str:
    """Map dated provider model names to a configured pricing family."""
    for family in MODEL_PRICING_PER_MILLION:
        if name == family or name.startswith(family + "-"):
            return family
    raise ValueError(
        f"No price configured for model {name!r}; add its input/output rates to "
        "MODEL_PRICING_PER_MILLION before calculating cost."
    )


def read_tokens(handler: UsageMetadataCallbackHandler) -> tuple[dict, int, int, int]:
    by_model = {}
    total_input = total_output = total_cached = 0
    for model_name, usage in handler.usage_metadata.items():
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        details = usage.get("input_token_details") or {}
        cached_tokens = int(
            details.get("cache_read", 0)
            or details.get("cached_tokens", 0)
            or details.get("cache_read_tokens", 0)
            or 0
        )
        cached_tokens = min(cached_tokens, input_tokens)
        by_model[model_name] = {
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_tokens,
            "uncached_input_tokens": input_tokens - cached_tokens,
            "output_tokens": output_tokens,
        }
        total_input += input_tokens
        total_output += output_tokens
        total_cached += cached_tokens
    return by_model, total_input, total_output, total_cached


def calculate_cost(by_model: dict) -> dict:
    input_cost = cached_cost = output_cost = 0.0
    for model_name, usage in by_model.items():
        prices = MODEL_PRICING_PER_MILLION[model_family(model_name)]
        input_cost += usage["uncached_input_tokens"] * prices["input"] / 1_000_000
        cached_cost += usage["cached_input_tokens"] * prices["cached_input"] / 1_000_000
        output_cost += usage["output_tokens"] * prices["output"] / 1_000_000
    return {
        "input_usd": input_cost,
        "cached_input_usd": cached_cost,
        "output_usd": output_cost,
        "total_usd": input_cost + cached_cost + output_cost,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=5, help="Golden questions to run (default: 5).")
    parser.add_argument("--repeats", type=int, default=1, help="Runs per question (default: 1).")
    args = parser.parse_args()
    if args.limit < 1 or args.repeats < 1:
        parser.error("limit and repeats must be positive")
    if not GOLDEN_PATH.is_file():
        raise FileNotFoundError(f"Cost-eval golden dataset not found: {GOLDEN_PATH}")

    goldens = load_goldens(GOLDEN_PATH)
    samples = goldens[: args.limit]
    questions = [item.get("question") or item.get("query") for item in samples]
    if not questions or any(not question for question in questions):
        raise ValueError("Each selected golden must contain a question or query.")

    self_rag = load_self_rag()
    rows = []
    total_runs = len(questions) * args.repeats
    print(f"Measuring Self-RAG chat-model usage for {total_runs} run(s)...", flush=True)

    for question_index, question in enumerate(questions):
        for repeat in range(1, args.repeats + 1):
            usage_handler = UsageMetadataCallbackHandler()
            start = time.perf_counter()
            result = self_rag.invoke_self_rag(question, callbacks=[usage_handler])
            elapsed_ms = (time.perf_counter() - start) * 1000
            by_model, input_tokens, output_tokens, cached_tokens = read_tokens(usage_handler)
            costs = calculate_cost(by_model)
            rows.append(
                {
                    "golden_id": samples[question_index].get("id"),
                    "question": question,
                    "repeat": repeat,
                    "latency_ms": round(elapsed_ms, 3),
                    "answer_characters": len(result.get("answer", "") or ""),
                    "model_usage": by_model,
                    "input_tokens": input_tokens,
                    "cached_input_tokens": cached_tokens,
                    "output_tokens": output_tokens,
                    **costs,
                }
            )
            print(
                f"Measured {len(rows)}/{total_runs}: ${costs['total_usd']:.8f} "
                f"({input_tokens} input, {output_tokens} output tokens)",
                flush=True,
            )

    mean_cost = mean(row["total_usd"] for row in rows)
    summary = {
        "sample_count": len(rows),
        "mean_input_tokens": mean(row["input_tokens"] for row in rows),
        "mean_cached_input_tokens": mean(row["cached_input_tokens"] for row in rows),
        "mean_output_tokens": mean(row["output_tokens"] for row in rows),
        "mean_cost_usd_per_query": mean_cost,
        "min_cost_usd": min(row["total_usd"] for row in rows),
        "max_cost_usd": max(row["total_usd"] for row in rows),
        "mean_input_cost_usd": mean(row["input_usd"] + row["cached_input_usd"] for row in rows),
        "mean_output_cost_usd": mean(row["output_usd"] for row in rows),
    }
    output = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "Full Self-RAG graph via invoke_self_rag",
        "golden_dataset": str(GOLDEN_PATH.relative_to(REPO_ROOT)),
        "pricing_usd_per_million_tokens": MODEL_PRICING_PER_MILLION,
        "cost_scope": "Chat-model usage reported by LangChain callbacks; embedding API usage is excluded.",
        "sample_questions": len(questions),
        "repeats_per_question": args.repeats,
        "summary": summary,
        "runs": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\nSelf-RAG estimated chat-model cost")
    print(f"Runs: {summary['sample_count']}")
    print(f"Mean tokens/query: {summary['mean_input_tokens']:.0f} input "
          f"({summary['mean_cached_input_tokens']:.0f} cached), "
          f"{summary['mean_output_tokens']:.0f} output")
    print(f"Mean cost/query: ${mean_cost:.8f} "
          f"(min ${summary['min_cost_usd']:.8f}, max ${summary['max_cost_usd']:.8f})")
    print("Embedding API cost is excluded from this estimate.")
    print(f"Saved results to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
