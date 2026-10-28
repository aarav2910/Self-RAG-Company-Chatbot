"""Measure cold-start and end-to-end latency for the Self-RAG graph."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from langchain_core.callbacks import BaseCallbackHandler

PROJECT_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_DIR
GOLDEN_PATH = PROJECT_DIR / "eval" / "golden" / "correctness_golden.jsonl"
SELF_RAG_PATH = PROJECT_DIR / "src" / "self-rag.py"
RESULTS_DIR = PROJECT_DIR / "eval" / "results"
RESULTS_PATH = RESULTS_DIR / "self_rag_latency_results.json"


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
    spec = importlib.util.spec_from_file_location("self_rag_latency_eval", SELF_RAG_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Self-RAG module: {SELF_RAG_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * pct / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summarize(values: list[float]) -> dict:
    if not values:
        return {}
    return {
        "sample_count": len(values),
        "mean_ms": mean(values),
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "min_ms": min(values),
        "max_ms": max(values),
    }


class FirstTokenCallback(BaseCallbackHandler):
    """Capture the first streamed LLM token produced during a pipeline run."""

    def reset(self) -> None:
        self.started_at = time.perf_counter()
        self.ttft_ms = None

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.ttft_ms is None and token:
            self.ttft_ms = (time.perf_counter() - self.started_at) * 1000


def timed_invoke(self_rag, question: str, token_callback) -> tuple[float, float | None, str]:
    token_callback.reset()
    start = time.perf_counter()
    result = self_rag.invoke_self_rag(question, callbacks=[token_callback])
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, token_callback.ttft_ms, result.get("answer", "No answer found.")


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    def format_row(row: list[str]) -> str:
        return "| " + " | ".join(value.ljust(width) for value, width in zip(row, widths)) + " |"

    print(border)
    print(format_row(headers))
    print(border)
    for row in rows:
        print(format_row(row))
    print(border)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=5, help="Number of golden questions to sample (default: 5).")
    parser.add_argument("--repeats", type=int, default=2, help="Warm runs per question (default: 2).")
    parser.add_argument("--warmup", type=int, default=1, help="Extra warm-up calls discarded before measurement (default: 1).")
    args = parser.parse_args()
    if args.limit < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error("limit and repeats must be positive; warmup cannot be negative")

    if not GOLDEN_PATH.is_file():
        raise FileNotFoundError(f"Latency-eval golden dataset not found: {GOLDEN_PATH}")
    goldens = load_goldens(GOLDEN_PATH)
    if not goldens:
        raise ValueError(f"Latency-eval golden dataset is empty: {GOLDEN_PATH}")
    samples = goldens[:args.limit]
    questions = [item.get("question") or item.get("query") for item in samples]
    if any(not question for question in questions):
        raise ValueError("Every golden latency sample must contain a question or query.")

    self_rag = load_self_rag()

    # Enable streaming for this evaluator so the callback can observe a token.
    self_rag.llm.streaming = True
    if str(PROJECT_DIR / "src") not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR / "src"))
    import generator

    generator.llm.streaming = True
    token_callback = FirstTokenCallback()

    # First real request captures process-local retriever/model initialization.
    cold_ms, cold_ttft_ms, cold_answer = timed_invoke(self_rag, questions[0], token_callback)
    print("Cold run (ms)")
    print_table(
        ["End-to-end", "First LLM token", "Answer chars"],
        [[f"{cold_ms:.1f}", f"{cold_ttft_ms:.1f}" if cold_ttft_ms is not None else "n/a", str(len(cold_answer))]],
    )

    for index in range(args.warmup):
        timed_invoke(self_rag, questions[index % len(questions)], token_callback)
    if args.warmup:
        print(f"Discarded {args.warmup} warm-up call(s).", flush=True)

    warm_runs = []
    for question_index, question in enumerate(questions):
        for repeat in range(args.repeats):
            elapsed_ms, ttft_ms, answer = timed_invoke(self_rag, question, token_callback)
            warm_runs.append(
                {
                    "golden_id": samples[question_index].get("id"),
                    "question": question,
                    "repeat": repeat + 1,
                    "latency_ms": elapsed_ms,
                    "ttft_ms": ttft_ms,
                    "answer_characters": len(answer),
                }
            )
            ttft_display = f"{ttft_ms:.1f} ms" if ttft_ms is not None else "n/a"
            print(
                f"Measured {len(warm_runs)}/{len(questions) * args.repeats}: "
                f"end-to-end {elapsed_ms:.1f} ms, first token {ttft_display}",
                flush=True,
            )

    warm_latencies = [run["latency_ms"] for run in warm_runs]
    warm_ttfts = [run["ttft_ms"] for run in warm_runs if run["ttft_ms"] is not None]
    output = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "Full Self-RAG graph via invoke_self_rag",
        "golden_dataset": str(GOLDEN_PATH.relative_to(REPO_ROOT)),
        "sample_questions": len(questions),
        "repeats_per_question": args.repeats,
        "discarded_warmups": args.warmup,
        "cold_start_ms": cold_ms,
        "cold_start_ttft_ms": cold_ttft_ms,
        "cold_start_question": questions[0],
        "warm_latency_summary": summarize(warm_latencies),
        "warm_ttft_summary": summarize(warm_ttfts),
        "warm_runs": warm_runs,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\nWarm runs (ms; first token is the first internal LLM token)")
    print_table(
        ["Case", "Run", "First token", "End-to-end", "Answer chars"],
        [
            [
                str(run["golden_id"] or "-"),
                str(run["repeat"]),
                f"{run['ttft_ms']:.1f}" if run["ttft_ms"] is not None else "n/a",
                f"{run['latency_ms']:.1f}",
                str(run["answer_characters"]),
            ]
            for run in warm_runs
        ],
    )

    latency_summary = output["warm_latency_summary"]
    ttft_summary = output["warm_ttft_summary"]

    def summary_values(summary: dict) -> list[str]:
        keys = ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "min_ms", "max_ms")
        return [f"{summary[key]:.1f}" if key in summary else "n/a" for key in keys]

    print("\nWarm-run summary (ms)")
    print_table(
        ["Metric", "Mean", "P50", "P95", "P99", "Min", "Max"],
        [
            ["First token", *summary_values(ttft_summary)],
            ["End-to-end", *summary_values(latency_summary)],
        ],
    )
    print(f"Saved detailed results to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
