"""Evaluate Self-RAG answers for correctness, completeness, and style."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case import LLMTestCase, SingleTurnParams
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_DIR
GOLDEN_PATH = PROJECT_DIR / "eval" / "golden" / "correctness_golden.jsonl"
SELF_RAG_PATH = PROJECT_DIR / "src" / "self-rag.py"
RESULTS_DIR = PROJECT_DIR / "eval" / "results"
RESULTS_PATH = RESULTS_DIR / "self_rag_quality_results.json"
JUDGE_MODEL = "gpt-4o-mini"
THRESHOLD = 0.7

load_dotenv(REPO_ROOT / ".env")


def load_goldens(path: Path) -> list[dict]:
    """Read JSONL or a JSON array stored with a .jsonl extension."""
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
    """Load the graph from its hyphenated module filename."""
    spec = importlib.util.spec_from_file_location("self_rag_quality_eval", SELF_RAG_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Self-RAG module: {SELF_RAG_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_metrics() -> list[GEval]:
    correctness = GEval(
        name="Correctness",
        evaluation_steps=[
            "Compare factual claims in the answer with the expected answer and the question.",
            "Penalize claims that contradict the expected answer or are factually unsupported.",
            "Judge factual accuracy only; do not penalize a shorter answer for omitting details.",
            "Do not penalize additional information when it is correct and relevant.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="All stated claims are correct; brevity is acceptable."),
            Rubric(score_range=(5, 8), expected_outcome="Mostly correct, with a minor factual error."),
            Rubric(score_range=(0, 4), expected_outcome="Contains a clear factual error or contradiction."),
        ],
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
    )
    completeness = GEval(
        name="Completeness",
        evaluation_steps=[
            "Identify the key facts requested by the question and included in the expected answer.",
            "Score how much of that requested information the actual answer covers.",
            "Penalize omitted or only partly addressed key facts.",
            "Judge coverage only, not factual accuracy or writing style.",
            "Do not penalize correct, relevant details beyond the expected answer.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="Covers essentially all key requested facts."),
            Rubric(score_range=(5, 8), expected_outcome="Covers the main point but misses one or more useful details."),
            Rubric(score_range=(0, 4), expected_outcome="Misses most of the requested information."),
        ],
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
    )
    style = GEval(
        name="Style",
        evaluation_steps=[
            "Judge clarity, tone, and readability only; do not score factual correctness or completeness.",
            "Reward concise, direct, professional, plain-language answers suitable for an employee handbook assistant.",
            "Penalize confusing phrasing, unnecessary verbosity, unexplained jargon, or irrelevant preambles.",
            "Do not require a particular answer length, bullets, or an analogy.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="Clear, direct, polished, and easy to understand."),
            Rubric(score_range=(5, 8), expected_outcome="Understandable but somewhat wordy, awkward, or overly formal."),
            Rubric(score_range=(0, 4), expected_outcome="Hard to follow, unprofessional, or unnecessarily verbose."),
        ],
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
    )
    return [correctness, completeness, style]


def main() -> None:
    if not GOLDEN_PATH.is_file():
        raise FileNotFoundError(f"Quality-eval golden dataset not found: {GOLDEN_PATH}")
    goldens = load_goldens(GOLDEN_PATH)
    if not goldens:
        raise ValueError(f"Quality-eval golden dataset is empty: {GOLDEN_PATH}")

    self_rag = load_self_rag()
    test_cases: list[LLMTestCase] = []
    case_details: list[dict] = []
    for index, golden in enumerate(goldens, start=1):
        question = golden.get("question") or golden.get("query")
        expected_answer = golden.get("ideal_answer") or golden.get("reference_answer")
        if not question or not expected_answer:
            raise ValueError(f"Golden record {golden.get('id', index)} needs a question and ideal_answer.")

        result = self_rag.invoke_self_rag(question)
        answer = result.get("answer", "No answer found.")
        test_cases.append(
            LLMTestCase(
                input=question,
                actual_output=answer,
                expected_output=expected_answer,
            )
        )
        case_details.append(
            {
                "id": golden.get("id", f"case_{index}"),
                "question": question,
                "expected_answer": expected_answer,
                "actual_answer": answer,
            }
        )
        print(f"Ran Self-RAG on {index}/{len(goldens)} questions", flush=True)

    metrics = build_metrics()
    evaluation = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(run_async=False),
    )

    scores_by_metric: dict[str, list[float]] = {metric.__name__: [] for metric in metrics}
    for case, test_result in zip(case_details, evaluation.test_results):
        case["metrics"] = {}
        for metric_data in test_result.metrics_data or []:
            case["metrics"][metric_data.name] = {
                "score": metric_data.score,
                "threshold": metric_data.threshold,
                "passed": metric_data.success,
                "reason": metric_data.reason,
            }
            if metric_data.score is not None:
                scores_by_metric.setdefault(metric_data.name, []).append(float(metric_data.score))

    aggregate = {
        name: {
            "average_score": mean(scores) if scores else None,
            "threshold": THRESHOLD,
            "passed_cases": sum(
                case.get("metrics", {}).get(name, {}).get("passed") is True
                for case in case_details
            ),
            "failed_cases": sum(
                case.get("metrics", {}).get(name, {}).get("passed") is False
                for case in case_details
            ),
        }
        for name, scores in scores_by_metric.items()
    }
    output = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "golden_dataset": str(GOLDEN_PATH.relative_to(REPO_ROOT)),
        "pipeline": "Full Self-RAG graph: retrieval, generation, support check, and usefulness check",
        "judge_model": JUDGE_MODEL,
        "threshold": THRESHOLD,
        "query_count": len(case_details),
        "aggregate_metrics": aggregate,
        "per_query_results": case_details,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nSaved detailed results to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
