"""Evaluate Self-RAG's generator node using golden contexts only."""

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
from langchain_core.documents import Document

PROJECT_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_DIR
GOLDEN_PATH = PROJECT_DIR / "eval" / "golden" / "faithfullness_golden.jsonl"
SELF_RAG_PATH = PROJECT_DIR / "src" / "self-rag.py"
RESULTS_DIR = PROJECT_DIR / "eval" / "results"
RESULTS_PATH = RESULTS_DIR / "generator_metrics.json"
JUDGE_MODEL = "gpt-4o-mini"
THRESHOLD = 0.7

load_dotenv(REPO_ROOT / ".env")


def load_self_rag():
    """Import the graph module despite its hyphenated filename."""
    spec = importlib.util.spec_from_file_location("self_rag_eval_module", SELF_RAG_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Self-RAG module: {SELF_RAG_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    if not GOLDEN_PATH.is_file():
        raise FileNotFoundError(f"Faithfulness golden dataset not found: {GOLDEN_PATH}")

    # Despite its .jsonl suffix, this dataset is stored as a JSON array.
    goldens = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    self_rag = load_self_rag()

    test_cases = []
    case_details = []
    for golden in goldens:
        context = golden["ideal_context"]
        context_docs = [Document(page_content=chunk) for chunk in context]

        # Call the same node used by the Self-RAG graph; retrieval is skipped.
        result = self_rag.generate_from_context(
            {"question": golden["query"], "relevant_docs": context_docs}
        )
        test_cases.append(
            LLMTestCase(
                input=golden["query"],
                actual_output=result["answer"],
                retrieval_context=context,
            )
        )
        case_details.append(
            {
                "id": golden.get("id"),
                "query": golden["query"],
                "actual_output": result["answer"],
            }
        )
        print(f"Generated {len(test_cases)}/{len(goldens)} answers", flush=True)

    metrics = [
        FaithfulnessMetric(
            threshold=THRESHOLD,
            model=JUDGE_MODEL,
            include_reason=True,
        ),
        AnswerRelevancyMetric(
            threshold=THRESHOLD,
            model=JUDGE_MODEL,
            include_reason=True,
        ),
    ]

    # Run sequentially to avoid the async timeout seen with parallel judging.
    evaluation = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(run_async=False),
    )

    scores_by_metric = {metric.__name__: [] for metric in metrics}
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
                scores_by_metric.setdefault(metric_data.name, []).append(
                    float(metric_data.score)
                )

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
        "generator_node": "Self-RAG generate_from_context",
        "evaluation_mode": "golden ideal_context supplied directly; retrieval skipped",
        "judge_model": JUDGE_MODEL,
        "threshold": THRESHOLD,
        "query_count": len(case_details),
        "aggregate_metrics": aggregate,
        "per_query_results": case_details,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved evaluation results to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
