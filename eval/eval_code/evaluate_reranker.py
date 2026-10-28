"""Compare the hybrid retriever's top three with its reranked top three."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_DIR
SRC_DIR = PROJECT_DIR / "src"
EVAL_CODE_DIR = Path(__file__).resolve().parent
GOLDEN_DIR = PROJECT_DIR / "eval" / "golden"
RESULTS_DIR = PROJECT_DIR / "eval" / "results"
K = 5

GOLDEN_PATH = GOLDEN_DIR / "valve_handbook_retriever_golden.jsonl"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(EVAL_CODE_DIR))
load_dotenv(REPO_ROOT / ".env")

def load_goldens(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".json":
        data = json.loads(text)
        return data if isinstance(data, list) else data["goldens"]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def relevant_pages_for(golden: dict) -> set[int]:
    """Use all labeled relevant pages, with compatibility for older goldens."""
    pages = golden.get("pdf_pages") or []
    if not pages and golden.get("pdf_page") is not None:
        pages = [golden["pdf_page"]]
    return {int(page) for page in pages}


def recall_at_k(pages: list[int | None], relevant: set[int], k: int) -> float:
    return len(set(pages[:k]) & relevant) / len(relevant) if relevant else 0.0


def reciprocal_rank_at_k(pages: list[int | None], relevant: set[int], k: int) -> float:
    for rank, page in enumerate(pages[:k], start=1):
        if page in relevant:
            return 1 / rank
    return 0.0


def precision_at_k(pages: list[int | None], relevant: set[int], k: int) -> float:
    return sum(page in relevant for page in pages[:k]) / k


class RankingMetric(BaseMetric):
    """Expose deterministic retrieval scores through DeepEval's terminal report."""

    def __init__(self, name: str, metric_function, threshold: float | None = None):
        self.metric_name = name
        self.metric_function = metric_function
        self.threshold = threshold
        self.score = None
        self.success = None
        self.reason = None
        self.error = None
        self.evaluation_cost = None
        self.async_mode = False
        self.verbose_mode = False
        self.include_reason = False

    @property
    def __name__(self):
        return self.metric_name

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        self.score = self.metric_function(test_case)
        self.success = self.score >= self.threshold if self.threshold is not None else None
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)


def page_numbers(documents) -> list[int | None]:
    return [
        int(doc.metadata["page_number"])
        if doc.metadata.get("page_number") is not None
        else None
        for doc in documents[:K]
    ]


def make_test_case(question: str, relevant_pages: set[int], pages: list[int | None]) -> LLMTestCase:
    return LLMTestCase(
        input=question,
        actual_output=f"Retrieved {len(pages)} chunks.",
        expected_output=json.dumps({"relevant_pages": sorted(relevant_pages)}),
        retrieval_context=[
            json.dumps({"page_number": page}) for page in pages
        ],
    )


def score_case(test_case: LLMTestCase, metric_function) -> float:
    expected = json.loads(test_case.expected_output or "{}")
    relevant_pages = {int(page) for page in expected.get("relevant_pages", [])}
    pages = []
    for context in test_case.retrieval_context or []:
        try:
            page = json.loads(context).get("page_number")
            pages.append(int(page) if page is not None else None)
        except (TypeError, ValueError, json.JSONDecodeError):
            pages.append(None)
    return metric_function(pages, relevant_pages, K)


def evaluate_condition(name: str, test_cases: list[LLMTestCase]) -> None:
    evaluate(
        test_cases=test_cases,
        metrics=[
            RankingMetric(
                f"Recall@{K} {name}",
                lambda case: score_case(case, recall_at_k),
                threshold=1.0,
            ),
            RankingMetric(
                f"MRR@{K} {name}",
                lambda case: score_case(case, reciprocal_rank_at_k),
            ),
            RankingMetric(
                f"Precision@{K} {name}",
                lambda case: score_case(case, precision_at_k),
            ),
        ],
        async_config=AsyncConfig(run_async=False),
        display_config=DisplayConfig(
            show_indicator=True,
            print_results=True,
            verbose_mode=False,
            inspect_after_run=False,
            truncate_passing_cases=True,
        ),
    )


def main() -> None:
    if not GOLDEN_PATH.is_file():
        raise FileNotFoundError(f"Retriever golden dataset not found: {GOLDEN_PATH}")
    goldens = load_goldens(GOLDEN_PATH)
    if not goldens:
        raise ValueError(f"Golden dataset is empty: {GOLDEN_PATH}")

    from retriever import RETRIEVAL_K, create_hybrid_retriever
    from reranker import CROSS_ENCODER_MODEL
    from sentence_transformers import CrossEncoder

    # Both conditions share the exact same hybrid candidates. The only changed
    # operation is whether the cross-encoder reorders those candidates.
    hybrid = create_hybrid_retriever(k=RETRIEVAL_K, bm25_weight=0.5, vector_weight=0.5)
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

    raw_test_cases: list[LLMTestCase] = []
    reranked_test_cases: list[LLMTestCase] = []
    per_query_results = []

    for index, golden in enumerate(goldens, start=1):
        question = golden["question"]
        # Keep the full union from BM25 and vector search. The no-reranker
        # condition uses its first K; the reranker chooses K from this same pool.
        candidates = hybrid.invoke(question)
        raw_top_k = candidates[:K]

        if candidates:
            pairs = [(question, doc.page_content) for doc in candidates]
            scores = cross_encoder.predict(pairs)
            reranked = [
                doc
                for _, doc in sorted(
                    zip(scores, candidates),
                    key=lambda item: item[0],
                    reverse=True,
                )[:K]
            ]
        else:
            reranked = []

        raw_pages = page_numbers(raw_top_k)
        reranked_pages = page_numbers(reranked)
        relevant = relevant_pages_for(golden)

        raw_scores = {
            "recall_at_3": recall_at_k(raw_pages, relevant, K),
            "mrr_at_3": reciprocal_rank_at_k(raw_pages, relevant, K),
            "precision_at_3": precision_at_k(raw_pages, relevant, K),
        }
        reranked_scores = {
            "recall_at_3": recall_at_k(reranked_pages, relevant, K),
            "mrr_at_3": reciprocal_rank_at_k(reranked_pages, relevant, K),
            "precision_at_3": precision_at_k(reranked_pages, relevant, K),
        }

        raw_test_cases.append(make_test_case(question, relevant, raw_pages))
        reranked_test_cases.append(make_test_case(question, relevant, reranked_pages))
        per_query_results.append(
            {
                "id": golden.get("id"),
                "question": question,
                "relevant_pages": sorted(relevant),
                "hybrid_only_pages_at_3": raw_pages,
                "reranked_pages_at_3": reranked_pages,
                "hybrid_only_metrics": raw_scores,
                "reranked_metrics": reranked_scores,
            }
        )
        print(f"Compared {index}/{len(goldens)} questions", flush=True)

    print("\nDeepEval: hybrid retriever without reranker")
    evaluate_condition("hybrid_only", raw_test_cases)
    print("\nDeepEval: same hybrid candidates with cross-encoder reranker")
    evaluate_condition("reranked", reranked_test_cases)

    metric_names = ("recall_at_3", "mrr_at_3", "precision_at_3")
    aggregate = {
        condition: {
            metric: mean(result[key][metric] for result in per_query_results)
            for metric in metric_names
        }
        for condition, key in (
            ("hybrid_only", "hybrid_only_metrics"),
            ("reranked", "reranked_metrics"),
        )
    }
    output = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "golden_dataset": str(GOLDEN_PATH.relative_to(REPO_ROOT)),
        "hybrid_weights": {"bm25": 0.5, "vector": 0.5},
        "retrieval_k": RETRIEVAL_K,
        "final_k": K,
        "comparison_method": "Both conditions use the same hybrid candidates; the reranker only changes their order.",
        "metric_averaging": "macro average across questions",
        "aggregate_metrics": aggregate,
        "per_query_results": per_query_results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "reranker_comparison.json"
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\nRetriever comparison (macro averages at K=5)")
    for condition, label in (("hybrid_only", "Hybrid only"), ("reranked", "With reranker")):
        scores = aggregate[condition]
        print(
            f"{label}: Recall@3={scores['recall_at_3']:.4f} | "
            f"MRR@3={scores['mrr_at_3']:.4f} | "
            f"Precision@3={scores['precision_at_3']:.4f}"
        )
    print(f"\nSaved detailed results to: {output_path}")


if __name__ == "__main__":
    main()
