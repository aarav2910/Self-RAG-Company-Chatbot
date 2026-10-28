
import json
import sys
from pathlib import Path
from statistics import mean
from datetime import datetime, timezone

from dotenv import load_dotenv

# Paths
PROJECT_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_DIR
SRC_DIR = PROJECT_DIR / "src"
GOLDEN_DIR = PROJECT_DIR / "eval" / "golden"
RESULTS_DIR = PROJECT_DIR / "eval" / "results"

CUTOFFS = [3, 4, 5]

sys.path.insert(0, str(SRC_DIR))
load_dotenv(REPO_ROOT / ".env")


# Load JSON or JSONL golden dataset
def load_goldens(path):
    text = path.read_text(encoding="utf-8").strip()

    if not text:
        return []

    if path.suffix == ".json":
        data = json.loads(text)
        return data if isinstance(data, list) else data["goldens"]

    return [json.loads(line) for line in text.splitlines() if line.strip()]


# Calculate retrieval metrics
def calculate_metrics(pages, relevant_pages, k):
    top_pages = pages[:k]

    # Recall@K
    if relevant_pages:
        recall = len(set(top_pages) & relevant_pages) / len(relevant_pages)
    else:
        recall = 0.0

    # MRR@K
    mrr = 0.0
    for rank, page in enumerate(top_pages, start=1):
        if page in relevant_pages:
            mrr = 1 / rank
            break

    # Precision@K
    precision = sum(page in relevant_pages for page in top_pages) / k

    return {
        "recall_at_k": recall,
        "mrr_at_k": mrr,
        "precision_at_k": precision
    }


def get_relevant_pages(golden):
    """Read multi-page labels, falling back to the older single-page field."""
    pages = golden.get("pdf_pages") or []
    if not pages and golden.get("pdf_page") is not None:
        pages = [golden["pdf_page"]]
    return {int(page) for page in pages}


def main():
    # Find golden dataset
    golden_path = GOLDEN_DIR / "valve_handbook_retriever_golden.jsonl"
    if not golden_path.is_file():
        raise FileNotFoundError(f"Retriever golden dataset not found: {golden_path}")

    goldens = load_goldens(golden_path)

    if not goldens:
        raise ValueError("Golden dataset is empty.")

    # Match Self-RAG's current path: hybrid retrieval without cross-encoder reranking.
    from retriever import RETRIEVAL_K, create_hybrid_retriever
    retriever = create_hybrid_retriever(k=RETRIEVAL_K)

    results = []

    for i, golden in enumerate(goldens, start=1):
        question = golden["question"]

        # Retrieve top 5 chunks
        documents = retriever.invoke(question)[:max(CUTOFFS)]

        # Preserve missing page numbers so ranks don't shift
        pages = [
            int(doc.metadata["page_number"])
            if doc.metadata.get("page_number") is not None
            else None
            for doc in documents
        ]

        relevant_pages = get_relevant_pages(golden)

        metrics_by_k = {
            str(k): calculate_metrics(pages, relevant_pages, k)
            for k in CUTOFFS
        }

        results.append({
            "id": golden.get("id"),
            "question": question,
            "relevant_pages": sorted(relevant_pages),
            "retrieved_pages_at_5": pages,
            "metrics_by_k": metrics_by_k
        })

        print(f"Processed {i}/{len(goldens)} questions")

    # Calculate average metrics across all questions
    aggregate = {}

    for k in CUTOFFS:
        aggregate[str(k)] = {
            metric: mean(
                result["metrics_by_k"][str(k)][metric]
                for result in results
            )
            for metric in [
                "recall_at_k",
                "mrr_at_k",
                "precision_at_k"
            ]
        }

    # Save results
    output = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "golden_dataset": str(golden_path),
        "retriever": "BM25 + Chroma hybrid (no cross-encoder reranker)",
        "retrieval_k": RETRIEVAL_K,
        "query_count": len(results),
        "metric_averaging": "macro average across questions",
        "aggregate_metrics_by_k": aggregate,
        "per_query_results": results
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RESULTS_DIR / "retriever_metrics.json"
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )

    # Print results
    print("\nAverage Retrieval Metrics")

    for k in CUTOFFS:
        scores = aggregate[str(k)]

        print(
            f"K={k}: "
            f"Recall@{k}={scores['recall_at_k']:.4f} | "
            f"MRR@{k}={scores['mrr_at_k']:.4f} | "
            f"Precision@{k}={scores['precision_at_k']:.4f}"
        )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
