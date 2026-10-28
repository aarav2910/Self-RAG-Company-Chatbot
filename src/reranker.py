"""Cross-encoder reranking for the BM25 + Chroma hybrid retriever."""

from langsmith import traceable

from retriever import RETRIEVAL_K, create_hybrid_retriever

FINAL_K = 5
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(self, hybrid_retriever, cross_encoder):
        self.hybrid_retriever = hybrid_retriever
        self.cross_encoder = cross_encoder

    @traceable(run_type="retriever", name="cross_encoder_reranker")
    def invoke(self, query: str):
        # Use 10 hybrid results as candidates, then return the best 5.
        candidates = self.hybrid_retriever.invoke(query)[:RETRIEVAL_K]
        if not candidates:
            return []

        pairs = [(query, document.page_content) for document in candidates]

        @traceable(run_type="chain", name="cross_encoder_scoring")
        def score_candidates(candidate_pairs):
            return self.cross_encoder.predict(candidate_pairs)

        scores = score_candidates(pairs)
        ranked = sorted(
            zip(scores, candidates),
            key=lambda result: result[0],
            reverse=True,
        )
        return [document for _, document in ranked[:FINAL_K]]


import time

@traceable(run_type="chain", name="initialize_reranker")
def create_reranker():
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "Cross-encoder reranking requires sentence-transformers. "
            "Install it with: pip install sentence-transformers"
        ) from exc

    start = time.perf_counter()
    hybrid_retriever = create_hybrid_retriever(k=RETRIEVAL_K)
    print(f"Hybrid retriever initialization: {time.perf_counter() - start:.2f}s")

    start = time.perf_counter()
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    print(f"Cross-encoder initialization: {time.perf_counter() - start:.2f}s")

    return CrossEncoderReranker(hybrid_retriever, cross_encoder)


if __name__ == "__main__":
    reranker = create_reranker()
    query = input("Search query: ").strip()
    for result in reranker.invoke(query):
        print(f"\n{result.metadata}\n{result.page_content}")
