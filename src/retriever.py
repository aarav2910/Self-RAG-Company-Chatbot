"""Equal-weight BM25 and Chroma hybrid retrieval."""

import importlib.util
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langsmith import traceable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "document_chunks"
EMBEDDING_MODEL = "text-embedding-3-small"
RETRIEVAL_K = 5


@traceable(run_type="chain", name="initialize_hybrid_retriever")
def create_hybrid_retriever(
    k: int = RETRIEVAL_K,
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
):
    """Create an EnsembleRetriever with BM25 and Chroma weighted equally."""
    if importlib.util.find_spec("rank_bm25") is None:
        raise RuntimeError(
            "BM25Retriever requires rank_bm25. Install it with: pip install rank_bm25"
        )

    try:
        import chromadb
        from langchain_classic.retrievers import EnsembleRetriever
        from langchain_community.retrievers import BM25Retriever
        from langchain_community.vectorstores import Chroma
    except ImportError as exc:
        raise RuntimeError(
            "Hybrid retrieval dependencies are missing. Install chromadb, "
            "langchain-community, langchain-classic, langchain-openai, and rank_bm25."
        ) from exc

    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing from the environment or .env file")

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    stored = collection.get(include=["documents", "metadatas"])

    texts = stored["documents"] or []
    metadatas = stored["metadatas"] or [{} for _ in texts]
    documents = [
        Document(page_content=text, metadata=metadata or {})
        for text, metadata in zip(texts, metadatas)
        if text
    ]
    if not documents:
        raise ValueError(
            f'Chroma collection "{COLLECTION_NAME}" is empty. '
            "Run doc-preprocessing.py to add documents first."
        )

    bm25_retriever = BM25Retriever.from_documents(documents, k=k)
    vector_store = Chroma(
        client=chroma_client,
        collection_name=COLLECTION_NAME,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    )
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": k})

    if bm25_weight < 0 or vector_weight < 0 or bm25_weight + vector_weight <= 0:
        raise ValueError("Retriever weights must be non-negative and have a positive sum.")

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[bm25_weight, vector_weight],
    )


if __name__ == "__main__":
    retriever = create_hybrid_retriever()
    query = input("Search query: ").strip()
    for result in retriever.invoke(query):
        print(f"\n{result.metadata}\n{result.page_content}")
