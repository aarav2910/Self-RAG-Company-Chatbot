"""Streamlit chat app for the existing Self-RAG pipeline."""

from __future__ import annotations

import atexit
import hashlib
import importlib
import importlib.util
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Company Handbook Copilot",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
SAMPLE_PATH = PROJECT_DIR / "docs" / "Sample_Employee_Handbook_Valve.pdf"
SUPPORTED_TYPES = {".pdf", ".txt", ".md", ".docx"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024   # per file
MAX_FILES = 10

SAMPLE_CHOICE = "Sample Employee Handbook"
UPLOAD_CHOICE = "Upload documents"
SUGGESTIONS = [
    "How does sick leave work?",
    "Who can authorize overtime?",
    "What holidays does the company observe?",
]
WELCOME = {
    "role": "assistant",
    "content": "**How can I help?**\n\nAsk about company policies, benefits, leave, or workplace procedures.",
}


# --------------------------------------------------------------------------- setup
@st.cache_resource(show_spinner=False)
def get_runtime_resources():
    """Create temporary storage once for this Streamlit server process."""
    root = Path(tempfile.mkdtemp(prefix="selfrag-streamlit-"))
    (root / "uploads").mkdir()
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    return root, threading.Lock()


APP_TEMP_DIR, REQUEST_LOCK = get_runtime_resources()
CHROMA_DIR = APP_TEMP_DIR / "chroma_db"


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@st.cache_resource(show_spinner=False)
def build_pipeline(api_key: str, document_paths: tuple[str, ...]):
    """Index all selected documents into ONE collection and load the Self-RAG graph.

    Cached, so it runs once per (API key, set of documents).
    """
    paths = [Path(p) for p in document_paths]
    for path in paths:
        if path.suffix.lower() not in SUPPORTED_TYPES:
            raise ValueError(f"{path.name}: choose a PDF, TXT, MD, or DOCX file.")
        if not path.is_file():
            raise FileNotFoundError(f"Document not found: {path.name}")

    os.environ["OPENAI_API_KEY"] = api_key
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    preprocessing = load_module(SRC_DIR / "doc-preprocessing.py", "streamlit_doc_preprocessing")
    preprocessing.CHROMA_DIR = CHROMA_DIR

    file_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    set_hash = hashlib.sha256("".join(sorted(file_hashes.values())).encode()).hexdigest()
    collection_name = f"docs_{set_hash[:24]}"
    preprocessing.COLLECTION_NAME = collection_name
    client, collection = preprocessing.create_clients()

    if collection.count() == 0:
        documents = []
        for path in paths:
            try:
                pages = preprocessing.load_file(path)
            except Exception as exc:
                raise RuntimeError(f"Could not read {path.name}: {exc}") from exc
            documents.append({"filename": path.name, "document_id": file_hashes[path], "pages": pages})
        preprocessing.store_all_documents(documents, client, collection)

    retriever_module = importlib.import_module("retriever")
    retriever_module.CHROMA_DIR = CHROMA_DIR
    retriever_module.COLLECTION_NAME = collection_name

    generator_module = importlib.import_module("generator")
    importlib.reload(generator_module)

    self_rag = load_module(SRC_DIR / "self-rag.py", "streamlit_self_rag")
    self_rag.CHROMA_DIR = CHROMA_DIR
    self_rag.COLLECTION_NAME = collection_name
    self_rag.retriever = None
    return self_rag, collection_name


def get_pipeline(api_key: str, document_paths: list[Path]):
    """Return the cached pipeline and point the shared retriever module at this document set."""
    os.environ["OPENAI_API_KEY"] = api_key
    key = tuple(sorted(str(path) for path in document_paths))
    self_rag, collection_name = build_pipeline(api_key, key)
    retriever_module = importlib.import_module("retriever")
    retriever_module.CHROMA_DIR = CHROMA_DIR
    retriever_module.COLLECTION_NAME = collection_name
    return self_rag


def save_uploaded_files(uploaded_files) -> list[Path]:
    """Validate and store every uploaded file (identical files are kept once)."""
    if len(uploaded_files) > MAX_FILES:
        raise ValueError(f"Upload at most {MAX_FILES} files at a time.")
    paths, seen = [], set()
    for uploaded in uploaded_files:
        name = Path(uploaded.name).name
        if Path(name).suffix.lower() not in SUPPORTED_TYPES:
            raise ValueError(f"{name}: upload a PDF, TXT, MD, or DOCX document.")
        content = uploaded.getvalue()
        if len(content) > MAX_UPLOAD_BYTES:
            raise ValueError(f"{name} is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
        digest = hashlib.sha256(content).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        folder = APP_TEMP_DIR / "uploads" / digest
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / name
        if not path.exists():
            path.write_bytes(content)
        paths.append(path)
    return paths


# --------------------------------------------------------------------------- helpers
def source_records(documents: list) -> list[dict]:
    records = []
    for doc in documents:
        metadata = getattr(doc, "metadata", {}) or {}
        records.append({
            "filename": metadata.get("filename", "Document"),
            "page": metadata.get("page_number"),
            "content": doc.page_content.strip(),
        })
    return records


def summarize_run(result: dict) -> dict:
    return {
        "retrieved": bool(result.get("need_retrieval")),
        "chunks": len(result.get("relevant_docs", [])),
        "rewrites": result.get("rewrite_tries", 0),
        "revisions": result.get("retries", 0),
        "checked": bool(result.get("context")),   # empty when no supporting passage was found
        "issup": result.get("issup", ""),
        "isuse": result.get("isuse", ""),
    }


def run_info_html(info: dict | None, running: bool = False) -> str:
    if running:
        body = "<p>Running Self-RAG…</p>"
    elif not info:
        body = "<p>Ask a question to see how the answer was checked.</p>"
    else:
        label = lambda value: value.replace("_", " ") or "–"
        rows = [("Retrieval", "used" if info["retrieved"] else "skipped (general answer)")]
        if info["retrieved"]:
            rows.append(("Relevant chunks", info["chunks"]))
            rows.append(("Query rewrites", info["rewrites"]))
            if info["checked"]:
                rows += [
                    ("Revisions", info["revisions"]),
                    ("Support check", label(info["issup"])),
                    ("Usefulness check", label(info["isuse"])),
                ]
            else:
                rows.append(("Result", "no supporting passage found"))
        body = "".join(
            f'<div class="info-row"><span>{key}</span><strong>{value}</strong></div>'
            for key, value in rows
        )
    return f'<div class="info-box"><div class="kicker">LAST ANSWER</div>{body}</div>'


def stream_text(text: str):
    """Stream the final verified answer in small visible chunks."""
    for start in range(0, len(text), 28):
        yield text[start:start + 28]
        time.sleep(0.012)


def show_sources(documents: list[dict]):
    with st.expander(f"Sources ({len(documents)})", expanded=False):
        for index, doc in enumerate(documents, start=1):
            page = f" · page {doc['page']}" if doc.get("page") else ""
            st.markdown(f"**{index}. {doc['filename']}{page}**")
            st.text(doc["content"])
            if index < len(documents):
                st.divider()


def render_message(message: dict):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("documents"):
            show_sources(message["documents"])


def queue_question(text: str):
    st.session_state.pending = text


STYLES = """
<style>
.block-container { max-width: 860px; padding-top: 1.6rem; padding-bottom: 5rem; }
[data-testid="stChatMessage"] { border: 1px solid rgba(128,128,128,.25); border-radius: 14px; padding: .8rem 1rem; }
div.stButton > button { width: 100%; border-radius: 9px; }
.topline { border-bottom: 1px solid rgba(128,128,128,.25); padding-bottom: 10px; margin-bottom: 14px; font-weight: 700; }
.topline span { float: right; font-size: 10px; font-weight: 600; opacity: .65; }
.brand-tag { display: inline-block; padding: 6px 10px; border-radius: 7px; background: #15372d; color: #73ddb4; font-size: 10px; font-weight: 800; letter-spacing: .12em; }
.side-title { margin: 22px 0 8px; font-size: 26px; font-weight: 800; line-height: 1.1; letter-spacing: -.03em; }
.side-copy { font-size: 12px; line-height: 1.6; opacity: .7; }
.info-box { padding: 12px 14px; border: 1px solid rgba(128,128,128,.3); border-radius: 12px; }
.info-box p { margin: 0; font-size: 12px; opacity: .75; }
.kicker { margin-bottom: 8px; font-size: 9px; font-weight: 700; letter-spacing: .12em; opacity: .6; }
.info-row { display: flex; justify-content: space-between; gap: 12px; padding: 5px 0; font-size: 12px; border-bottom: 1px solid rgba(128,128,128,.15); }
.info-row:last-child { border-bottom: 0; }
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)

st.session_state.setdefault("messages", [WELCOME.copy()])
st.session_state.setdefault("run_info", None)

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div class="brand-tag">SELF-RAG PROJECT</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-title">Company<br>Handbook<br>Copilot</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="side-copy">Clear answers from your company handbook, grounded in your documents.</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    api_key = st.text_input("OpenAI API key", type="password", placeholder="sk-…")
    document_choice = st.radio("Knowledge document", [SAMPLE_CHOICE, UPLOAD_CHOICE])
    uploaded_files = st.file_uploader(
        f"Upload up to {MAX_FILES} files (PDF, TXT, MD, DOCX)",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
        disabled=document_choice != UPLOAD_CHOICE,
    )
    upload_paths = []
    upload_index_ready = document_choice != UPLOAD_CHOICE
    if document_choice == SAMPLE_CHOICE:
        st.caption(f"Using {SAMPLE_PATH.name}")
    elif uploaded_files:
        st.caption(f"{len(uploaded_files)} file(s) selected. Questions search across all of them.")
        if not api_key:
            st.info("Add your OpenAI API key to start embedding these documents.")
        else:
            try:
                upload_paths = save_uploaded_files(uploaded_files)
                key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
                signature = hashlib.sha256(
                    (key_hash + "|" + "|".join(str(path) for path in upload_paths)).encode("utf-8")
                ).hexdigest()

                if st.session_state.get("indexed_upload_signature") != signature:
                    with st.status("Embedding uploaded documents…", expanded=True) as status:
                        status.write("Reading and indexing the selected files.")
                        with REQUEST_LOCK:
                            get_pipeline(api_key, upload_paths)
                        status.update(
                            label=f"Embedded {len(upload_paths)} document(s). Ready to chat.",
                            state="complete",
                        )
                    st.session_state.indexed_upload_signature = signature
                else:
                    st.success("Documents embedded and ready to chat.")
                upload_index_ready = True
            except Exception as exc:
                st.error(f"Could not embed the uploaded documents: {exc}")
    st.caption("Uploads and the temporary index are cleared when the app stops.")

    if st.button("Clear conversation"):
        st.session_state.messages = [WELCOME.copy()]
        st.session_state.run_info = None
        st.rerun()

    st.divider()
    info_panel = st.empty()
    info_panel.markdown(run_info_html(st.session_state.run_info), unsafe_allow_html=True)

needs_upload = document_choice == UPLOAD_CHOICE and not uploaded_files
ready = bool(api_key) and not needs_upload and upload_index_ready
if not api_key:
    placeholder = "Add your OpenAI API key in the sidebar to start"
elif needs_upload:
    placeholder = "Upload at least one document in the sidebar to start"
else:
    placeholder = (
        "Ask a question about the handbook…" if document_choice == SAMPLE_CHOICE
        else "Ask a question about your documents…"
    )

# --------------------------------------------------------------------------- chat
st.markdown(
    '<div class="topline">● Company Handbook Copilot<span>PRIVATE KNOWLEDGE</span></div>',
    unsafe_allow_html=True,
)

prompt = st.chat_input(placeholder, disabled=not ready)
question = (prompt or st.session_state.pop("pending", None) or "").strip()
if question:
    st.session_state.messages.append({"role": "user", "content": question})

# History (including the question just asked) is drawn first, so the new answer always lands below it.
for message in st.session_state.messages:
    render_message(message)

if not question and len(st.session_state.messages) == 1 and document_choice == SAMPLE_CHOICE:
    st.caption("TRY ASKING")
    for column, suggestion in zip(st.columns(len(SUGGESTIONS)), SUGGESTIONS):
        column.button(suggestion, key=f"suggest_{suggestion}", disabled=not ready,
                      on_click=queue_question, args=(suggestion,))

if question:
    documents: list[dict] = []
    with st.chat_message("assistant"):
        try:
            document_paths = upload_paths if document_choice == UPLOAD_CHOICE else [SAMPLE_PATH]
            info_panel.markdown(run_info_html(None, running=True), unsafe_allow_html=True)
            with st.spinner("Searching your documents and checking the answer…"):
                with REQUEST_LOCK:
                    result = get_pipeline(api_key, document_paths).invoke_self_rag(question)
            answer = result.get("answer") or "No answer found."
            documents = source_records(result.get("relevant_docs", []))
            st.write_stream(stream_text(answer))
            if documents:
                show_sources(documents)
            st.session_state.run_info = summarize_run(result)
        except Exception as exc:
            answer = f"I couldn't complete that request: {exc}"
            st.error(answer)
    info_panel.markdown(run_info_html(st.session_state.run_info), unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": answer, "documents": documents})
