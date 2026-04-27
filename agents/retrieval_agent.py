"""
Retrieval Agent — indexes documents with LlamaIndex and performs semantic search.

Uses a persistent VectorStoreIndex backed by a simple in-memory (or disk-cached)
vector store. On first run it indexes all .txt / .md / .pdf files in docs_dir.
"""

import os
import logging
from typing import List, Dict, Any

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core.schema import NodeWithScore
from llama_index.llms.anthropic import Anthropic as LlamaAnthropic
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from core.state import ResearchState

logger = logging.getLogger(__name__)

PERSIST_DIR = ".index_cache"
_index_cache: Dict[str, VectorStoreIndex] = {}


def _configure_llama_settings():
    """Point LlamaIndex at local embeddings + Anthropic LLM."""
    Settings.llm = LlamaAnthropic(
        model="claude-opus-4-5",
        api_key=os.getenv("ANTHROPIC_API_KEY", ""),
    )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )


def _build_or_load_index(docs_dir: str) -> VectorStoreIndex:
    """Return a cached or freshly-built VectorStoreIndex for docs_dir."""
    if docs_dir in _index_cache:
        return _index_cache[docs_dir]

    _configure_llama_settings()

    cache_path = os.path.join(PERSIST_DIR, docs_dir.replace("/", "_"))

    if os.path.exists(cache_path):
        logger.info("Loading index from cache: %s", cache_path)
        storage_context = StorageContext.from_defaults(persist_dir=cache_path)
        index = load_index_from_storage(storage_context)
    else:
        logger.info("Building new index from: %s", docs_dir)
        if not os.path.isdir(docs_dir):
            os.makedirs(docs_dir, exist_ok=True)

        reader = SimpleDirectoryReader(
            input_dir=docs_dir,
            recursive=True,
            required_exts=[".txt", ".md", ".pdf"],
        )
        documents = reader.load_data()
        logger.info("Loaded %d document(s).", len(documents))

        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        index.storage_context.persist(persist_dir=cache_path)

    _index_cache[docs_dir] = index
    return index


def _retrieve(query: str, docs_dir: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Return top-k chunks as dicts with text, score, and source."""
    index = _build_or_load_index(docs_dir)
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes: List[NodeWithScore] = retriever.retrieve(query)

    chunks = []
    for node in nodes:
        chunks.append({
            "text": node.node.get_content(),
            "score": float(node.score) if node.score is not None else 0.0,
            "source": node.node.metadata.get("file_name", "unknown"),
        })
    return chunks


async def retrieval_node(state: ResearchState, docs_dir: str = "data/sample_docs") -> ResearchState:
    """LangGraph node: retrieve relevant document chunks for the query."""
    query = state["query"]
    logs = list(state.get("agent_logs", []))
    logs.append(f"[RetrieverAgent] Searching for: '{query}'")

    try:
        chunks = _retrieve(query, docs_dir)
        logs.append(f"[RetrieverAgent] Retrieved {len(chunks)} chunk(s).")
        return {**state, "retrieved_chunks": chunks, "agent_logs": logs, "error": None}
    except Exception as exc:
        error_msg = f"[RetrieverAgent] ERROR: {exc}"
        logs.append(error_msg)
        logger.exception("Retrieval failed.")
        return {**state, "retrieved_chunks": [], "agent_logs": logs, "error": str(exc)}
