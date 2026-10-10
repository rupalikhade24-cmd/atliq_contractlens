"""Retrieval over the three collections (same shape as the RAG chatbot's merged retriever)."""

from __future__ import annotations

from langchain_core.documents import Document

import config
from ingest_contracts import COLLECTIONS

LABELS = {"signed": "SIGNED", "incoming": "DRAFT", "notes": "NOTES"}


def cite(doc: Document) -> str:
    m = doc.metadata
    return f"{m['doc_id']} §{m['clause']}" if m.get("clause") else m["doc_id"]


def search(query: str, collections=("signed", "notes"), k: int | None = None,
           where: dict | None = None) -> list[Document]:
    """Top-k per collection, merged in collection order. `where` is a Chroma metadata filter."""
    k = k or config.TOP_K
    out: list[Document] = []
    if config.SEARCH_BACKEND == "keyword":
        from keyword_search import index
        for name in collections:
            out.extend(index(name).search(query, k, where))
        return out
    from vectorstore import get_store
    for name in collections:
        store = get_store(name)
        out.extend(store.similarity_search(query, k=k, filter=where) if where
                   else store.similarity_search(query, k=k))
    return out


def format_context(docs: list[Document]) -> str:
    """Numbered, source-labelled context. The model cites by number; the app maps numbers back."""
    if not docs:
        return "(no relevant documents were retrieved)"
    return "\n\n---\n\n".join(
        f"[{i}] {LABELS.get(COLLECTIONS.get(d.metadata.get('kind'), ''), 'SOURCE')} | {cite(d)}\n"
        f"{d.page_content.strip()}" for i, d in enumerate(docs, start=1))


def describe_sources(docs: list[Document]) -> list[dict]:
    seen, out = set(), []
    for i, d in enumerate(docs, start=1):
        key = (cite(d), d.page_content[:80])
        if key in seen:
            continue
        seen.add(key)
        out.append({"n": i, "cite": cite(d), "title": d.metadata.get("title", ""),
                    "excerpt": d.page_content.strip()})
    return out
