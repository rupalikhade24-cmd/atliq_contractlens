"""Shared embedding model + Chroma collection helpers (same pattern as the RAG chatbot).

The embedding model is loaded once per process. Set EMBEDDING_MODEL=fake to run the
offline tests without downloading a model.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document

import config


@lru_cache(maxsize=1)
def get_embeddings():
    if config.EMBEDDING_MODEL == "fake":
        from langchain_core.embeddings import DeterministicFakeEmbedding
        return DeterministicFakeEmbedding(size=384)
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=8)
def get_store(collection: str) -> Chroma:
    """Open (or create) a persisted collection under chroma_store/."""
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )


def count(collection: str) -> int:
    try:
        return get_store(collection)._collection.count()
    except Exception:
        return 0


def replace_collection(collection: str, documents: Sequence[Document], ids: Sequence[str]) -> int:
    """Idempotent ingest: wipe the collection, then write it fresh."""
    store = get_store(collection)
    existing = store._collection.get(include=[])["ids"]
    if existing:
        store._collection.delete(ids=existing)
    store.add_documents(list(documents), ids=list(ids))
    return len(documents)
