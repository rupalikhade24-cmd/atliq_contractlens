"""Load the capstone dataset into three Chroma collections: signed, incoming, notes.

Contracts are chunked by clause, not by fixed size, so every retrieved chunk cites exactly
one clause ("<document> §<clause>"). Re-running is idempotent.

    python ingest_contracts.py
"""

from __future__ import annotations

import re

from langchain_core.documents import Document

from dataset import load_docs

COLLECTIONS = {"signed": "signed", "incoming": "incoming", "note": "notes"}

CLAUSE_RE = re.compile(r"^\s*(?:\*\*)?(?:§\s*)?(\d{1,2}(?:\.\d{1,2}){0,2})[.)]?\s")
HEADING_RE = re.compile(r"^\s*#{1,4}\s+(.*)")
MAX_CHARS = 1800


def chunk_document(doc) -> list[Document]:
    """One chunk per clause; the party name and section heading are prepended for retrieval."""
    chunks: list[Document] = []
    buf: list[str] = []
    clause, heading = "", ""

    def flush() -> None:
        text = "\n".join(buf).strip()
        if not text or HEADING_RE.fullmatch(text):
            return
        prefix = f"[{doc.title} | {heading}]\n" if heading else f"[{doc.title}]\n"
        chunks.append(Document(page_content=prefix + text, metadata={
            "doc_id": doc.doc_id, "kind": doc.kind, "title": doc.title,
            "tracker_id": doc.tracker_id, "counterparty": doc.tracker.get("counterparty", ""),
            "clause": clause, "heading": heading}))

    for line in doc.text.splitlines():
        h, c = HEADING_RE.match(line), CLAUSE_RE.match(line)
        if h or c:
            flush()
            buf = []
            if h:
                heading, clause = h.group(1).strip("* "), ""
            if c:
                clause = c.group(1)
        buf.append(line)
        if sum(len(b) for b in buf) > MAX_CHARS:
            flush()
            buf = []
    flush()
    return chunks


def build_documents(kind: str) -> tuple[list[Document], list[str]]:
    docs, ids = [], []
    for d in load_docs():
        if d.kind != kind:
            continue
        for i, chunk in enumerate(chunk_document(d)):
            docs.append(chunk)
            ids.append(f"{d.doc_id}-{i:03d}")
    return docs, ids


def main() -> None:
    from vectorstore import replace_collection
    for kind, collection in COLLECTIONS.items():
        docs, ids = build_documents(kind)
        written = replace_collection(collection, docs, ids)
        print(f"[{collection}] ingested {written} clause chunks")


if __name__ == "__main__":
    main()
