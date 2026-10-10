"""Keyword (BM25) search over the same clause chunks the Chroma index uses.

Used where a local embedding model can't run, such as a Vercel serverless function: no model
download, no index on disk, built in memory from the dataset on first use. Same interface as
the Chroma path in retriever.search, including the `where` metadata filter.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache

from langchain_core.documents import Document

_WORD = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
_STOP = set("""a an and are as at be by for from has have in is it its of on or that the this to
was were will with shall may any such which who whom what when where how not no than then their
there these those our we you your i me my he she they them his her""".split())


def tokens(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in _STOP and len(t) > 1]


class BM25:
    def __init__(self, docs: list[Document], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.tf = [Counter(tokens(d.page_content)) for d in docs]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = (sum(self.len) / len(self.len)) if self.len else 1.0
        df = Counter(t for c in self.tf for t in c)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def score(self, i: int, q: list[str]) -> float:
        tf, dl, s = self.tf[i], self.len[i], 0.0
        for t in q:
            f = tf.get(t)
            if f:
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
        return s

    def search(self, query: str, k: int, where: dict | None = None) -> list[Document]:
        q = tokens(query)
        hits = [(self.score(i, q), i) for i, d in enumerate(self.docs) if _match(d.metadata, where)]
        hits = [h for h in hits if h[0] > 0]
        hits.sort(key=lambda h: -h[0])
        return [self.docs[i] for _, i in hits[:k]]


def _match(meta: dict, where: dict | None) -> bool:
    """Supports the Chroma filters this app uses: {"field": value} and {"field": {"$in": [...]}}."""
    for key, cond in (where or {}).items():
        val = meta.get(key)
        if isinstance(cond, dict) and "$in" in cond:
            if val not in cond["$in"]:
                return False
        elif val != cond:
            return False
    return True


@lru_cache(maxsize=None)
def index(collection: str) -> BM25:
    from ingest_contracts import COLLECTIONS, build_documents
    kind = next(k for k, c in COLLECTIONS.items() if c == collection)
    return BM25(build_documents(kind)[0])


def count(collection: str) -> int:
    return len(index(collection).docs)
