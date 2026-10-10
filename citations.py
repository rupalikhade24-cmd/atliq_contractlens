"""Citation validator (PRD FR-15): a finding is shown only if its quote appears verbatim in the source.

Tolerated differences are formatting only: markdown bold markers, curly vs straight quotes,
dashes and whitespace/line breaks. Wording, numbers and punctuation must match exactly.
"""

from __future__ import annotations

import re

_TRANS = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'",
                        "–": "-", "—": "-", " ": " "})


def normalize(s: str) -> str:
    s = (s or "").translate(_TRANS).replace("**", "").replace("__", "")
    return re.sub(r"\s+", " ", s).strip()


def find_quote(quote: str, text: str) -> bool:
    q = normalize(quote).strip(" .…")
    if len(q) < 12:          # too short to prove anything
        return False
    return q.lower() in normalize(text).lower()


def validate(findings: list, sources: dict) -> tuple:
    """Keep findings whose quote is found in the named source text. Returns (kept, dropped)."""
    kept, dropped = [], []
    for f in findings:
        src = sources.get(f.get("source_id") or "draft", "")
        (kept if find_quote(f.get("quote", ""), src) else dropped).append(f)
    return kept, dropped
