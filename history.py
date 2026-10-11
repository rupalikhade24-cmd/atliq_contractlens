"""Has this contract been here before? Deterministic; nothing here calls a model.

A contract counts as the same one when either
- its text is largely the same as a file already in the dataset (word 5-gram overlap), or
- it is for the same client and the same document type as another file or tracker row.

The brief shows the result as "Already signed on <date>" and "Seen N times before".
"""

from __future__ import annotations

import re
from functools import lru_cache

from dataset import load_docs, load_tracker

SAME_TEXT = 0.85        # overlap at or above this: the same contract
SIMILAR_TEXT = 0.5      # overlap at or above this: largely the same text

LEGAL_SUFFIXES = {"inc", "ltd", "pvt", "llc", "llp", "gmbh", "plc", "co", "corp", "limited", "private",
                  "the", "and"}
DOC_TYPES = ["subcontractor baa", "subcontractor agreement", "contractor agreement", "partnership agreement",
             "pilot agreement", "services agreement", "staff aug", "waiver letter", "settlement letter",
             "vendor agreement", "baa", "dpa", "nda", "msa", "sow"]


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def norm_party(name: str) -> str:
    """'Al Noor Hospitality Group' -> 'al noor hospitality group'; legal suffixes and '(intercompany)' dropped."""
    name = re.sub(r"\(.*?\)", " ", name or "")
    return " ".join(w for w in _words(name) if w not in LEGAL_SUFFIXES)


def same_party(a: str, b: str) -> bool:
    """Same client if one name is the other name or starts with it ('Al Noor' / 'Al Noor Hospitality')."""
    a, b = norm_party(a), norm_party(b)
    if not a or not b:
        return False
    short, long_ = sorted((a, b), key=len)
    return long_ == short or long_.startswith(short + " ")


def norm_type(doc_type: str) -> str:
    """'Mutual NDA' and 'NDA' are the same type; 'SOW-1' and 'SOW-2' are not."""
    t = " ".join(w for w in _words(doc_type) if w not in ("mutual", "draft", "signed"))
    for k in DOC_TYPES:
        if t.startswith(k):
            rest = t[len(k):].strip()
            return k + (" " + rest.split()[0] if rest and rest.split()[0].isdigit() else "")
    return t


def _shingles(text: str, n: int = 5) -> frozenset:
    w = _words(text)
    return frozenset(" ".join(w[i:i + n]) for i in range(max(len(w) - n + 1, 1)))


@lru_cache(maxsize=256)
def _doc_shingles(doc_id: str, text: str) -> frozenset:
    return _shingles(text)


def overlap(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def guess_from_filename(stem: str) -> dict:
    """For a file the tracker doesn't know: '2026-10-11_al_noor_hospitality_msa' ->
    counterparty 'al noor hospitality', doc_type 'msa', date '2026-10-11'."""
    m = re.match(r"(\d{4}-\d{2}-\d{2})_(.*)", stem)
    when, rest = (m.group(1), m.group(2)) if m else ("", stem)
    words = [w for w in rest.split("_") if w not in ("draft", "signed", "scanned", "copy", "test", "v1", "v2")]
    text = " ".join(words)
    for k in sorted(DOC_TYPES, key=len, reverse=True):
        for v in (k, "mutual " + k):
            if text.endswith(" " + v):
                return {"counterparty": text[: -len(v) - 1], "doc_type": k, "date": when}
    return {"counterparty": text, "doc_type": "", "date": when}


def _doc_facts(d) -> dict:
    """Who/what/when for a dataset file, from its tracker row or, failing that, its file name."""
    t, g = d.tracker, guess_from_filename(d.doc_id)
    signed = d.kind == "signed" or t.get("status") == "Signed"
    return {
        "contract_id": t.get("contract_id", ""), "doc_id": d.doc_id,
        "counterparty": t.get("counterparty") or g["counterparty"],
        "doc_type": t.get("doc_type") or g["doc_type"],
        "status": "Signed" if signed else (t.get("status") or "Draft"),
        "date": (t.get("signed_date") if signed else t.get("received_date")) or g["date"],
        "where": "signed contracts folder" if d.kind == "signed" else "incoming folder",
    }


def check(text: str, tracker: dict, doc_id: str = "") -> dict:
    """Every earlier copy of this contract in the dataset and tracker, and whether one is signed."""
    me_id = tracker.get("contract_id", "")
    doc_id = re.sub(r"\.(md|txt)$", "", doc_id or "")   # an upload of a dataset file is that file, not a repeat
    party = tracker.get("counterparty") or guess_from_filename(doc_id)["counterparty"]
    dtype = norm_type(tracker.get("doc_type") or guess_from_filename(doc_id)["doc_type"])
    mine = _shingles(text)
    matches, seen_ids = [], set()

    for d in load_docs():
        if d.kind == "note" or d.doc_id == doc_id:
            continue
        f = _doc_facts(d)
        if me_id and f["contract_id"] == me_id:
            continue
        score = overlap(mine, _doc_shingles(d.doc_id, d.text))
        named = same_party(party, f["counterparty"]) and dtype and norm_type(f["doc_type"]) == dtype
        if score >= SIMILAR_TEXT:
            how = f"same text ({score:.0%} match)" if score >= SAME_TEXT else f"largely the same text ({score:.0%} match)"
        elif named:
            how = "same client and document type"
        else:
            continue
        matches.append({**f, "how": how, "text_match": round(score, 2)})
        seen_ids.add(f["contract_id"])

    for row in load_tracker().values():   # tracker rows with no file (e.g. the duplicate C-010 / C-011)
        cid = row["contract_id"]
        if cid == me_id or cid in seen_ids:
            continue
        if same_party(party, row.get("counterparty", "")) and dtype and norm_type(row.get("doc_type", "")) == dtype:
            signed = row.get("status") == "Signed"
            matches.append({"contract_id": cid, "doc_id": "", "counterparty": row["counterparty"],
                            "doc_type": row["doc_type"], "status": row.get("status") or "not recorded",
                            "date": (row.get("signed_date") if signed else row.get("received_date")) or "",
                            "where": "tracker only (no file)", "how": "same client and document type",
                            "text_match": None})

    matches.sort(key=lambda m: m["date"] or "9999")
    signed = [m for m in matches if m["status"] == "Signed"]
    return {
        "times_seen": len(matches),
        "already_signed": bool(signed),
        "signed_on": [m["date"] for m in signed if m["date"]],
        "matches": matches,
    }


def summary(h: dict) -> str:
    """One line for the brief, or '' if this contract is new."""
    if not h or not h.get("times_seen"):
        return ""
    parts = []
    if h["already_signed"]:
        dates = h["signed_on"]
        parts.append("Already signed" + (" on " + ", ".join(dates) if dates else ""))
    n = h["times_seen"]
    parts.append(f"Seen {n} time{'s' if n != 1 else ''} before")
    return ". ".join(parts) + "."
