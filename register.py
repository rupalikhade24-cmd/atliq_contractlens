"""Obligations register: what AtliQ already promised in signed contracts, and the collision check.

The collision check is deterministic and recall-first (PRD: missing a real conflict is the worst
error). It flags a draft whenever an entry's triggers match; the model only explains the match,
it can never remove a flag.
"""

from __future__ import annotations

import json
import re
from datetime import date
from functools import lru_cache

import config
from citations import find_quote
from dataset import get_doc

SEED = config.DATA_DIR / "register_seed.json"
CONFIRMED = config.REVIEWS_DIR / "register_confirmed.json"   # entries a person confirmed in the app


def _load_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def _verify(entry: dict) -> dict:
    """Check every quote against its source file. Unverified entries are never used for flags."""
    e = dict(entry)
    try:
        src = get_doc(e["source"]).text
        missing = [q for q in e.get("quotes", []) if not find_quote(q, src)]
        for ev in e.get("evidence", []):
            if not find_quote(ev["quote"], get_doc(ev["source"]).text):
                missing.append(ev["quote"])
    except KeyError:
        missing = ["source file not found"]
    e["verified"] = not missing
    e["unverified_quotes"] = missing
    until = e.get("active_until")
    e["status"] = "expired" if until and date.fromisoformat(until) < config.TODAY else "active"
    return e


@lru_cache(maxsize=1)
def load_register() -> tuple:
    entries = _load_json(SEED, {"entries": []})["entries"]
    entries += _load_json(CONFIRMED, [])
    return tuple(_verify(e) for e in entries)


def reload_register():
    load_register.cache_clear()


def save_confirmed(entry: dict):
    """A proposed entry counts only after a person confirms it (PRD: register update gate)."""
    config.REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    current = _load_json(CONFIRMED, [])
    current = [e for e in current if e["id"] != entry["id"]] + [entry]
    CONFIRMED.write_text(json.dumps(current, indent=2), encoding="utf-8")
    reload_register()


# ---------------------------------------------------------------- collision check

ROLE_ALIASES = {
    "senior data engineer": ["senior data engineer", "sr data engineer", "sr. data engineer", "senior de", "sr de"],
    "bi developer": ["bi developer", "power bi developer", "bi engineer"],
    "data engineer": ["data engineer"],
}


def hourly_rates(text: str) -> dict:
    """Pull role -> lowest $/hour from rate tables and notes ("$72/hr senior DE")."""
    found = {}
    for line in text.splitlines():
        if "$" not in line:
            continue
        low = line.lower()
        for role, aliases in ROLE_ALIASES.items():   # senior first, so "data engineer" isn't double-counted
            hit = next((a for a in aliases if a in low), None)
            if not hit:
                continue
            m = re.search(r"\$\s?(\d{2,3}(?:\.\d+)?)", line)
            if m:
                found[role] = min(found.get(role, 1e9), float(m.group(1)))
            low = low.replace(hit, "")
    return found


def _kw(pattern: str, text: str) -> str:
    """The text that matched (e.g. "Hotels"), or "" if none."""
    m = re.search(r"\b" + pattern + r"\b", text, re.I)
    return m.group(0) if m else ""


def collisions(draft_text: str, tracker: dict, deal_counterparties=()) -> list:
    """Register entries this draft may collide with. Every entry is checked; each result says why."""
    haystack = "\n".join([draft_text, tracker.get("notes", ""), tracker.get("counterparty", ""),
                          tracker.get("client_country", "")])
    counterparty = tracker.get("counterparty", "")
    doc_type = tracker.get("doc_type", "").lower()
    out = []
    for e in load_register():
        t = e.get("triggers")
        if not t or not e["verified"]:
            continue
        why = []
        if "deal" in t:
            names = " ".join([counterparty, *deal_counterparties])
            hits = [k for k in t["deal"] if k.lower() in names.lower()]
            if not hits:
                continue
            why.append("is part of the " + ", ".join(hits) + " deal")
        if "any" in t:
            hits = [k for k in t["any"] if _kw(k, haystack)]
            if not hits:
                continue
            why.append("mentions " + ", ".join(hits))
        if "doc_types" in t and not any(k in doc_type for k in t["doc_types"]):
            continue
        if "not_counterparty" in t and any(k.lower() in counterparty.lower() for k in t["not_counterparty"]):
            continue
        if "all_groups" in t:
            groups = [[m for m in (_kw(k, haystack) for k in g) if m] for g in t["all_groups"]]
            if not all(groups):
                continue
            why.append("mentions " + " + ".join(f'"{g[0]}"' for g in groups))
        if "rates_below" in t:
            rates = hourly_rates(haystack)
            lower = {r: v for r, v in rates.items() if r in t["rates_below"] and v < t["rates_below"][r]}
            if not lower:
                continue
            why.append("; ".join(f"{r} ${v:g}/hr vs ${t['rates_below'][r]}/hr" for r, v in lower.items()))
        severity = e["severity"]
        direct = t.get("direct_if_counterparty_matches")
        if direct and not any(k in counterparty.lower() for k in direct):
            severity = e.get("severity_if_indirect", severity)
        if e["status"] == "expired":
            severity = "info"
        related = [r for r in load_register() if r["id"] in e.get("related", []) and not r.get("triggers")]
        out.append({"entry": e, "why": "; ".join(why), "severity": severity, "related": related})
    return out


# ---------------------------------------------------------------- proposals (model-assisted)

PROPOSE_SYSTEM = """You build an obligations register for AtliQ from one SIGNED contract. List every \
obligation that limits what AtliQ (or its affiliates) may do for OTHER clients or partners, or that \
outlives the work: non-compete, exclusivity, most-favoured pricing, IP handed over, non-solicit or \
no-hire, data-location limits, insurance minimums.

Reply with a JSON list only. Each item:
{"type": str, "title": short plain-English summary, "clause": clause number,
 "quote": exact words from the contract, copied character for character, 20-400 characters,
 "scope": who/what/where it covers, "active_until": "YYYY-MM-DD" or "" if not stated}
Use only what the contract says. If there are none, reply []."""


def propose_from_signed(doc_id: str, llm=None) -> tuple[list, list]:
    """Model proposes entries; only quotes found verbatim survive. Nothing counts until confirmed."""
    from chain import call_json

    text = get_doc(doc_id).text
    items = call_json(PROPOSE_SYSTEM, text[:60000], max_tokens=1500, llm=llm)
    items = items if isinstance(items, list) else []
    kept, dropped = [], []
    for it in items:
        if isinstance(it, dict) and find_quote(str(it.get("quote", "")), text):
            kept.append(it)
        else:
            dropped.append(it)
    return kept, dropped
