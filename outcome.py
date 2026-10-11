"""Turn a decided review into the email Karandeep sends, plus an internal note for counsel.

Deterministic: every sentence is either the draft's own words (already citation-checked) or a
counter-proposal written in the playbook / register. The model writes nothing here, and other
clients' names and terms never reach the client email. Nothing is sent; the person copies it.
"""

from __future__ import annotations

import json
import re

import config
import register
import rules
from decisions import open_items
from workflow_log import Trace

GROUPS = [("blocker", "Must change before we can sign"),
          ("high", "We need these changed"),
          ("medium", "We propose"),
          ("low", "Smaller points")]
QUOTE_CHARS = 300
# Headings for rule ids that have no playbook label.
LABELS = {"LAW-FOREIGN": "Governing law", "ENTITY-MISMATCH": "AtliQ contracting entity",
          "INVOICE-ENTITY-MISMATCH": "Invoicing entity", "BUNDLE-MISSING": "Related agreement"}
# Deal documents AtliQ signs with its own people (freelancers, Pune team) are not the client's business.
INTERNAL_DOCS = ("Subcontractor", "Intercompany", "Sub-processor")


def _positions() -> dict:
    data = json.loads((config.DATA_DIR / "playbook.json").read_text(encoding="utf-8"))
    return data.get("client_positions", {})


def _clause_key(f: dict) -> tuple:
    nums = re.findall(r"\d+", f.get("clause") or "")
    return tuple(int(n) for n in nums) if nums else (999,)


def _short(quote: str) -> str:
    q = " ".join((quote or "").replace("**", "").split())
    return q if len(q) <= QUOTE_CHARS else q[:QUOTE_CHARS].rsplit(" ", 1)[0] + " …"


def client_position(f: dict, positions: dict | None = None) -> str:
    """AtliQ's ask for this flag, safe to send to the other side; '' if it must stay internal."""
    positions = positions if positions is not None else _positions()
    if f["origin"] == "register":
        rid = f["rule_id"].removeprefix("REGISTER-")
        entry = next((e for e in register.load_register() if e["id"] == rid), {})
        return entry.get("client_position", "")
    text = positions.get(f["rule_id"], "")
    item = f["title"].split(":", 1)[1].strip() if ":" in f["title"] else f["title"]
    if f["rule_id"] == "BUNDLE-MISSING" and any(k in item for k in INTERNAL_DOCS):
        return ""
    return text.replace("{item}", item)


def _label(f: dict) -> str:
    """Heading for one point in the client email. Register titles name other clients, so not those."""
    if f["origin"] == "register":
        return "Scope"
    if f["rule_id"] == "MISSING-ATTACHMENT":
        return f["title"].split(" is ")[0]
    pb = rules.playbook().get(f["rule_id"], {})
    return LABELS.get(f["rule_id"]) or pb.get("label") or f["title"]


def build(record: dict, by: str = "Karandeep", session_id: str = "outcome") -> dict:
    """Client email + counsel note from a review whose blocker and high flags all have a decision."""
    remaining = open_items(record)
    if remaining:
        raise ValueError("Decide these first: " + ", ".join(remaining))
    positions = _positions()
    decided = record.get("decisions", {})
    counterparty = record["brief"].get("counterparty") or "the counterparty"
    doc_type = record["brief"].get("doc_type") or "agreement"

    groups = {s: [] for s, _ in GROUPS}
    internal, counsel, left_out = [], [], []
    for f in record["brief"]["findings"]:
        action = decided.get(f["id"], {}).get("action", "")
        if f["severity"] == "info" or action == "dismiss":
            left_out.append(f)
            continue
        ask = client_position(f, positions)
        if action == "counsel":
            counsel.append(f)
        elif not ask:
            internal.append(f)
        if ask and f["severity"] in groups:
            groups[f["severity"]].append((f, ask, action == "counsel"))

    lines, n = [], 0
    for sev, heading in GROUPS:
        items = sorted(groups[sev], key=lambda x: _clause_key(x[0]))
        if not items:
            continue
        lines += ["", heading]
        for f, ask, with_counsel in items:
            n += 1
            where = f" (clause {f['clause']})" if f.get("clause") and f["origin"] != "register" else ""
            point = f"{n}. {_label(f)}{where}."
            if f["origin"] != "register" and f.get("quote") and f.get("source_id", "draft") == "draft":
                point += f' The draft says: “{_short(f["quote"])}”'
            point += " " + ask
            if with_counsel:
                point += " Our legal team will follow up on this point."
            lines.append(point)

    subject = f"AtliQ comments on the {counterparty} {doc_type} draft"
    if n:
        body = [f"Dear {counterparty} team,", "",
                f"Thank you for the draft {doc_type}. We have reviewed it and would like the points below "
                "addressed before we sign."] + lines + ["", "Happy to walk through these on a call.", "",
                                                         "Regards,", by]
    else:
        body = [f"Dear {counterparty} team,", "", f"Thank you for the draft {doc_type}. We have reviewed it "
                "and have no changes to request.", "", "Regards,", by]

    note = [f"Internal note: {counterparty} {doc_type} ({record.get('key', '')}). Not for the client.", ""]
    if counsel:
        note.append("For outside counsel:")
        for f in counsel:
            note += _note_item(f)
    if internal:
        note.append("Accepted, but not in the client email (needs your own wording or is internal):")
        for f in internal:
            note += _note_item(f)
    if left_out:
        note.append("Left out of the email: " + ", ".join(
            f"{f['id']} ({'dismissed' if decided.get(f['id'], {}).get('action') == 'dismiss' else 'looks risky, within standard'})"
            for f in left_out))
    Trace(session_id, record.get("key", "")).add(
        "Outcome prepared", f"by {by}: {n} points in client email, {len(counsel)} for counsel")
    return {"subject": subject, "client_email": "\n".join(body),
            "counsel_note": "\n".join(note).strip(), "points": n,
            "counsel_count": len(counsel), "internal_count": len(internal)}


def _note_item(f: dict) -> list[str]:
    src = "this draft" if f.get("source_id", "draft") == "draft" else f"signed: {f['source_id']}"
    clause = f" cl. {f['clause']}" if f.get("clause") else ""
    out = [f"- {f['id']} [{f['severity']}] {f['title']}"]
    if f.get("quote"):
        out.append(f'  Quote ({src}{clause}): "{_short(f["quote"])}"')
    if f.get("draft_quote"):
        out.append(f'  In the draft: "{_short(f["draft_quote"])}"')
    if f.get("why"):
        out.append(f"  Why: {f['why']}")
    if f.get("ask_for"):
        out.append(f"  Suggested action: {f['ask_for']}")
    return out + [""]
