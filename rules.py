"""Deterministic checks: anything with a right answer is decided by code, not by the model (PRD section 5).

- entity and governing law (FR-12)
- deal bundle completeness: NDA + BAA + MSA + subcontractor BAAs, DPAs (FR-13)
- referenced annexes / exhibits that are not attached
- personal-data gate that runs before any model call
- guardrail for requests to sign, approve, send or give a legal opinion (FR-17)
- playbook: rule_id -> severity (FR-16)
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

import config
from dataset import load_docs, load_tracker

US = {"usa", "usa?", "us", "united states"}
EU = {"germany", "france", "netherlands", "ireland", "spain", "italy", "belgium", "austria",
      "poland", "sweden", "denmark", "finland", "portugal"}

SEVERITY_ORDER = ["blocker", "high", "medium", "low", "info"]


@lru_cache(maxsize=1)
def playbook() -> dict:
    return json.loads((config.DATA_DIR / "playbook.json").read_text(encoding="utf-8"))["rules"]


def severity_for(rule_id: str) -> str:
    """FR-16: severity is looked up, never taken from the model."""
    return playbook().get(rule_id, playbook()["OTHER"])["severity"]


def finding(rule_id: str, title: str, quote: str = "", clause: str = "", why: str = "",
            severity: str | None = None, source_id: str = "draft", origin: str = "rule") -> dict:
    return {"rule_id": rule_id, "title": title, "quote": quote.strip(), "clause": clause,
            "why": why, "severity": severity or severity_for(rule_id),
            "source_id": source_id, "origin": origin}


# --------------------------------------------------------------- role, entity, law

def atliq_role(tracker: dict) -> str:
    t = tracker.get("doc_type", "").lower()
    if any(k in t for k in ("contractor", "subcontractor", "vendor")):
        return "buyer"
    if "partner" in t or "referral" in t:
        return "partner"
    if "nda" in t:
        return "nda"
    return "supplier"


_INC = re.compile(r"\bAtliQ,?\s+Inc\b\.?", re.I)
_PVT = re.compile(r"\bAtliQ\s+Techn[o0]logies\s+(?:Pvt\.?\s+Ltd|Private\s+Limited)\b", re.I)


def entity_named(text: str) -> tuple[str, str]:
    """The AtliQ entity in the parties block: whichever appears first. Returns (entity, line)."""
    hits = []
    for name, rx in (("AtliQ Inc", _INC), ("AtliQ Technologies Pvt Ltd", _PVT)):
        m = rx.search(text)
        if m:
            hits.append((m.start(), name))
    if not hits:
        return "", ""
    pos, name = min(hits)
    line = text[text.rfind("\n", 0, pos) + 1: text.find("\n", pos)]
    return name, line


def governing_law(text: str) -> tuple[str, str]:
    """Returns (jurisdiction bucket: india | us | foreign | '', the sentence)."""
    for line in text.splitlines():
        low = line.lower()
        if line.lstrip().startswith("#") or not re.search(r"governed by|governing law", low):
            continue
        if re.search(r"governed by (this agreement|the msa|the master|the intercompany|the parties'|"
                     r"the independent)", low) and "law" not in low.split("governed by")[1][:40]:
            continue
        if not re.search(r"\blaws? (of|and regulations)\b|\(laws of", low):
            continue
        if re.search(r"\bindia\b", low):
            bucket = "india"
        elif re.search(r"state of [a-z]|united states|u\.s\. federal|delaware|texas|illinois|ohio|"
                       r"arizona|new jersey|new york|california", low):
            bucket = "us"
        else:
            bucket = "foreign"
        return bucket, re.sub(r"^\s*[\d.()a-z]*\s*(\*\*[^*]+\*\*)?\s*", "", line).strip()
    return "", ""


def expected_entity(tracker: dict) -> str:
    return "AtliQ Inc" if tracker.get("client_country", "").strip().lower() in US \
        else "AtliQ Technologies Pvt Ltd"


def entity_law_checks(text: str, tracker: dict) -> list[dict]:
    out = []
    role = atliq_role(tracker)
    entity, entity_line = entity_named(text)
    country = tracker.get("client_country", "") or "unknown"
    internal = "atliq" in tracker.get("counterparty", "").lower()   # intercompany paper
    if role in ("supplier", "buyer") and entity and not internal and entity != expected_entity(tracker):
        out.append(finding(
            "ENTITY-MISMATCH", f"Wrong AtliQ entity: {entity} named for a {country} counterparty",
            quote=entity_line, why=f"{expected_entity(tracker)} should contract with a {country} "
            "counterparty (finance fact sheet). The entity decides who invoices, the tax regime "
            "and the bank account.", severity="blocker"))
    inv = re.search(r"[Ii]nvoices?[^.\n]{0,80}?(?:raised|issued) by (AtliQ[^,.\n]*)", text)
    if inv and entity:
        inv_entity = "AtliQ Inc" if _INC.search(inv.group(1)) else "AtliQ Technologies Pvt Ltd"
        if inv_entity != entity:
            out.append(finding(
                "INVOICE-ENTITY-MISMATCH", "Contracting and invoicing entities differ",
                quote=inv.group(0), why=f"Parties clause names {entity}, invoices come from "
                f"{inv_entity}. This mismatch delayed payments twice in 2025.", severity="high"))
    bucket, law_line = governing_law(text)
    if entity and bucket:
        home = "us" if entity == "AtliQ Inc" else "india"
        if bucket != home:
            out.append(finding(
                "LAW-FOREIGN", f"{entity} would sit under {'non-US' if home == 'us' else 'non-Indian'} law",
                quote=law_line, why="AtliQ has no presence in that jurisdiction. Some past deals "
                "accepted local law (Northwind: English law, Al Noor: DIFC); Brightwater shows the cost "
                "when nobody notices.", severity="high" if home == "us" else "medium"))
    return out


# --------------------------------------------------------------- NDA mutuality

def nda_mutuality(text: str, tracker: dict) -> list[dict]:
    """An NDA is mutual only if Confidential Information flows both ways in its definition."""
    if atliq_role(tracker) != "nda":
        return []
    m = re.search(r"[^\n]*Confidential Information\W{0,6} means[^\n]*", text)
    if not m:
        return []
    definition = m.group(0)
    if re.search(r"Disclosing Party|one Party|a Party|either Party|each Party|by a Party", definition, re.I):
        return []
    titled_mutual = "mutual" in (tracker.get("doc_type", "") + text[:600]).lower()
    return [finding("NDA-NOT-MUTUAL",
                    "NDA titled mutual, but only AtliQ's obligations are defined" if titled_mutual
                    else "One-way NDA: only AtliQ is bound",
                    quote=definition, why="Confidential Information is defined as what the other side "
                    "discloses to AtliQ, so AtliQ's own architecture, pricing and code shared in the "
                    "walkthrough are not protected. AtliQ signs mutual NDAs only (GlobalMart, Mar 2026).")]


# --------------------------------------------------------------- attachments

def missing_attachments(text: str) -> list[dict]:
    """Annexes / exhibits the draft says form part of it, but which are not in the file."""
    out = []
    refs = set(re.findall(r"\b(Annex(?:ure)?|Exhibit|Schedule)\s+([A-Z0-9]{1,2})\b", text))
    headings = "\n".join(l for l in text.splitlines() if l.lstrip().startswith("#") or
                         re.match(r"^\s*\*\*(annex|exhibit|schedule)", l, re.I))
    for kind, ref in sorted(refs):
        if re.search(rf"\b{kind}\s+{ref}\b", headings, re.I):
            continue
        m = re.search(rf"[^\n]*\b{kind}\s+{ref}\b[^\n]*", text)
        if m and re.search(r"attached|forms? (an integral )?part|annexed|incorporated", m.group(0), re.I):
            line = re.sub(r"^\s*\d+(\.\d+)*\s*", "", m.group(0)).strip()
            out.append(finding("MISSING-ATTACHMENT", f"{kind} {ref} is referenced but not attached",
                               quote=line[:400], why="You would be signing terms you have not seen.",
                               severity="high"))
    return out


# --------------------------------------------------------------- deal bundles

def _first_word(name: str) -> str:
    return re.split(r"[\s,]+", name.strip())[0] if name else ""


def deal_members(tracker: dict, text: str = "") -> list[dict]:
    """Tracker rows that belong to the same deal: same counterparty, rows whose notes say
    'for <counterparty>', and the counterparty a buyer-side draft is staffing."""
    rows = load_tracker()
    key = _first_word(tracker.get("counterparty", ""))
    if not key:
        return []
    members = {r["contract_id"]: r for r in rows.values()
               if _first_word(r["counterparty"]).lower() == key.lower()
               or re.search(rf"\bfor {re.escape(key)}\b", r.get("notes", ""), re.I)}
    # freelancer / agency drafts that name this client belong to its deal
    if atliq_role(tracker) == "supplier":
        for d in load_docs():
            if d.kind == "incoming" and atliq_role(d.tracker) == "buyer" \
                    and re.search(rf"\b{re.escape(key)}\b", d.text):
                members[d.tracker_id] = rows[d.tracker_id]
    # a buyer-side draft (freelancer, agency) belongs to the client deal it names
    if atliq_role(tracker) == "buyer" and text:
        for r in rows.values():
            ck = _first_word(r["counterparty"])
            if ck and ck.lower() != key.lower() and atliq_role(r) == "supplier" \
                    and r.get("status") in ("In review", "Awaiting client", "") \
                    and re.search(rf"\b{re.escape(ck)}\b", text):
                members.update({x["contract_id"]: x for x in deal_members(r)})
    return sorted(members.values(), key=lambda r: r["contract_id"])


def _status(rows: list[dict], pattern: str) -> tuple[str, str]:
    for r in rows:
        if re.search(pattern, r.get("doc_type", ""), re.I):
            return (r.get("status") or "No status"), r["contract_id"]
    return "Missing", ""


def bundle_check(tracker: dict, text: str) -> dict:
    """What documents this deal needs, and which exist. Returns {type, items, findings}."""
    members = deal_members(tracker, text)
    client = next((r for r in members if atliq_role(r) == "supplier"), tracker)
    country = (client.get("client_country") or tracker.get("client_country", "")).strip().lower()
    client_docs = {d.tracker_id: d.text for d in load_docs() if d.kind == "incoming"}
    all_text = text + client_docs.get(client.get("contract_id", ""), "") + \
        " ".join(r.get("notes", "") for r in members)
    is_health = bool(re.search(r"HIPAA|Protected Health Information|\bPHI\b|\bBAA\b", all_text)) or \
        any("BAA" in r.get("doc_type", "") for r in members)
    is_eu = country in EU and bool(re.search(r"GDPR|personal data", all_text, re.I))
    items, findings = [], []
    if not members:
        return {"type": "", "items": items, "findings": findings}

    client_name = client["counterparty"]
    people = [r for r in members if atliq_role(r) == "buyer"]

    if is_health and country in US:
        dtype = "US healthcare (patient data)"
        for label, pat in (("Mutual NDA", r"NDA"), ("Services agreement (MSA)", r"MSA"),
                           ("HIPAA Business Associate Agreement", r"^BAA$")):
            st, cid = _status([r for r in members if r["counterparty"] == client_name], pat)
            items.append({"doc": label, "status": st, "tracker_id": cid})
        # who touches the data: every freelancer / agency on the deal, plus the Pune team
        sub_baas = [r for r in load_tracker().values() if "Subcontractor BAA" in r.get("doc_type", "")]
        for p in people:
            if "NDA" in p.get("doc_type", ""):
                continue
            has = any(_first_word(p["counterparty"]).lower() in (s["counterparty"] + s.get("notes", "")).lower()
                      for s in sub_baas)
            items.append({"doc": f"Subcontractor BAA: {p['counterparty']}",
                          "status": "Signed" if has else "Missing", "tracker_id": ""})
        has_pune = any(_first_word(client_name).lower() in (s.get("notes", "") + s.get("file_location", "")).lower()
                       for s in sub_baas)
        items.append({"doc": f"Intercompany subcontractor BAA (Pune team) for {client_name}",
                      "status": "Signed" if has_pune else "Missing",
                      "tracker_id": "" if has_pune else "C-044 covers CareBridge only"})
    elif is_eu:
        dtype = "EU personal data (GDPR)"
        st, cid = _status([r for r in members if r["counterparty"] == client_name], r"DPA")
        if st == "Missing" and re.search(r"Data Processing Agreement", text):
            st = "Referenced, not attached"
        items.append({"doc": "GDPR Art. 28 Data Processing Agreement with the client",
                      "status": st, "tracker_id": cid})
        for p in people:
            items.append({"doc": f"Sub-processor DPA + client consent: {p['counterparty']}",
                          "status": "Missing", "tracker_id": ""})
    else:
        return {"type": "", "items": [], "findings": []}

    for it in items:
        if it["status"] in ("Missing", "Referenced, not attached"):
            findings.append(finding(
                "BUNDLE-MISSING", f"Deal is missing: {it['doc']}",
                why=f"{dtype} deals need this signed before any data changes hands. "
                    + (it["tracker_id"] if it["tracker_id"] else ""), severity="blocker"))
    return {"type": dtype, "items": items, "findings": findings,
            "members": [r["contract_id"] for r in members]}


# --------------------------------------------------------------- personal-data gate

_PII = [
    ("US Social Security number", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("medical record number", re.compile(r"\bMRN\s*[:#]?\s*\d{5,}", re.I)),
    ("date of birth value", re.compile(r"\b(?:DOB|date of birth)\s*[:=,]?\s*\d{1,4}[/-]\d{1,2}[/-]\d{1,4}", re.I)),
]
_ICD = re.compile(r"\b[A-TV-Z]\d{2}\.\d{1,4}\b")


def pii_gate(text: str) -> list[str]:
    """Runs before any model call. Returns the kinds of personal data found (empty = safe)."""
    found = [label for label, rx in _PII if rx.search(text)]
    if len(_ICD.findall(text)) >= 3:
        found.append("diagnosis codes")
    return found


# --------------------------------------------------------------- guardrail

_ACTION = re.compile(r"\b(sign|e-?sign|execute|approve|countersign|send|email|reject)\b.{0,40}"
                     r"\b(it|this|that|the|contract|agreement|nda|msa|sow|baa|draft|them)\b", re.I)
_OPINION = re.compile(r"\b(legal (advice|opinion)|is (this|it) legal|is (this|it) enforceable|"
                      r"will (we|i) win|can (we|i) be sued)\b", re.I)


def guardrail(question: str) -> str:
    """Return a refusal message, or '' if the question is fine to answer."""
    if _ACTION.search(question) and re.match(r"\s*(please\s+)?(can you\s+|could you\s+)?"
                                              r"(sign|e-?sign|execute|approve|countersign|send|email|reject)",
                                              question, re.I):
        return ("I can't sign, approve, reject or send a contract. Karandeep decides and signs; "
                "I can show the flags, the clauses behind them and what was agreed before.")
    if _OPINION.search(question):
        return ("I can't give a legal opinion. I can point to the clauses and past positions, and "
                "flag what should go to outside counsel (Mehta Rao for India/GCC, Calloway Stern for US).")
    return ""


# --------------------------------------------------------------- signed portfolio

def portfolio_check() -> list[dict]:
    """Entity/law check over everything already signed (Brightwater-style surprises)."""
    out = []
    for d in load_docs():
        if d.kind != "signed" or not d.tracker:
            continue
        for f in entity_law_checks(d.text, d.tracker):
            out.append({"doc_id": d.doc_id, "contract": d.title, **f})
    return out
