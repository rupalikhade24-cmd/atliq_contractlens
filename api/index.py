"""Web API for Atliq ContractLens on Vercel (Python serverless function).

Same review engine as the Streamlit app and the CLI: rules.py, register.py, review_graph.py
(LangGraph + Groq), citations.py. The page in public/index.html calls these endpoints.

Vercel functions keep no disk between requests, so the browser holds each review record
(brief + decisions) and sends it back with every decision; the server checks it and returns
the updated record. Nothing here signs, approves, rejects or sends a contract.

Run on the laptop:  uvicorn api.index:app --reload   (then open http://127.0.0.1:8000)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import config  # noqa: E402
import decisions  # noqa: E402
import register  # noqa: E402
import rules  # noqa: E402
from dataset import get_doc, incoming_queue, load_docs, load_tracker  # noqa: E402
from workflow_log import read_log  # noqa: E402

app = FastAPI(title="Atliq ContractLens")

PEOPLE = {"Karandeep": "CEO, decides and signs", "Bhavin": "founder, sales", "Dhaval": "founder, sales",
          "Jay": "sales", "Pranav": "CTO"}
DOC_TYPES = ["MSA", "SOW", "Mutual NDA", "NDA", "BAA", "Partnership Agreement", "Contractor Agreement",
             "Subcontractor Agreement"]


def _who(who: str) -> str:
    if who not in PEOPLE:
        raise HTTPException(400, "Pick who you are.")
    return who


def _ceo(who: str) -> bool:
    return _who(who) == "Karandeep"


def _need_ceo(who: str) -> None:
    if not _ceo(who):
        raise HTTPException(403, "Only Karandeep can do this.")


def _mine(who: str) -> list[dict]:
    """Sellers see their own deals only (PRD: never other clients' terms)."""
    rows = incoming_queue()
    return rows if _ceo(who) else [r for r in rows if r["doc"].tracker.get("requested_by") == who]


def _redact(record: dict, who: str) -> dict:
    """Sellers see that a register conflict exists, not the other client's terms."""
    if _ceo(who):
        return record
    rec = {**record, "brief": {**record["brief"]}}
    out = []
    for f in rec["brief"]["findings"]:
        if f["origin"] == "register":
            f = {k: f[k] for k in ("id", "rule_id", "severity", "origin") if k in f}
            f["title"] = "Conflicts with a commitment AtliQ made to another client"
            f["why"] = "Karandeep can see the details."
        out.append(f)
    rec["brief"]["findings"] = out
    return rec


# ------------------------------------------------------------------ page

@app.get("/")
def home():
    return FileResponse(ROOT / "public" / "index.html")


# ------------------------------------------------------------------ read

@app.get("/api/meta")
def meta():
    model, served = "", []
    if config.GROQ_API_KEY:
        from chain import available_models, model_name
        model, served = model_name(), available_models()
    return {"today": config.TODAY.isoformat(), "model": model, "models_for_this_key": served,
            "has_key": bool(config.GROQ_API_KEY), "people": PEOPLE, "doc_types": DOC_TYPES,
            "search": config.SEARCH_BACKEND, "on_vercel": config.ON_VERCEL}


@app.get("/api/queue")
def queue(who: str):
    rows = []
    for r in _mine(who):
        d = r["doc"]
        rows.append({"key": d.tracker_id or d.doc_id, "doc_id": d.doc_id, "title": d.title,
                     "counterparty": d.tracker.get("counterparty"), "doc_type": d.tracker.get("doc_type"),
                     "requested_by": d.tracker.get("requested_by"), "notes": d.tracker.get("notes") or "",
                     "deadline": r["deadline"].isoformat() if r["deadline"] else None,
                     "days_left": r["days_left"], "value_usd": d.tracker.get("value_usd") or ""})
    return rows


@app.get("/api/register")
def get_register(who: str):
    _need_ceo(who)
    signed = [{"doc_id": d.doc_id, "title": d.title} for d in load_docs() if d.kind == "signed"]
    return {"entries": list(register.load_register()), "signed": signed}


@app.get("/api/portfolio")
def portfolio(who: str):
    _need_ceo(who)
    in_files = {d.tracker_id for d in load_docs() if d.kind == "signed"}
    missing = [{"contract_id": r["contract_id"], "counterparty": r["counterparty"], "doc_type": r["doc_type"],
                "file_location": r.get("file_location") or "unknown"}
               for r in load_tracker().values() if r.get("status") == "Signed" and r["contract_id"] not in in_files]
    return {"flags": rules.portfolio_check(), "no_file": missing}


@app.get("/api/audit")
def audit(who: str):
    _need_ceo(who)
    return read_log()


# ------------------------------------------------------------------ review

class ReviewIn(BaseModel):
    who: str
    session_id: str = "web"
    key: str = ""                 # tracker id from the queue, or empty for an upload
    text: str = ""                # uploaded contract text
    filename: str = ""
    counterparty: str = ""
    client_country: str = "India"
    doc_type: str = "MSA"


@app.post("/api/review")
def review(body: ReviewIn):
    from review_graph import run_review

    who = _who(body.who)
    if body.key:
        row = next((r for r in _mine(who) if (r["doc"].tracker_id or r["doc"].doc_id) == body.key), None)
        if not row:
            raise HTTPException(404, "That draft isn't in your queue.")
        d = row["doc"]
        text, tracker, key, doc_id = d.text, d.tracker, d.tracker_id or d.doc_id, d.doc_id
    else:
        if not body.text.strip():
            raise HTTPException(400, "Upload a .md or .txt contract.")
        key = "upload-" + "".join(ch for ch in body.filename if ch.isalnum())[:40]
        tracker = {"counterparty": body.counterparty, "client_country": body.client_country,
                   "doc_type": body.doc_type, "requested_by": who, "contract_id": key, "notes": ""}
        text, doc_id = body.text, body.filename or key

    state = run_review(text, tracker, doc_id=doc_id, session_id=body.session_id)
    if state["status"] == "blocked_personal_data":
        return {"status": "blocked_personal_data", "pii": state["pii"], "steps": state.get("steps", [])}
    record = decisions.new_record(key, state["brief"], who)
    return {"status": state["status"], "steps": state.get("steps", []), "record": _redact(record, who)}


class DecideIn(BaseModel):
    who: str
    session_id: str = "web"
    record: dict
    finding_id: str = ""
    action: str = ""
    reason: str = ""


@app.post("/api/decide")
def decide(body: DecideIn):
    _need_ceo(body.who)
    try:
        return decisions.decide_record(body.record, body.finding_id, body.action, body.reason,
                                       body.who, body.session_id)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/complete")
def complete(body: DecideIn):
    _need_ceo(body.who)
    try:
        return decisions.complete_record(body.record, body.who, body.session_id)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/outcome")
def prepare_outcome(body: DecideIn):
    """Client email + counsel note from the decisions. Drafted only; the tool never sends it."""
    _need_ceo(body.who)
    import outcome
    try:
        return outcome.build(body.record, _who(body.who), body.session_id)
    except ValueError as e:
        raise HTTPException(400, str(e))


# ------------------------------------------------------------------ register proposals

class ProposeIn(BaseModel):
    who: str
    doc_id: str


@app.post("/api/register/propose")
def propose(body: ProposeIn):
    _need_ceo(body.who)
    if not config.GROQ_API_KEY:
        raise HTTPException(400, "Needs GROQ_API_KEY.")
    try:
        get_doc(body.doc_id)
    except KeyError:
        raise HTTPException(404, "Unknown signed contract.")
    kept, dropped = register.propose_from_signed(body.doc_id)
    return {"items": kept, "dropped": len(dropped)}


class ConfirmIn(BaseModel):
    who: str
    doc_id: str
    item: dict
    keywords: str = ""
    n: int = 1


@app.post("/api/register/confirm")
def confirm(body: ConfirmIn):
    _need_ceo(body.who)
    it, kw = body.item, body.keywords
    register.save_confirmed({
        "id": f"R-C{body.n}-{body.doc_id[:10]}", "type": it.get("type", ""), "title": it.get("title", ""),
        "source": body.doc_id, "clause": it.get("clause", ""), "bound": "AtliQ", "scope": it.get("scope", ""),
        "active_from": "", "active_until": it.get("active_until") or None, "quotes": [it.get("quote", "")],
        "severity": "high",
        "triggers": {"any": [k.strip() for k in kw.split(",") if k.strip()]} if kw.strip() else None,
        "confirmed_by": body.who})
    return {"ok": True}


# ------------------------------------------------------------------ ask

class AskIn(BaseModel):
    who: str
    question: str
    history: list[dict] = []


@app.post("/api/ask")
def ask(body: AskIn):
    who = _who(body.who)
    refusal = rules.guardrail(body.question)
    if refusal:
        return {"text": refusal, "sources": [], "bad": []}
    if not config.GROQ_API_KEY:
        return {"text": "Needs GROQ_API_KEY to answer questions.", "sources": [], "bad": []}
    from chain import answer
    kw = {} if _ceo(who) else {"collections": ("incoming",),
                               "where": {"tracker_id": {"$in": [r["doc"].tracker_id for r in _mine(who)] or ["none"]}}}
    try:
        text, sources, bad = answer(body.question, body.history[-6:], **kw)
    except Exception as e:  # model or network failure: say so, don't pretend
        return {"text": f"Couldn't answer: {e}", "sources": [], "bad": []}
    return {"text": text, "sources": sources, "bad": bad}
