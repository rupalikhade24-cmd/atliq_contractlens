"""The review workflow as a LangGraph state graph (PRD section 9.3).

    pii_gate -> extract -> rules -> retrieve -> (clause_scan || conflict_check) -> citation_check -> assemble

Who decides what:
- code decides everything with a right answer: entity and law, missing documents, register
  collisions (recall-first), severity (looked up from the playbook by rule_id);
- the model reads the contract: it extracts fields, finds clauses below AtliQ's standard and
  explains register collisions, always quoting the contract;
- the citation check drops any model finding whose quote is not in the source, word for word;
- a person decides: accept, dismiss with a reason, or send to counsel. Nothing is signed or sent.
"""

from __future__ import annotations

import operator
import re
import uuid
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

import config
import history
import register
import rules
from chain import call_json, model_name
from citations import find_quote
from dataset import load_docs, load_tracker
from workflow_log import Trace

_LLM_OVERRIDE: dict[str, Any] = {"llm": None}   # tests inject a fake model here


class ReviewState(TypedDict, total=False):
    doc_id: str
    text: str
    tracker: dict
    session_id: str
    status: str
    pii: list
    extracted: dict
    rule_findings: list
    bundle: dict
    collisions: list
    draft_context: dict
    ai_findings: Annotated[list, operator.add]
    conflict_notes: Annotated[list, operator.add]
    dropped: Annotated[list, operator.add]
    errors: Annotated[list, operator.add]
    steps: Annotated[list, operator.add]
    brief: dict


def _trace(state) -> Trace:
    return Trace(state.get("session_id", "review"), state.get("tracker", {}).get("contract_id", ""))


def _llm_ready() -> bool:
    return bool(_LLM_OVERRIDE["llm"] or config.GROQ_API_KEY)


# --------------------------------------------------------------- nodes

def pii_gate(state: ReviewState) -> dict:
    """Stop before any model call if the upload contains real personal data."""
    found = rules.pii_gate(state["text"])
    step = _trace(state).add("Upload", f"{state.get('doc_id', 'upload')}, personal data: "
                                       f"{', '.join(found) if found else 'none'}")
    if found:
        return {"status": "blocked_personal_data", "pii": found, "steps": [step]}
    return {"status": "contract_received", "pii": [], "steps": [step]}


EXTRACT_SYSTEM = """You extract fields from a contract for a review tool. Reply with one JSON object, \
no prose:
{"counterparty": str, "atliq_entity_named": str, "governing_law": str, "contract_value": str,
 "services_summary": str (one sentence), "locations": [countries or cities where the work or
 the client's properties are], "industries": [str], "personal_data": bool, "patient_data": bool}
Use only what the contract says. Use "" or [] when it is not stated."""


def extract(state: ReviewState) -> dict:
    if not _llm_ready():
        return {"extracted": {}, "errors": ["AI extraction not run: no GROQ_API_KEY"],
                "steps": [_trace(state).add("Extract", "skipped, no API key")]}
    try:
        data = call_json(EXTRACT_SYSTEM, state["text"][:15000], max_tokens=500, fast=True,
                         llm=_LLM_OVERRIDE["llm"])
        data = data if isinstance(data, dict) else {}
        return {"extracted": data, "status": "extracted",
                "steps": [_trace(state).add("Extract", f"{len(data)} fields")]}
    except Exception as e:
        return {"extracted": {}, "errors": [f"Extraction failed: {e}"],
                "steps": [_trace(state).add("Extract", "failed")]}


def run_rules(state: ReviewState) -> dict:
    text, tracker = state["text"], state.get("tracker", {})
    found = (rules.entity_law_checks(text, tracker) + rules.nda_mutuality(text, tracker)
             + rules.missing_attachments(text))
    bundle = rules.bundle_check(tracker, text)
    deal_names = [load_tracker()[m]["counterparty"] for m in bundle.get("members", []) if m in load_tracker()]
    hits = register.collisions(text, tracker, deal_names)
    step = _trace(state).add("Rules", f"entity/law {len(found)} flags, bundle "
                                      f"{len(bundle.get('findings', []))} missing, register "
                                      f"{len(hits)} entries matched")
    return {"rule_findings": found + bundle.get("findings", []), "bundle": bundle,
            "collisions": hits, "status": "checks_running", "steps": [step]}


def _passages(text: str, terms: list[str], n: int = 4) -> list[str]:
    """Draft paragraphs that mention the register entry's trigger terms (keyword retrieval,
    works for uploaded contracts that are not in the vector store)."""
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n(?=\s*\d+\.\d)", text) if len(p.strip()) > 40]
    scored = []
    for p in paras:
        score = sum(1 for t in terms if re.search(r"\b" + t + r"\b", p, re.I))
        if score:
            scored.append((score, p[:900]))
    return [p for _, p in sorted(scored, key=lambda x: -x[0])[:n]]


def retrieve(state: ReviewState) -> dict:
    """Recall-first: every matched register entry gets the draft passages most related to it."""
    ctx = {}
    for hit in state.get("collisions", []):
        t = hit["entry"].get("triggers") or {}
        terms = [k for g in t.get("all_groups", []) for k in g] + t.get("any", []) + t.get("deal", [])
        terms += ["rate", "hour", r"\$"] if "rates_below" in t else []
        ctx[hit["entry"]["id"]] = _passages(state["text"], terms)
    step = _trace(state).add("Retrieve", f"{len(ctx)} register entries, "
                                         f"{sum(len(v) for v in ctx.values())} draft passages")
    return {"draft_context": ctx, "steps": [step]}


def _playbook_for_prompt(role: str) -> str:
    lines = []
    for rid, r in rules.playbook().items():
        if r["side"] in ("any", role) or (role in ("nda", "partner") and r["side"] == "supplier"):
            lines.append(f"- {rid}: {r['label']}. AtliQ standard: {r['standard']}")
    return "\n".join(lines)


SCAN_SYSTEM = """You review a contract draft for AtliQ, an IT services company with no legal team. \
AtliQ's role in this contract: {role}.

Find clauses that fall below AtliQ's standard positions listed here (rule_id: label. standard):
{playbook}

{fairness}
Reply with a JSON list only. Each item:
{{"rule_id": one rule_id from the list, "clause": clause number as written,
  "quote": the exact words from the contract, copied character for character, 15-300 characters,
  "why": one or two plain sentences on what this means for AtliQ,
  "ask_for": the change to ask for, based on the standard}}
Rules: quote must be copied exactly from the contract text, never paraphrased. Do not judge \
severity. Flag at most 8 items for this part, most important first. Add at most 2 ONLY-LOOKS-RISKY items for \
clauses that sound alarming but meet the standard. If nothing is below standard, reply []."""

FAIRNESS = {
    "buyer": "AtliQ is the BUYER here (its own paper, sent to a vendor or freelancer). AtliQ wants to "
             "treat them as fairly as it asks clients to treat AtliQ: flag clauses that would be "
             "unacceptable if a client imposed them on AtliQ, using UNFAIR-TO-COUNTERPARTY.",
    "nda": "This is an NDA. Check whether it is genuinely mutual: does it bind both parties equally? "
           "Also flag any non-compete, non-solicit or residuals clause hidden inside it (NEW-RESTRICTION).",
    "partner": "This is a partnership agreement. Flag exclusivity, non-solicit and one-sided "
               "termination as NEW-RESTRICTION or TERM-ASYMMETRIC.",
    "supplier": "AtliQ is the SUPPLIER here, usually on the client's paper. Also flag any new "
                "non-compete, exclusivity, most-favoured pricing or non-solicit as NEW-RESTRICTION.",
}


def _split(text: str, size: int) -> list[str]:
    """Split on blank lines into parts under `size` characters, so a long contract fits Groq's
    free-tier tokens-per-minute limit. Each part is scanned separately."""
    parts, cur = [], ""
    for para in re.split(r"(\n\s*\n)", text):
        if len(cur) + len(para) > size and cur.strip():
            parts.append(cur)
            cur = ""
        cur += para
    if cur.strip():
        parts.append(cur)
    return parts


def clause_scan(state: ReviewState) -> dict:
    if not _llm_ready():
        return {"errors": ["AI clause scan not run: no GROQ_API_KEY"],
                "steps": [_trace(state).add("LLM clause scan", "skipped, no API key")]}
    role = rules.atliq_role(state.get("tracker", {}))
    system = SCAN_SYSTEM.format(role=role, playbook=_playbook_for_prompt(role),
                                fairness=FAIRNESS.get(role, ""))
    items = []
    parts = _split(state["text"], config.SCAN_CHARS)
    for i, part in enumerate(parts, start=1):
        try:
            got = call_json(system, f"(Part {i} of {len(parts)} of the contract)\n\n{part}",
                            max_tokens=config.MAX_TOKENS, llm=_LLM_OVERRIDE["llm"])
            items += got if isinstance(got, list) else []
        except Exception as e:
            return {"errors": [f"AI clause scan failed on part {i} of {len(parts)}: {e}"],
                    "steps": [_trace(state).add("LLM clause scan", "failed")]}
    found, seen = [], set()
    for it in items:
        if not isinstance(it, dict):
            continue
        sig = (str(it.get("rule_id", "")).upper(), " ".join(str(it.get("quote", "")).split()).lower())
        if sig in seen:          # the same clause flagged from two overlapping parts
            continue
        seen.add(sig)
        rid = str(it.get("rule_id", "OTHER")).upper()
        rid = rid if rid in rules.playbook() else "OTHER"
        pb = rules.playbook()[rid]
        found.append(rules.finding(rid, pb["label"], quote=str(it.get("quote", "")),
                                   clause=str(it.get("clause", "")), why=str(it.get("why", "")),
                                   origin="ai") | {"ask_for": str(it.get("ask_for", "")),
                                                   "standard": pb["standard"]})
    return {"ai_findings": found, "steps": [_trace(state).add("LLM clause scan", f"{len(found)} findings")]}


CONFLICT_SYSTEM = """You check whether a contract draft collides with obligations AtliQ already \
signed. For each register entry you get its exact wording and the draft passages most related to it.

Reply with a JSON list only, one item per register entry:
{"register_id": str, "assessment": "conflict" | "possible" | "unlikely",
 "draft_quote": exact words from a draft passage that create the collision (copied character for character, or "" if none),
 "why": two plain sentences: what the signed contract forbids and what the draft asks for}
Say "possible" when it depends on facts not in the text (affiliates, territory, dates, waivers). \
You cannot clear a flag: a person decides."""


def conflict_check(state: ReviewState) -> dict:
    hits = state.get("collisions", [])
    if not hits:
        return {"steps": [_trace(state).add("LLM conflict check", "nothing to check")]}
    if not _llm_ready():
        return {"errors": ["AI conflict explanation not run: no GROQ_API_KEY (flags still shown)"],
                "steps": [_trace(state).add("LLM conflict check", "skipped, no API key")]}
    blocks = []
    for h in hits:
        e = h["entry"]
        passages = state.get("draft_context", {}).get(e["id"], [])
        blocks.append(f"REGISTER {e['id']} ({e['type']}, {e['status']}, until {e.get('active_until') or 'no end'}): "
                      f"{e['title']}\nScope: {e['scope']}\nBound: {e['bound']}\nSigned wording: "
                      + " | ".join(e["quotes"]) + "\nDraft passages:\n" + "\n...\n".join(passages or ["(none found)"]))
    try:
        items = call_json(CONFLICT_SYSTEM, "\n\n=====\n\n".join(blocks), max_tokens=1500,
                          llm=_LLM_OVERRIDE["llm"])
        items = items if isinstance(items, list) else []
    except Exception as e:
        return {"errors": [f"AI conflict explanation failed: {e} (flags still shown)"],
                "steps": [_trace(state).add("LLM conflict check", "failed")]}
    return {"conflict_notes": [i for i in items if isinstance(i, dict)],
            "steps": [_trace(state).add("LLM conflict check", f"{len(items)} explained")]}


def citation_check(state: ReviewState) -> dict:
    """FR-15: drop any model finding whose quote is not in the draft word for word."""
    kept, dropped = [], []
    for f in state.get("ai_findings", []):
        (kept if find_quote(f["quote"], state["text"]) else dropped).append(f)
    notes = []
    for n in state.get("conflict_notes", []):
        if n.get("draft_quote") and not find_quote(n["draft_quote"], state["text"]):
            n = {**n, "draft_quote": "", "quote_dropped": True}
        notes.append(n)
    step = _trace(state).add("Citation", f"{len(kept)} kept, {len(dropped)} dropped")
    # replace (not add) the lists: operator.add would append, so return the delta via brief
    return {"brief": {"ai_kept": kept, "conflict_notes": notes}, "dropped": dropped,
            "status": "citation_checked", "steps": [step]}


def _precedent(f: dict) -> list[dict]:
    """RAG over negotiation_notes.md and meeting notes: what did AtliQ do last time?"""
    try:
        from retriever import cite, search
        docs = search(f"{f['title']} {f.get('quote', '')[:200]}", collections=("notes",), k=2)
        return [{"cite": cite(d), "text": d.page_content.split("\n", 1)[-1][:400]} for d in docs]
    except Exception:
        return []


def not_checked() -> list[str]:
    """Be explicit about what the register could NOT see (signed contracts with no file)."""
    in_files = {d.tracker_id for d in load_docs() if d.kind == "signed"}
    missing = [f"{r['contract_id']} {r['counterparty']} ({r['doc_type']})"
               for r in load_tracker().values()
               if r.get("status") == "Signed" and r["contract_id"] not in in_files]
    return missing


def assemble(state: ReviewState) -> dict:
    tracker = state.get("tracker", {})
    brief_in = state.get("brief", {})
    findings = list(state.get("rule_findings", []))
    notes = {n.get("register_id"): n for n in brief_in.get("conflict_notes", [])}
    for h in state.get("collisions", []):
        e = h["entry"]
        n = notes.get(e["id"], {})
        findings.append({
            "rule_id": "REGISTER-" + e["id"], "title": e["title"], "severity": h["severity"],
            "quote": e["quotes"][0], "source_id": e["source"], "clause": e["clause"],
            "why": (n.get("why") or f"Flagged because the draft {h['why']}. Scope: {e['scope']}.")
                   + f" Binds {e['bound']}, until {e.get('active_until') or 'no end date'} ({e['status']})."
                   + "".join(f" Related: {r['title']}." for r in h.get("related", [])),
            "draft_quote": n.get("draft_quote", ""), "assessment": n.get("assessment", "not assessed"),
            "ask_for": e.get("action", ""), "origin": "register", "evidence": e.get("evidence", []),
        })
    findings += brief_in.get("ai_kept", [])
    order = {s: i for i, s in enumerate(rules.SEVERITY_ORDER)}
    findings.sort(key=lambda f: order.get(f["severity"], 9))
    for i, f in enumerate(findings, start=1):
        f["id"] = f"F{i}"
        if f["origin"] != "register" and f["severity"] in ("blocker", "high"):
            f["precedent"] = _precedent(f)
    counts = {s: sum(1 for f in findings if f["severity"] == s) for s in rules.SEVERITY_ORDER}
    ai_done = not any("clause scan" in e for e in state.get("errors", []))
    if counts["blocker"]:
        verdict = "Do not sign as drafted"
    elif counts["high"]:
        verdict = "Needs changes before signing"
    else:
        verdict = "No blocker or high flags in what was checked"
    counsel = any(f["severity"] == "blocker" and (f["origin"] == "register" or
                  f["rule_id"] in ("PHI-OFFSHORE", "BUNDLE-MISSING", "LIAB-UNLIMITED")) for f in findings)
    brief = {
        "contract_id": tracker.get("contract_id", ""), "doc_id": state.get("doc_id", ""),
        "counterparty": tracker.get("counterparty", ""), "doc_type": tracker.get("doc_type", ""),
        "verdict": verdict, "counts": counts, "findings": findings,
        "bundle": state.get("bundle", {}), "extracted": state.get("extracted", {}),
        "ai_completed": ai_done and _llm_ready(), "errors": state.get("errors", []),
        "dropped_count": len(state.get("dropped", [])), "recommend_counsel": counsel,
        "not_checked": not_checked(), "role": rules.atliq_role(tracker),
        "history": history.check(state["text"], tracker, state.get("doc_id", "")),
        "model": model_name() if _llm_ready() else "",
    }
    step = _trace(state).add("Brief", f"{len(findings)} flags ({counts['blocker']} blocker, "
                                      f"{counts['high']} high), ai_completed={brief['ai_completed']}")
    return {"brief": brief, "status": "brief_ready" if brief["ai_completed"] else "review_not_completed",
            "steps": [step]}


# --------------------------------------------------------------- graph

def build_graph():
    g = StateGraph(ReviewState)
    for name, fn in (("pii_gate", pii_gate), ("extract", extract), ("rules", run_rules),
                     ("retrieve", retrieve), ("clause_scan", clause_scan),
                     ("conflict_check", conflict_check), ("citation_check", citation_check),
                     ("assemble", assemble)):
        g.add_node(name, fn)
    g.add_edge(START, "pii_gate")
    g.add_conditional_edges("pii_gate", lambda s: END if s["status"] == "blocked_personal_data" else "extract")
    g.add_edge("extract", "rules")
    g.add_edge("rules", "retrieve")
    g.add_edge("retrieve", "clause_scan")       # these two run in parallel
    g.add_edge("retrieve", "conflict_check")
    g.add_edge(["clause_scan", "conflict_check"], "citation_check")
    g.add_edge("citation_check", "assemble")
    g.add_edge("assemble", END)
    return g.compile()


def run_review(text: str, tracker: dict, doc_id: str = "upload", session_id: str | None = None,
               llm=None) -> dict:
    _LLM_OVERRIDE["llm"] = llm
    try:
        state = build_graph().invoke({"doc_id": doc_id, "text": text, "tracker": tracker,
                                      "session_id": session_id or uuid.uuid4().hex[:12]})
    finally:
        _LLM_OVERRIDE["llm"] = None
    return state
