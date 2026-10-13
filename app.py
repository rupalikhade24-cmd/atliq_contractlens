"""Streamlit UI for Atliq ContractLens.

Pages: review queue, contract review (brief + decisions), obligations register, ask the contracts
(RAG chat), signed portfolio check, audit log. The assistant never signs, approves, rejects or
sends anything; Karandeep decides.

    streamlit run app.py
"""

from __future__ import annotations

import uuid
from datetime import date
from urllib.parse import quote

import pandas as pd
import streamlit as st

import config
import decisions
import history
import outcome
import register
import rules
from chain import answer
from dataset import get_doc, incoming_queue, load_docs, load_tracker
from review_graph import run_review
from vectorstore import count
from workflow_log import read_log

st.set_page_config(page_title="Atliq ContractLens", page_icon="📑", layout="wide")

SEV_ICON = {"blocker": "🛑", "high": "🔴", "medium": "🟠", "low": "🟡", "info": "⚪"}
PEOPLE = {"Karandeep (CEO, decides and signs)": "Karandeep", "Bhavin (founder, sales)": "Bhavin",
          "Dhaval (founder, sales)": "Dhaval", "Jay (sales)": "Jay", "Pranav (CTO)": "Pranav"}


@st.cache_resource(show_spinner="Loading the local embedding model…")
def warm_up() -> dict[str, int]:
    """Pay the one-off model load at boot, not on the first question.

    On a fresh deployment (Streamlit Community Cloud) the vector store is not in the repo, so it
    is built here the first time the app starts.
    """
    counts = {name: count(name) for name in ("signed", "incoming", "notes")}
    if not all(counts.values()):
        import ingest_contracts
        ingest_contracts.main()
        counts = {name: count(name) for name in ("signed", "incoming", "notes")}
    return counts


if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]
    st.session_state.messages = []

counts = warm_up()

# --- sidebar ---------------------------------------------------------------
with st.sidebar:
    st.title("📑 Atliq ContractLens")
    st.caption("Contract review assistant for AtliQ. It flags and cites; Karandeep decides and signs.")
    who_label = st.selectbox("I am", list(PEOPLE))
    me = PEOPLE[who_label]
    is_ceo = me == "Karandeep"
    page = st.radio("Go to", ["Review queue", "Review a contract", "Obligations register",
                              "Ask the contracts", "Signed portfolio check", "Audit log"])
    st.divider()
    st.caption(f"Today (dataset): {config.TODAY:%d %b %Y}")
    st.caption(f"Model: `{config.GROQ_MODEL}`" if config.GROQ_API_KEY else
               "⚠️ No GROQ_API_KEY: rule and register checks still run, AI steps are skipped.")
    st.caption("Index: " + ", ".join(f"{k} {v}" for k, v in counts.items()) +
               ("" if any(counts.values()) else "  ·  run `python ingest_all.py`"))


def my_rows(rows):
    """Sellers see their own deals only (PRD: never other clients' terms)."""
    return rows if is_ceo else [r for r in rows if r["doc"].tracker.get("requested_by") == me]


def sev_badge(sev: str) -> str:
    return f"{SEV_ICON.get(sev, '')} {sev.upper()}"


# --- pages -----------------------------------------------------------------
def page_queue() -> None:
    st.header("Review queue")
    st.caption("Incoming drafts from the tracker, most urgent first. Deadlines come from the tracker's notes column.")
    rows = []
    for r in my_rows(incoming_queue()):
        d = r["doc"]
        saved = decisions.load(d.tracker_id or d.doc_id)
        b = saved["brief"] if saved else None
        rows.append({
            "ID": d.tracker_id or "new file", "Counterparty": d.tracker.get("counterparty"), "Type": d.tracker.get("doc_type"),
            "Requested by": d.tracker.get("requested_by"),
            "Deadline": r["deadline"].strftime("%d %b") if r["deadline"] else "none recorded",
            "Days left": r["days_left"],
            "Value (USD)": d.tracker.get("value_usd") or "",
            "Review": ("✅ complete" if saved and saved.get("completed") else b["verdict"]) if b else "not reviewed",
            "Blockers": b["counts"]["blocker"] if b else "", "High": b["counts"]["high"] if b else "",
        })
    if not rows:
        st.info("No drafts requested by you in the queue.")
        return
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    total = sum(float(x["Value (USD)"] or 0) for x in rows)
    st.caption(f"{len(rows)} drafts, about ${total:,.0f} of deal value waiting on review.")


def render_finding(f: dict, record: dict | None, key: str) -> None:
    hide_other_clients = not is_ceo and f["origin"] == "register"
    title = f"{sev_badge(f['severity'])} · {f['title']}"
    with st.expander(f"{f['id']} · {title}", expanded=f["severity"] in ("blocker", "high")):
        if hide_other_clients:
            st.write("This deal conflicts with a commitment AtliQ made to another client. "
                     "Karandeep can see the details.")
            return
        src = f.get("source_id", "draft")
        where = "this draft" if src == "draft" else f"signed: {src}"
        clause = f" §{f['clause']}" if f.get("clause") else ""
        if f.get("quote"):
            st.markdown(f"> {f['quote']}")
            st.caption(f"Source: {where}{clause}  ·  quote checked word for word")
        if f.get("draft_quote"):
            st.markdown(f"**In this draft:** > {f['draft_quote']}")
        if f.get("evidence"):
            for ev in f["evidence"]:
                st.markdown(f"**Also:** > {ev['quote']}")
                st.caption(f"Source: {ev['source']}")
        if f.get("why"):
            st.write(f["why"])
        if f.get("assessment") and f["origin"] == "register":
            st.caption(f"AI assessment: {f['assessment']} (the flag stays until a person decides)")
        if f.get("ask_for"):
            st.markdown(f"**Ask for / do:** {f['ask_for']}")
        if f.get("standard"):
            st.caption(f"AtliQ standard: {f['standard']}")
        for p in f.get("precedent", []):
            st.caption(f"Past negotiation ({p['cite']}): {p['text']}")
        st.caption(f"Found by: {'rules' if f['origin'] == 'rule' else 'register check' if f['origin'] == 'register' else 'AI clause scan, citation-checked'}")
        if record is None:
            return
        dec = record["decisions"].get(f["id"])
        if dec:
            st.success(f"{decisions.ACTIONS[dec['action']]} by {dec['by']}"
                       + (f": {dec['reason']}" if dec["reason"] else ""))
        if is_ceo:
            c1, c2, c3 = st.columns([1, 1, 2])
            reason = c3.text_input("Reason (needed to dismiss)", key=f"r-{key}-{f['id']}")
            if c1.button("Accept", key=f"a-{key}-{f['id']}"):
                decisions.decide(key, f["id"], "accept", reason, me, st.session_state.session_id)
                st.rerun()
            if c2.button("Send to counsel", key=f"c-{key}-{f['id']}"):
                decisions.decide(key, f["id"], "counsel", reason, me, st.session_state.session_id)
                st.rerun()
            if c3.button("Dismiss", key=f"d-{key}-{f['id']}"):
                try:
                    decisions.decide(key, f["id"], "dismiss", reason, me, st.session_state.session_id)
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))


def render_brief(record: dict, key: str) -> None:
    b = record["brief"]
    c = b["counts"]
    color = st.error if c["blocker"] else st.warning if c["high"] else st.info
    color(f"**{b['verdict']}**  ·  {c['blocker']} blocker · {c['high']} high · {c['medium']} medium · "
          f"{c['low']} low · {c['info']} info")
    if not b["ai_completed"]:
        st.warning("**AI review not completed.** Rule and register checks below are complete; the AI clause "
                   "scan did not run, so no clause is shown as checked by it. " + " ".join(b["errors"]))
    if b["dropped_count"]:
        st.caption(f"Citation check removed {b['dropped_count']} AI finding(s) whose quote was not in the contract.")
    if b["recommend_counsel"] and is_ceo:
        st.info("Recommend outside counsel before signing (blocker on a restriction, patient data or liability).")
    h = b.get("history") or {}
    if h.get("times_seen"):
        (st.error if h["already_signed"] else st.warning)("**" + history.summary(h) + "**")
        st.dataframe(pd.DataFrame([{
            "Tracker ID": m["contract_id"] or "no tracker row", "Counterparty": m["counterparty"],
            "Type": m["doc_type"], "Status": m["status"], "Date": m["date"], "Where": m["where"],
            "Why it matches": m["how"]} for m in h["matches"]]), hide_index=True, use_container_width=True)
    if b["bundle"].get("items"):
        st.subheader(f"Deal documents: {b['bundle']['type']}")
        st.dataframe(pd.DataFrame(b["bundle"]["items"]).rename(
            columns={"doc": "Document", "status": "Status", "tracker_id": "Tracker / note"}),
            hide_index=True, use_container_width=True)
    st.subheader("Flags")
    for f in b["findings"]:
        render_finding(f, record if record.get("key") else None, key)
    with st.expander("What this review could not check"):
        st.write("Signed contracts listed in the tracker but with no file in the dataset, so they are "
                 "NOT in the obligations register:")
        st.write("\n".join(f"- {x}" for x in b["not_checked"]))
        st.write("Also not checked: whether the scope matches the proposal (no proposals in the dataset).")
    if is_ceo and record.get("key"):
        remaining = decisions.open_items(record)
        if record.get("completed"):
            st.success(f"Review marked complete by {record['completed']['by']}. Signing stays with Karandeep, outside this tool.")
        elif st.button("Mark review complete", disabled=bool(remaining), type="primary",
                       help="Needs a decision on every blocker and high flag" if remaining else None):
            decisions.complete(key, me, st.session_state.session_id)
            st.rerun()
        if remaining:
            st.caption("Decide these before completing: " + ", ".join(remaining))
        st.subheader("Outcome brief (email or WhatsApp)")
        st.caption("One email built from your decisions: accepted flags become numbered asks in AtliQ's playbook "
                   "wording, dismissed flags are left out, other clients are never named. Nothing is sent from here.")
        if st.button("Prepare outcome brief", disabled=bool(remaining), key=f"o-{key}"):
            st.session_state[f"outcome-{key}"] = outcome.build(record, me, st.session_state.session_id)
        o = st.session_state.get(f"outcome-{key}")
        if o:
            st.caption(f"{o['points']} point(s) in the client email · {o['counsel_count']} for counsel · "
                       f"{o['internal_count']} kept internal. Use the copy icon on each box.")
            st.text_input("Subject", o["subject"], key=f"os-{key}")
            st.code(o["client_email"], language=None)
            st.download_button("Download as Outlook draft (.eml)", file_name="outcome.eml", mime="message/rfc822",
                               data=f"X-Unsent: 1\r\nSubject: {o['subject']}\r\nContent-Type: text/plain; "
                                    f"charset=utf-8\r\n\r\n" + o["client_email"].replace("\n", "\r\n"))
            st.caption("WhatsApp version (bold headings):")
            st.code(o["whatsapp"], language=None)
            st.link_button("Share on WhatsApp", "https://wa.me/?text=" + quote(o["whatsapp"]))
            with st.expander("Internal note for counsel (not for the client)"):
                st.code(o["counsel_note"], language=None)


def page_review() -> None:
    st.header("Review a contract")
    queue = my_rows(incoming_queue())
    options = {f"{r['doc'].tracker_id or 'new file'} · {r['doc'].title}": r["doc"] for r in queue}
    mode = st.radio("Contract", ["From the queue", "Upload a file (.md / .txt)"], horizontal=True)
    if mode == "From the queue":
        if not options:
            st.info("No drafts requested by you.")
            return
        label = st.selectbox("Draft", list(options))
        d = options[label]
        text, tracker, key, doc_id = d.text, d.tracker, d.tracker_id or d.doc_id, d.doc_id
        st.caption(f"Requested by {tracker.get('requested_by')} · tracker note: {tracker.get('notes') or '—'}")
    else:
        up = st.file_uploader("Contract text", type=["md", "txt"])
        c1, c2, c3 = st.columns(3)
        tracker = {"counterparty": c1.text_input("Counterparty"),
                   "client_country": c2.text_input("Counterparty country", "India"),
                   "doc_type": c3.selectbox("Type", ["MSA", "SOW", "Mutual NDA", "NDA", "BAA",
                                                    "Partnership Agreement", "Contractor Agreement",
                                                    "Subcontractor Agreement"]),
                   "requested_by": me, "contract_id": "", "notes": ""}
        if not up:
            st.caption("Tip: try data/demo/harrington_sample_extract.txt to see the personal-data gate.")
            return
        text = up.read().decode("utf-8", errors="replace")
        key = "upload-" + "".join(ch for ch in up.name if ch.isalnum())[:40]
        tracker["contract_id"] = key
        doc_id = up.name

    saved = decisions.load(key)
    if st.button("Run review", type="primary"):
        with st.status("Reviewing…", expanded=True) as status:
            st.write("Personal-data gate → extract → entity/law, documents, register → retrieve → "
                     "clause scan ∥ conflict check → citation check → brief")
            state = run_review(text, tracker, doc_id=doc_id, session_id=st.session_state.session_id)
            for s in state.get("steps", []):
                st.write(s)
            if state["status"] == "blocked_personal_data":
                status.update(label="Stopped: personal data found", state="error")
                st.error("This file contains personal data (" + ", ".join(state["pii"]) + "). It was NOT sent "
                         "to any model. Remove it and use the secure channel agreed with the client.")
                return
            saved = decisions.save_review(key, state["brief"], me)
            status.update(label="Review ready" if state["brief"]["ai_completed"] else "Review not completed",
                          state="complete" if state["brief"]["ai_completed"] else "error")
    if saved:
        st.caption(f"Last run by {saved['run_by']} at {saved['run_at']} UTC")
        render_brief(saved, key)


def page_register() -> None:
    st.header("Obligations register")
    st.caption("What AtliQ has already promised. Every entry quotes its signed contract; quotes are checked "
               "word for word at start-up. Nobody at AtliQ had this list before.")
    if not is_ceo:
        st.info("The register holds other clients' terms, so only Karandeep can open it. Your deal reviews "
                "still check against it.")
        return
    entries = register.load_register()
    df = pd.DataFrame([{"ID": e["id"], "Type": e["type"], "What": e["title"], "Binds": e["bound"],
                        "Until": e.get("active_until") or "no end date", "Status": e["status"],
                        "Quotes verified": "✅" if e["verified"] else "❌"} for e in entries])
    st.dataframe(df, hide_index=True, use_container_width=True)
    for e in entries:
        with st.expander(f"{e['id']} · {e['title']}"):
            for q in e["quotes"]:
                st.markdown(f"> {q}")
            st.caption(f"Source: {e['source']} §{e['clause']} · scope: {e['scope']}")
            if e.get("until_note"):
                st.caption(e["until_note"])
            if e.get("action"):
                st.markdown(f"**If a draft hits this:** {e['action']}")

    st.subheader("Add entries from a signed contract")
    st.caption("The model proposes; only quotes found word for word are shown; nothing counts until you confirm.")
    signed = {d.title + " · " + d.doc_id: d.doc_id for d in load_docs() if d.kind == "signed"}
    pick = st.selectbox("Signed contract", list(signed))
    if st.button("Propose entries"):
        if not config.GROQ_API_KEY:
            st.error("Needs GROQ_API_KEY.")
        else:
            try:
                kept, dropped = register.propose_from_signed(signed[pick])
                st.session_state.proposals = {"doc": signed[pick], "items": kept, "dropped": len(dropped)}
            except Exception as e:
                st.error(f"Model step failed: {e}")
    prop = st.session_state.get("proposals")
    if prop and prop["doc"] == signed[pick]:
        st.caption(f"{len(prop['items'])} proposals with verified quotes, {prop['dropped']} dropped.")
        for i, it in enumerate(prop["items"]):
            with st.container(border=True):
                st.markdown(f"**{it.get('type')}: {it.get('title')}** (§{it.get('clause')})")
                st.markdown(f"> {it.get('quote')}")
                kw = st.text_input("Flag drafts that mention (comma-separated keywords)", key=f"kw{i}")
                if st.button("Confirm into register", key=f"ok{i}"):
                    register.save_confirmed({
                        "id": f"R-C{i + 1}-{prop['doc'][:10]}", "type": it.get("type", ""),
                        "title": it.get("title", ""), "source": prop["doc"], "clause": it.get("clause", ""),
                        "bound": "AtliQ", "scope": it.get("scope", ""), "active_from": "",
                        "active_until": it.get("active_until") or None, "quotes": [it.get("quote", "")],
                        "severity": "high",
                        "triggers": {"any": [k.strip() for k in kw.split(",") if k.strip()]} if kw.strip() else None,
                        "confirmed_by": me})
                    st.success("Added. It now counts in every review.")


def page_ask() -> None:
    st.header("Ask the contracts")
    st.caption("Answers come only from the signed contracts, drafts and notes, with numbered sources. "
               "It will not sign, approve, send, or give legal opinions.")
    if not any(counts.values()):
        st.warning("The index is empty. Run `python ingest_all.py` first.")
        return
    samples = ["What have we promised Al Noor, and until when?",
               "Which signed contracts restrict who we can work with?",
               "What did we agree on liquidated damages with Acme and Trustline?",
               "What does the intercompany BAA allow the Pune team to do?"] if is_ceo else \
              ["What is still missing for my deals?", "What does my client's draft say about payment terms?"]
    cols = st.columns(len(samples))
    for c, s in zip(cols, samples):
        if c.button(s, use_container_width=True):
            st.session_state.pending = s
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    q = st.chat_input("Ask about any contract or past negotiation") or st.session_state.pop("pending", None)
    if not q:
        return
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        refusal = rules.guardrail(q)
        if refusal:
            text = refusal
            st.markdown(text)
        elif not config.GROQ_API_KEY:
            text = "Needs GROQ_API_KEY in .env to answer questions."
            st.markdown(text)
        else:
            if is_ceo:
                kw = {}
            else:   # sellers: only their own deals' drafts
                mine = [r["doc"].tracker_id for r in my_rows(incoming_queue())]
                kw = {"collections": ("incoming",), "where": {"tracker_id": {"$in": mine or ["none"]}}}
            try:
                text, sources, bad = answer(q, st.session_state.messages[:-1], **kw)
            except Exception as e:
                text, sources, bad = f"Couldn't answer: {e}", [], []
            st.markdown(text)
            if bad:
                st.warning(f"The answer cites source(s) {bad} that were not retrieved; treat those parts as unsupported.")
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"**[{s['n']}] {s['cite']}**")
                        st.caption(s["excerpt"][:600])
    st.session_state.messages.append({"role": "assistant", "content": text})


def page_portfolio() -> None:
    st.header("Signed portfolio check")
    st.caption("The same entity and governing-law rules, run over everything already signed.")
    if not is_ceo:
        st.info("Only Karandeep can see signed contract terms.")
        return
    rows = rules.portfolio_check()
    st.dataframe(pd.DataFrame([{"Contract": r["contract"], "Severity": sev_badge(r["severity"]),
                                "Flag": r["title"], "Clause": r["quote"][:140],
                                "Why": r["why"][:160]} for r in rows]),
                 hide_index=True, use_container_width=True)
    st.subheader("Signed, but no file in the dataset")
    in_files = {d.tracker_id for d in load_docs() if d.kind == "signed"}
    st.write("\n".join(f"- {r['contract_id']} {r['counterparty']} ({r['doc_type']}), stored: "
                       f"{r.get('file_location') or 'unknown'}"
                       for r in load_tracker().values() if r.get("status") == "Signed" and r["contract_id"] not in in_files))


def page_audit() -> None:
    st.header("Audit log")
    st.caption("Every step and decision, with IDs and counts only. No contract text is logged.")
    rows = read_log()
    if not rows:
        st.info("Nothing logged yet.")
        return
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


{"Review queue": page_queue, "Review a contract": page_review, "Obligations register": page_register,
 "Ask the contracts": page_ask, "Signed portfolio check": page_portfolio, "Audit log": page_audit}[page]()
