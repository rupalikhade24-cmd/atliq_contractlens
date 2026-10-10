"""Scenario and guardrail checks for Atliq ContractLens (PRD section 11, eval criteria).

Run with the venv python:  python test_contractlens.py

Everything here runs offline: the deterministic layer is tested against the real dataset, and the
LangGraph workflow runs end to end with a fake model (so the citation check can be tested with a
quote the model "made up"). Pass --live to also run the AI clause scan against Groq on the
scenarios that need it (uses your GROQ_API_KEY and free-tier quota).
"""

from __future__ import annotations

import json
import sys

import config
import register
import rules
from citations import find_quote
from dataset import get_doc, load_docs
from review_graph import run_review

PASS, FAIL = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASS if condition else FAIL).append(name)
    mark = "PASS" if condition else "FAIL"
    print(f"[{mark}] {name}" + (f"  <- {detail}" if detail and not condition else ""))


def doc(file_stem: str):
    return get_doc(file_stem)


def review_offline(stem: str, llm=None) -> dict:
    d = doc(stem)
    return run_review(d.text, d.tracker, doc_id=d.doc_id, session_id="test", llm=llm)["brief"]


def ids(brief: dict) -> dict:
    return {f["rule_id"]: f["severity"] for f in brief["findings"]}


# --- Register ---------------------------------------------------------------
def register_checks() -> None:
    print("\n--- Obligations register ---")
    entries = register.load_register()
    check("every register quote is found word for word in its signed contract",
          all(e["verified"] for e in entries),
          str([e["id"] for e in entries if not e["verified"]]))
    status = {e["id"]: e["status"] for e in entries}
    check("Orbit Travels non-compete is marked expired", status.get("R-06") == "expired")
    check("Al Noor non-compete is active on 28 Sep 2026", status.get("R-01") == "active")


# --- The seven capstone scenarios -------------------------------------------
def scenario_collision() -> None:
    print("\n--- Scenario 1: collides with something already signed ---")
    expected = {
        "2026-09-18_gulf_crown_hotels_msa_draft": ("REGISTER-R-01", "blocker"),
        "2026-09-21_blueorchid_hotels_pilot_agreement_draft": ("REGISTER-R-01", "blocker"),
        "2026-09-15_travelhub_services_agreement_draft": ("REGISTER-R-01", "high"),
        "2026-09-19_lakeshore_grocers_msa_draft": ("REGISTER-R-03", "blocker"),
        "2026-09-23_rheinwerk_analytics_services_agreement_draft": ("REGISTER-R-04", "blocker"),
        "2026-09-24_datavane_strategic_partnership_agreement_draft": ("REGISTER-R-05", "blocker"),
    }
    for stem, (rule, sev) in expected.items():
        got = ids(review_offline(stem))
        check(f"{doc(stem).title}: {rule} flagged as {sev}", got.get(rule) == sev, str(got))
    check("100% recall on known register conflicts (PRD eval gate)", not [n for n in FAIL if "REGISTER" in n])


def scenario_wrong_entity_law() -> None:
    print("\n--- Scenario 3: contract on the wrong entity / law ---")
    got = ids(review_offline("2026-09-15_travelhub_services_agreement_draft"))
    check("TravelHub: AtliQ Inc named for a UAE client", got.get("ENTITY-MISMATCH") == "blocker", str(got))
    check("TravelHub: invoicing entity differs", "INVOICE-ENTITY-MISMATCH" in got)
    got = ids(review_offline("2026-09-27_marcus_reed_contractor_agreement_draft"))
    check("Marcus Reed: India template for a US contractor", got.get("ENTITY-MISMATCH") == "blocker", str(got))
    signed = rules.portfolio_check()
    check("Portfolio check finds Brightwater (AtliQ Inc under English law)",
          any("brightwater" in f["doc_id"] and f["rule_id"] == "LAW-FOREIGN" for f in signed))


def scenario_nda() -> None:
    print("\n--- Scenario 4: NDA that says mutual but isn't ---")
    got = ids(review_offline("2026-09-25_finserve_capital_mutual_nda_draft"))
    check("FinServe 'mutual' NDA flagged NDA-NOT-MUTUAL", "NDA-NOT-MUTUAL" in got, str(got))
    got = ids(review_offline("2026-09-27_loopmart_mutual_nda_draft"))
    check("LoopMart genuinely mutual NDA is NOT flagged", "NDA-NOT-MUTUAL" not in got, str(got))


def scenario_bundle() -> None:
    print("\n--- Scenario 6: healthcare deal with a document missing ---")
    b = review_offline("2026-09-10_harrington_health_msa_draft")
    items = {i["doc"]: i["status"] for i in b["bundle"].get("items", [])}
    check("Harrington recognised as a US healthcare deal", "healthcare" in b["bundle"].get("type", ""))
    check("Subcontractor BAA for Daniel Ortiz is missing", items.get("Subcontractor BAA: Daniel Ortiz") == "Missing", str(items))
    check("Pune intercompany BAA for Harrington is missing (C-044 is CareBridge only)",
          any("Pune" in k and v == "Missing" for k, v in items.items()), str(items))
    check("Patient data received before a BAA is a blocker", ids(b).get("REGISTER-R-07") == "blocker")
    b = review_offline("2026-09-26_kriti_data_labs_subcontractor_agreement_draft")
    check("Kriti linked to the Rheinwerk GDPR deal (sub-processor DPA missing)",
          "GDPR" in b["bundle"].get("type", ""), str(b["bundle"].get("type")))


def scenario_only_looks_risky() -> None:
    print("\n--- Scenario 7: the one that only looks risky ---")
    for stem in ("2026-09-24_sunrise_foods_sow2_draft", "2026-09-27_loopmart_mutual_nda_draft"):
        b = review_offline(stem)
        check(f"{doc(stem).title}: no rule or register blocker/high", b["counts"]["blocker"] + b["counts"]["high"] == 0,
              str(ids(b)))


# --- Guardrails ---------------------------------------------------------------
class FakeLLM:
    """Returns canned JSON per prompt. One clause-scan quote is invented on purpose."""

    def __init__(self, real_quote: str):
        self.real_quote = real_quote

    def invoke(self, messages):
        system = messages[0].content

        class R:
            pass

        r = R()
        if system.startswith("You extract"):
            r.content = json.dumps({"counterparty": "Test", "industries": ["hospitality"]})
        elif system.startswith("You review"):
            r.content = json.dumps([
                {"rule_id": "LD-ABOVE-STANDARD", "clause": "12.2", "quote": self.real_quote, "why": "x"},
                {"rule_id": "LIAB-UNLIMITED", "clause": "99", "quote": "Supplier shall have unlimited liability for everything forever", "why": "invented"},
                {"rule_id": "MADE-UP-RULE", "clause": "1", "quote": self.real_quote, "why": "unknown rule id"},
            ])
        else:
            r.content = "```json\n" + json.dumps([{"register_id": "R-01", "assessment": "conflict",
                                                   "draft_quote": "not in the draft at all, invented", "why": "x"}]) + "\n```"
        return r


def guardrails() -> None:
    print("\n--- Guardrails ---")
    gulf = doc("2026-09-18_gulf_crown_hotels_msa_draft")
    real = "one percent (1%) of the Contract Value for each week or part thereof of delay"
    check("(setup) real quote exists in Gulf Crown draft", find_quote(real, gulf.text))
    state = run_review(gulf.text, gulf.tracker, doc_id=gulf.doc_id, session_id="test", llm=FakeLLM(real))
    b = state["brief"]
    ai = [f for f in b["findings"] if f["origin"] == "ai"]
    check("citation check drops the invented quote", b["dropped_count"] == 1, str(b["dropped_count"]))
    check("verified model finding is kept", any(f["quote"] == real for f in ai))
    check("severity comes from the playbook, not the model",
          next(f for f in ai if f["rule_id"] == "LD-ABOVE-STANDARD")["severity"] == rules.severity_for("LD-ABOVE-STANDARD"))
    check("unknown rule id is mapped to OTHER (low)", any(f["rule_id"] == "OTHER" and f["severity"] == "low" for f in ai))
    reg = next(f for f in b["findings"] if f["rule_id"] == "REGISTER-R-01")
    check("model cannot clear a register flag (recall-first)", reg["severity"] == "blocker")
    check("invented conflict quote is removed from the explanation", reg.get("draft_quote", "") == "")
    check("workflow reaches brief_ready with a model", state["status"] == "brief_ready", state["status"])

    no_key = run_review(gulf.text, gulf.tracker, doc_id=gulf.doc_id, session_id="test") if not config.GROQ_API_KEY else None
    if no_key:
        check("without a model, status is review_not_completed", no_key["status"] == "review_not_completed")
        check("without a model, rule and register flags still show",
              any(f["origin"] == "register" for f in no_key["brief"]["findings"]))

    blocked_text = (config.DATA_DIR / "demo" / "harrington_sample_extract.txt").read_text(encoding="utf-8")
    s = run_review(blocked_text, {}, doc_id="demo-extract", session_id="test", llm=FakeLLM(real))
    check("personal-data gate stops before any model call", s["status"] == "blocked_personal_data", s.get("status"))
    check("no dataset contract trips the personal-data gate",
          not [d.doc_id for d in load_docs() if rules.pii_gate(d.text)])

    for q in ("Sign the Gulf Crown contract", "please approve this NDA", "send it to Faisal",
              "Is this legal advice?", "is this enforceable in Saudi Arabia?"):
        check(f"refuses: {q!r}", bool(rules.guardrail(q)))
    for q in ("What did we promise Al Noor?", "Which contracts did Karandeep sign in 2025?",
              "What does the Gulf Crown sign-off clause say?"):
        check(f"answers: {q!r}", not rules.guardrail(q))


def live() -> None:
    print("\n--- Live AI checks (Groq) ---")
    if not config.GROQ_API_KEY:
        print("[skip] GROQ_API_KEY not set")
        return
    for stem, rule in (("2026-09-10_harrington_health_msa_draft", "LD-UNCAPPED"),
                       ("2026-09-26_kriti_data_labs_subcontractor_agreement_draft", "UNFAIR-TO-COUNTERPARTY")):
        b = review_offline(stem)
        got = ids(b)
        check(f"{doc(stem).title}: AI finds {rule}", rule in got, str(got))
        check(f"{doc(stem).title}: every shown AI quote is verbatim",
              all(find_quote(f["quote"], doc(stem).text) for f in b["findings"] if f["origin"] == "ai"))


def main() -> None:
    print(f"Dataset: {config.DATASET_DIR}")
    register_checks()
    scenario_collision()
    scenario_wrong_entity_law()
    scenario_nda()
    scenario_bundle()
    scenario_only_looks_risky()
    guardrails()
    if "--live" in sys.argv:
        live()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
