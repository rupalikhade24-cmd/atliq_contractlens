"""Command-line ContractLens: review a draft or ask a question without the UI.

    python main.py queue
    python main.py review C-025
    python main.py ask "What have we promised Al Noor?"
"""

from __future__ import annotations

import sys
import uuid

import rules
from chain import answer
from dataset import incoming_queue, load_docs
from review_graph import run_review


def cmd_queue() -> None:
    for r in incoming_queue():
        d = r["doc"]
        due = r["deadline"].strftime("%d %b") if r["deadline"] else "no date"
        print(f"{d.tracker_id}  {due:>7}  {d.tracker.get('requested_by', ''):<10} {d.title}")


def cmd_review(contract_id: str) -> None:
    doc = next((d for d in load_docs() if d.tracker_id == contract_id and d.kind == "incoming"), None)
    if not doc:
        sys.exit(f"No incoming draft with id {contract_id}. Try: python main.py queue")
    state = run_review(doc.text, doc.tracker, doc_id=doc.doc_id, session_id=uuid.uuid4().hex[:12])
    print("\n".join(state.get("steps", [])))
    if state["status"] == "blocked_personal_data":
        sys.exit("Stopped: personal data found. Nothing was sent to a model.")
    b = state["brief"]
    print(f"\n{doc.title}: {b['verdict']}  {b['counts']}")
    if not b["ai_completed"]:
        print("AI review NOT completed: " + " ".join(b["errors"]))
    for item in b["bundle"].get("items", []):
        print(f"  [deal] {item['doc']}: {item['status']}")
    for f in b["findings"]:
        src = "draft" if f.get("source_id", "draft") == "draft" else f["source_id"]
        print(f"\n{f['id']} {f['severity'].upper():8} {f['title']}")
        if f.get("quote"):
            clause = f" §{f['clause']}" if f.get("clause") else ""
            print(f"   \"{f['quote'][:220]}\"  ({src}{clause})")
        if f.get("why"):
            print(f"   {f['why'][:300]}")


def cmd_ask(question: str) -> None:
    refusal = rules.guardrail(question)
    if refusal:
        print(refusal)
        return
    text, sources, bad = answer(question)
    print(text)
    print("\nSources:")
    for s in sources:
        print(f"  [{s['n']}] {s['cite']}")
    if bad:
        print(f"Unsupported citations: {bad}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("queue", "review", "ask"):
        sys.exit(__doc__)
    {"queue": lambda: cmd_queue(), "review": lambda: cmd_review(sys.argv[2]),
     "ask": lambda: cmd_ask(" ".join(sys.argv[2:]))}[sys.argv[1]]()
