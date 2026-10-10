"""Saved reviews and human decisions (PRD: approval gates, audit trail).

Each review is stored as reviews/<contract id>.json with the brief, every decision and who made
it. A review can be marked complete only when every blocker and high flag has a decision.
Nothing here signs, approves or sends anything.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import config
from workflow_log import Trace

ACTIONS = {"accept": "Accepted (agree with the flag)",
           "dismiss": "Dismissed (with reason)",
           "counsel": "Send to outside counsel"}


def _path(key: str):
    return config.REVIEWS_DIR / f"{key}.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load(key: str) -> dict | None:
    p = _path(key)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def save_review(key: str, brief: dict, by: str) -> dict:
    """A new run replaces the brief; decisions on flags that still exist are kept."""
    config.REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    old = load(key) or {}
    keep = {}
    old_by_sig = {(f["rule_id"], f.get("quote", "")[:80]): f["id"] for f in old.get("brief", {}).get("findings", [])}
    for f in brief["findings"]:
        old_id = old_by_sig.get((f["rule_id"], f.get("quote", "")[:80]))
        if old_id and old_id in old.get("decisions", {}):
            keep[f["id"]] = old["decisions"][old_id]
    record = {"key": key, "brief": brief, "decisions": keep, "completed": None,
              "run_by": by, "run_at": _now()}
    _path(key).write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def decide_record(record: dict, finding_id: str, action: str, reason: str, by: str, session_id: str) -> dict:
    """Apply one decision to a review record (no file access; the web version keeps records in the browser)."""
    if action not in ACTIONS:
        raise ValueError(f"unknown action {action}")
    if action == "dismiss" and not reason.strip():
        raise ValueError("A dismissal needs a reason.")
    if finding_id not in {f["id"] for f in record["brief"]["findings"]}:
        raise ValueError(f"unknown flag {finding_id}")
    record["decisions"][finding_id] = {"action": action, "reason": reason.strip(), "by": by, "at": _now()}
    record["completed"] = None   # any change re-opens the review
    # audit: ids and the action only, never contract text
    Trace(session_id, record.get("key", "")).add("Decision", f"{finding_id} {action} by {by}")
    return record


def complete_record(record: dict, by: str, session_id: str) -> dict:
    remaining = open_items(record)
    if remaining:
        raise ValueError("Decide these first: " + ", ".join(remaining))
    record["completed"] = {"by": by, "at": _now()}
    Trace(session_id, record.get("key", "")).add("Review complete", f"by {by}; signing stays with Karandeep")
    return record


def new_record(key: str, brief: dict, by: str) -> dict:
    """A fresh review record without touching disk (web version)."""
    return {"key": key, "brief": brief, "decisions": {}, "completed": None, "run_by": by, "run_at": _now()}


def decide(key: str, finding_id: str, action: str, reason: str, by: str, session_id: str) -> dict:
    record = decide_record(load(key), finding_id, action, reason, by, session_id)
    _path(key).write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def open_items(record: dict) -> list[str]:
    """Blocker and high flags without a decision."""
    return [f["id"] for f in record["brief"]["findings"]
            if f["severity"] in ("blocker", "high") and f["id"] not in record["decisions"]]


def complete(key: str, by: str, session_id: str) -> dict:
    record = complete_record(load(key), by, session_id)
    _path(key).write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record
