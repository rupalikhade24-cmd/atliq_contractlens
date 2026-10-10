"""Workflow tracing and audit log (PRD section 8, Observability).

Records each step of a review - upload, extract, rules, retrieve, LLM, citation check, brief,
decisions - with IDs and counts only. Contract text and model reasoning are never logged.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import config

WORKFLOW_LOG = config.LOG_DIR / "workflow.jsonl"


class Trace:
    """Collects the steps of one review, for the log and for the UI panel."""

    def __init__(self, session_id: str, contract_id: str = "") -> None:
        self.session_id = session_id
        self.contract_id = contract_id
        self.steps: list[str] = []

    def add(self, stage: str, detail: str = "") -> str:
        line = f"[{stage}] {detail}".rstrip()
        self.steps.append(line)
        self._persist(stage, detail)
        return line

    def _persist(self, stage: str, detail: str) -> None:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "contract_id": self.contract_id,
            "stage": stage,
            "detail": detail,
        }
        with WORKFLOW_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_log(limit: int = 300) -> list[dict]:
    if not WORKFLOW_LOG.exists():
        return []
    lines = WORKFLOW_LOG.read_text(encoding="utf-8").splitlines()[-limit:]
    return [json.loads(l) for l in reversed(lines) if l.strip()]
