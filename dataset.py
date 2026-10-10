"""Loads the capstone dataset: contracts, tracker, notes. Read-only; nothing here calls a model."""

from __future__ import annotations
import csv
import re
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path

import config

# Each contract file mapped to its tracker row. Written by hand because the tracker
# has no file paths for most rows (that gap is part of the problem the dataset describes).
FILE_TO_TRACKER = {
    # incoming drafts
    "2026-09-10_harrington_health_msa_draft.md": "C-023",
    "2026-09-15_travelhub_services_agreement_draft.md": "C-024",
    "2026-09-16_harrington_health_baa_draft.md": "C-046",
    "2026-09-18_gulf_crown_hotels_msa_draft.md": "C-025",
    "2026-09-19_lakeshore_grocers_msa_draft.md": "C-026",
    "2026-09-21_blueorchid_hotels_pilot_agreement_draft.md": "C-027",
    "2026-09-23_rheinwerk_analytics_services_agreement_draft.md": "C-028",
    "2026-09-24_datavane_strategic_partnership_agreement_draft.md": "C-035",
    "2026-09-24_sunrise_foods_sow2_draft.md": "C-029",
    "2026-09-25_finserve_capital_mutual_nda_draft.md": "C-036",
    "2026-09-26_kriti_data_labs_subcontractor_agreement_draft.md": "C-037",
    "2026-09-27_daniel_ortiz_contractor_agreement_draft.md": "C-047",
    "2026-09-27_daniel_ortiz_nda_draft.md": "C-048",
    "2026-09-27_loopmart_mutual_nda_draft.md": "C-038",
    "2026-09-27_marcus_reed_contractor_agreement_draft.md": "C-040",
    # signed contracts
    "2022-07-01_orbit_travels_msa_scanned.md": "C-001",
    "2024-08-01_crestline_retail_msa.md": "C-002",
    "2025-01-22_al_noor_mutual_nda.md": "C-013",
    "2025-01-22_paytrack_technologies_msa.md": "C-005",
    "2025-03-10_acme_retail_group_msa.md": "C-006",
    "2025-03-15_al_noor_hospitality_msa.md": "C-007",
    "2025-04-18_sunrise_foods_msa.md": "C-008",
    "2025-04-28_rohan_iyer_contractor_agreement.md": "C-039",
    "2025-06-20_northwind_logistics_msa.md": "C-009",
    "2025-09-01_cloudspan_strategic_partnership_agreement.md": "C-032",
    "2025-10-06_atliq_intercompany_subcontractor_baa.md": "C-044",
    "2025-10-06_carebridge_clinics_baa.md": "C-043",
    "2025-11-28_brightwater_insurance_msa.md": "C-014",
    "2026-02-10_al_noor_waiver_letter.md": "C-016",
    "2026-07-16_meridian_healthcare_nda.md": "C-020",
    "2026-08-15_trustline_nbfc_msa.md": "C-021",
    "2026-08-20_harrington_health_mutual_nda.md": "C-045",
}

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


@dataclass
class Doc:
    doc_id: str            # file name without .md
    path: Path
    kind: str              # "incoming" | "signed" | "note"
    text: str
    tracker: dict = field(default_factory=dict)

    @property
    def title(self) -> str:
        t = self.tracker
        if t:
            return f"{t.get('counterparty', '')} – {t.get('doc_type', '')}".strip(" –")
        return self.doc_id.replace("_", " ")

    @property
    def tracker_id(self) -> str:
        return self.tracker.get("contract_id", "")


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


@lru_cache(maxsize=1)
def load_tracker() -> dict:
    """contract_id -> row. Keeps blanks as empty strings; the duplicate row (C-011) stays visible."""
    path = config.DATASET_DIR / "contract_tracker.csv"
    with path.open(encoding="utf-8", newline="") as f:
        return {row["contract_id"]: row for row in csv.DictReader(f)}


def deadline_from_notes(notes: str):
    """Deadlines only live in the tracker's free-text notes ("sign by 10 Oct!!", "by 3 Oct")."""
    m = re.search(r"\b(?:by|before|start)\s+(?:[A-Za-z]{3}\s+)?(\d{1,2})\s+([A-Za-z]{3})", notes or "", re.I)
    if not m or m.group(2).lower()[:3] not in MONTHS:
        return None
    return date(config.TODAY.year, MONTHS[m.group(2).lower()[:3]], int(m.group(1)))


@lru_cache(maxsize=1)
def load_docs() -> tuple:
    tracker = load_tracker()
    docs = []
    for kind, folder in (("incoming", "incoming"), ("signed", "signed_contracts")):
        for p in sorted((config.DATASET_DIR / folder).glob("*.md")):
            row = tracker.get(FILE_TO_TRACKER.get(p.name, ""), {})
            docs.append(Doc(p.stem, p, kind, _read(p), dict(row)))
    notes = [config.DATASET_DIR / n for n in (
        "negotiation_notes.md", "atliq_entities.md", "karandeep_contract_checklist.md")]
    notes += sorted((config.DATASET_DIR / "meeting_notes").glob("*.md"))
    for p in notes:
        if p.exists():
            docs.append(Doc(p.stem, p, "note", _read(p)))
    return tuple(docs)


def get_doc(doc_id: str) -> Doc:
    for d in load_docs():
        if d.doc_id == doc_id:
            return d
    raise KeyError(doc_id)


def incoming_queue() -> list:
    """Incoming drafts with deadline and days left, most urgent first."""
    rows = []
    for d in load_docs():
        if d.kind != "incoming":
            continue
        due = deadline_from_notes(d.tracker.get("notes", ""))
        rows.append({
            "doc": d,
            "deadline": due,
            "days_left": (due - config.TODAY).days if due else None,
        })
    return sorted(rows, key=lambda r: (r["deadline"] is None, r["deadline"] or date.max))
