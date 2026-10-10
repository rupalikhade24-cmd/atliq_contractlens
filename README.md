# Atliq ContractLens

A contract review assistant for AtliQ, a founder-led IT services company with no legal team.
Karandeep (CEO) reviews and signs every contract himself, late at night, from memory. ContractLens
reads a draft and tells him, with the exact clause quoted, what it is really asking AtliQ to sign:

- **what it collides with**: obligations AtliQ already signed (the Al Noor GCC hotel non-compete,
  the Crestline most-favoured pricing, PipeKit IP handed to Northwind, CloudSpan exclusivity)
- **what is buried in it**: uncapped liquidated damages, unlimited liability, indemnities for the
  client's own negligence, NDAs that say "mutual" but bind only AtliQ
- **whether it fits the entity signing it**: AtliQ Inc for US clients under US law, AtliQ
  Technologies Pvt Ltd for everyone else under Indian law
- **whether the deal is complete**: NDA + HIPAA BAA + MSA + a subcontractor BAA for everyone who
  touches patient data; a GDPR DPA for EU personal data
- **whether AtliQ's own paper is fair** to the freelancer or small agency receiving it

It never signs, approves, rejects or sends anything, and it gives no legal opinion. Every flag
quotes its source word for word; a person decides.

Built to the Atliq ContractLens PRD, on the same stack as the RAG chatbot (LangChain + Groq +
Chroma + local sentence-transformers + Streamlit), with LangGraph for the review workflow.

---

## Quick start

Put this folder next to the capstone `Dataset` folder:

```
C:\Atliq_ContractRisk\data1\
├── Dataset\                 <- the capstone dataset (unchanged)
└── atliq_contractlens\      <- this project
```

```powershell
cd C:\Atliq_ContractRisk\data1\atliq_contractlens

# 1. Project-local virtual environment
py -3 -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Dependencies (laptop: includes Streamlit and the local embeddings; Vercel uses the smaller requirements.txt)
pip install -r requirements-local.txt

# 3. API key (same free Groq key as the RAG chatbot works)
copy .env.example .env
notepad .env          # GROQ_API_KEY=gsk_...

# 4. Build the vector store (once; the embedding model is the chatbot's, so it is likely cached)
python ingest_all.py

# 5. Verify
python check_setup.py

# 6. Run
streamlit run app.py                 # browser UI (Streamlit screen)
uvicorn api.index:app --reload       # same app as on Vercel: open http://127.0.0.1:8000
python main.py queue                 # CLI: the review queue
python main.py review C-025          # CLI: review one draft
python main.py ask "What have we promised Al Noor?"

# 7. Scenario and guardrail checks (offline; add --live to include the AI clause scan)
python test_contractlens.py
```

Without a Groq key the app still runs every rule and register check; the AI steps show
"review not completed" instead of pretending to have checked.

---

## How it works

```
Draft contract (from the queue, or an uploaded .md / .txt)
     │
     ▼
Personal-data gate                   rules.pii_gate      stops BEFORE any model call
     │
     ▼
Extract fields                       Groq (fast tier)    counterparty, law, locations…
     │
     ▼
Deterministic checks                 rules.py, register.py
  ├── entity + governing law         (US client → AtliQ Inc / US law, else Pvt Ltd / India)
  ├── deal documents                 NDA, BAA, MSA, subcontractor BAAs, DPAs, missing annexes
  ├── NDA mutuality
  └── obligations register          10 entries from signed contracts, recall-first triggers
     │
     ▼
Retrieve                             draft passages per register hit; negotiation precedent
     │                               from Chroma (notes collection)
     ├──────────────────────┐
     ▼                      ▼
AI clause scan          AI conflict check          Groq, temperature 0, JSON out
(playbook rules)        (explains, cannot clear a flag)
     │                      │
     └──────────┬───────────┘
                ▼
Citation check                       every model quote must be in the contract word for word
                │
                ▼
Brief                                verdict, flags by severity, deal checklist,
                                     "could not check" list → Karandeep decides
```

The graph is in `review_graph.py` (LangGraph `StateGraph`; clause scan and conflict check run
in parallel).

### Who decides what

| Decision | Made by |
|---|---|
| Entity / law mismatch, missing deal documents, register collisions | Code (rules.py, register.py) |
| Severity of every flag | Looked up from `data/playbook.json` by rule id. The model never sets it |
| Finding clauses below AtliQ's standard, explaining collisions | Model, always with a quote |
| Whether a model finding is shown | Citation check (`citations.py`): quote must be verbatim |
| Accept / dismiss (with reason) / send to counsel; mark review complete | Karandeep, in the app |
| Signing | Karandeep, outside the tool |

### Guardrails

- **Personal data** in an upload (SSNs, MRNs, DOB values, diagnosis codes) stops the review before
  any model call. Try `data/demo/harrington_sample_extract.txt`.
- **No signing, approving, sending or legal opinions**: `rules.guardrail` refuses those requests
  in the chat; the review never produces a "safe to sign".
- **Recall-first**: a register collision stays flagged even if the model thinks it is unlikely.
- **Citation check**: model findings with a quote not found in the contract are dropped and counted.
- **Register update gate**: the model can propose new register entries from a signed contract;
  only verified quotes are shown and nothing counts until Karandeep confirms.
- **Seller view**: Bhavin, Dhaval, Jay and Pranav see their own deals and statuses only; other
  clients' terms (the register, the portfolio check, register-flag details) are hidden.
- **Review completion gate**: "Mark review complete" stays disabled until every blocker and high
  flag has a decision.
- **Audit log**: `logs/workflow.jsonl` records each step and decision with IDs and counts only.
  No contract text is logged.
- **"Review not completed"**: if the model call fails, the app says so and shows nothing as
  checked by it.

---

## Files

| File | What it does |
|---|---|
| `app.py` | Streamlit UI: queue, review, register, ask, portfolio check, audit log |
| `main.py` | CLI: `queue`, `review <id>`, `ask "<question>"` |
| `api/index.py` | Web API used on Vercel (FastAPI); same review engine |
| `public/index.html` | The web page Vercel serves; calls `api/index.py` |
| `keyword_search.py` | Keyword (BM25) search used on Vercel instead of local embeddings |
| `vercel.json`, `.vercelignore` | Vercel settings |
| `review_graph.py` | LangGraph review workflow |
| `rules.py` | Deterministic checks, personal-data gate, guardrail, playbook lookup |
| `register.py` | Obligations register, collision check, model-assisted proposals |
| `citations.py` | Verbatim quote validator |
| `chain.py` | Groq LLM builder, JSON calls, "Ask the contracts" RAG chain |
| `retriever.py`, `vectorstore.py`, `ingest_contracts.py`, `ingest_all.py` | Chroma RAG: clause-level chunks of signed contracts, drafts and notes |
| `dataset.py` | Loads the capstone dataset; maps each file to its tracker row; deadlines from notes |
| `decisions.py` | Saved reviews and human decisions |
| `workflow_log.py` | Step trace and audit log |
| `check_setup.py` | Diagnostic |
| `test_contractlens.py` | The seven capstone scenarios and the guardrails |
| `data/playbook.json` | Karandeep's positions written down: rule id → severity, standard, precedent |
| `data/register_seed.json` | Obligations register with verbatim quotes from the signed contracts |

## What was added to the dataset

- `data/register_seed.json` and `data/playbook.json`: written from the signed contracts,
  `karandeep_contract_checklist.md` and `negotiation_notes.md`. Every register quote is checked
  against its source file on start-up.
- `data/demo/harrington_sample_extract.txt`: an invented patient extract (fake names) to demo the
  personal-data gate. The meeting notes describe such a file but the dataset doesn't include it.
- The file-to-tracker mapping in `dataset.py`, because the tracker has no file paths for most rows.

The `Dataset` folder itself is never modified.

## Deployment (Vercel)

Vercel runs `api/index.py` as a Python function and serves `public/index.html`. The Streamlit
screen (`app.py`) is for the laptop only and is left out of the Vercel upload.

1. Copy the `Dataset` folder inside this project folder (next to `app.py`). The app finds it there
   on its own; no `DATASET_DIR` setting is needed.
2. Push the folder to GitHub (`.env` and `.venv` are ignored).
3. Vercel → Add New → Project → import the repository. Framework preset: **Other**; leave build
   and output settings empty (`vercel.json` sets them).
4. Settings → Environment Variables: `GROQ_API_KEY` = your Groq key. Nothing else is required.
5. Deploy, then open the site URL.

Differences from the laptop version: search is keyword-based (Vercel's size limit rules out the
local embedding model); review decisions are kept in your browser, and the audit log and newly
confirmed register entries last only while a Vercel instance is warm. The dataset is synthetic,
so publishing it is fine; real contracts need the enterprise setup described in the PRD.

## Limits (prototype)

- The register holds the 17 signed contracts in the dataset. Thirteen signed contracts in the
  tracker have no file (DataNest, Seaside SOW, ShopSwift, PixelCraft, GlobalMart NDA, CareBridge
  MSA…). The app lists them under "could not check" instead of implying they were checked.
- Register triggers are keyword-based by design (recall-first); expect some "possible" flags a
  person dismisses.
- The Groq free tier limits tokens per minute; long contracts are scanned in parts
  (`SCAN_CHARS`) and the client retries on rate limits.
