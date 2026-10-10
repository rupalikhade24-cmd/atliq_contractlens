"""Diagnostic: is this install ready?

    python check_setup.py

Checks the dataset folder, the register quotes, the API key, the vector index, and whether the
configured Groq model is actually served to this key.
"""

from __future__ import annotations

import sys

import config

OK, BAD = "[ok]  ", "[FAIL]"


def check_dataset() -> bool:
    need = ["contract_tracker.csv", "incoming", "signed_contracts", "negotiation_notes.md"]
    missing = [n for n in need if not (config.DATASET_DIR / n).exists()]
    if missing:
        print(f"{BAD} Dataset not found at {config.DATASET_DIR} (missing: {', '.join(missing)}).")
        print("      Put this folder next to the Dataset folder, or set DATASET_DIR in .env.")
        return False
    print(f"{OK} Dataset found at {config.DATASET_DIR}")
    return True


def check_register() -> bool:
    import register
    bad = [e["id"] for e in register.load_register() if not e["verified"]]
    if bad:
        print(f"{BAD} Register quotes not found in their contracts: {bad}")
        return False
    print(f"{OK} Obligations register: {len(register.load_register())} entries, all quotes verified")
    return True


def check_key() -> bool:
    if config.GROQ_API_KEY:
        if not config.GROQ_API_KEY.startswith("gsk_"):
            print(f"{BAD} GROQ_API_KEY in .env does not look like a Groq key (should start with 'gsk_').")
            return False
        print(f"{OK} GROQ_API_KEY loaded from .env")
        return True
    print(f"{BAD} GROQ_API_KEY missing. Add it to .env (see .env.example). Rule checks work without it.")
    return False


def check_index() -> bool:
    from vectorstore import count
    counts = {n: count(n) for n in ("signed", "incoming", "notes")}
    if not all(counts.values()):
        print(f"{BAD} Vector index incomplete {counts}. Run: python ingest_all.py")
        return False
    print(f"{OK} Vector index: {counts}")
    return True


def check_model() -> bool:
    try:
        from groq import Groq
        models = sorted(m.id for m in Groq(api_key=config.GROQ_API_KEY).models.list().data)
    except Exception as exc:
        print(f"{BAD} Could not reach the Groq API: {exc}")
        return False
    ok = True
    for name in sorted({config.GROQ_MODEL, config.GROQ_MODEL_FAST}):
        if name in models:
            print(f"{OK} model '{name}' is available")
        else:
            ok = False
            print(f"{BAD} model '{name}' is not served to this key. Set GROQ_MODEL in .env to one of:")
            for m in models:
                print(f"        - {m}")
    return ok


def main() -> None:
    print(f"Project: {config.BASE_DIR}\n")
    results = [check_dataset()]
    if results[0]:
        results.append(check_register())
    results.append(check_key())
    results.append(check_index())
    if config.GROQ_API_KEY:
        results.append(check_model())
    print()
    if all(results):
        print("Ready. Run `streamlit run app.py` or `python main.py queue`.")
    else:
        print("Setup incomplete - fix the [FAIL] lines above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
