"""Groq LLM + the "Ask the contracts" RAG chain (same pattern as the RAG chatbot's chain.py)."""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import config
from retriever import describe_sources, format_context, search


def build_llm(max_tokens: int | None = None, fast: bool = False):
    """Deterministic model on Groq with reasoning disabled, explicit output cap."""
    from langchain_groq import ChatGroq

    config.require_api_key()
    kwargs = dict(
        model=config.GROQ_MODEL_FAST if fast else config.GROQ_MODEL,
        temperature=config.TEMPERATURE,
        max_tokens=max_tokens or config.MAX_TOKENS,
        api_key=config.GROQ_API_KEY,
        max_retries=2,
    )
    try:
        return ChatGroq(reasoning_effort=config.REASONING_EFFORT, **kwargs)
    except TypeError:
        return ChatGroq(model_kwargs={"reasoning_effort": config.REASONING_EFFORT}, **kwargs)


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def parse_json(text: str):
    """Models sometimes wrap JSON in prose or code fences; take the first JSON value."""
    text = _strip_thinking(text)
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    candidates = [text] + ([fence.group(1)] if fence else [])
    start = min([i for i in (text.find("["), text.find("{")) if i >= 0], default=-1)
    if start >= 0:
        end = max(text.rfind("]"), text.rfind("}"))
        candidates.append(text[start:end + 1])
    for c in candidates:
        try:
            return json.loads(c)
        except (json.JSONDecodeError, TypeError):
            continue
    raise ValueError("model reply was not valid JSON")


def call_json(system: str, user: str, max_tokens: int, fast: bool = False, llm=None):
    """One structured call. `llm` lets tests pass a fake model."""
    model = llm or build_llm(max_tokens=max_tokens, fast=fast)
    reply = model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return parse_json(reply.content if hasattr(reply, "content") else str(reply))


# --------------------------------------------------------------- Ask the contracts

QA_SYSTEM = """You are Atliq ContractLens, a contract review assistant for AtliQ. Karandeep (CEO) \
reviews and signs every contract; you help him and the sellers find what the contracts say.

Answer using ONLY the numbered sources below: signed contracts, draft contracts and AtliQ's \
internal notes. Each source is labelled with its document and clause.

Rules:
1. Every factual sentence ends with the source number(s) it came from, like [2] or [1][4].
2. Never use knowledge from outside the sources. Do not invent clauses, dates, amounts or parties.
3. If the sources do not answer the question, say so in one sentence and name which contract \
or person would know. Do not guess.
4. You do not give legal opinions and you never sign, approve, reject or send anything. If the \
question needs a lawyer, say it should go to outside counsel.
5. Keep it short and plain. Quote the exact contract words when the wording matters.
6. Earlier messages only resolve references like "that clause"; they are not a source of facts.

Earlier in this conversation:
{history}

Sources:
{context}"""


def format_history(messages: list[dict], max_exchanges: int = 3) -> str:
    turns = [m for m in messages if m.get("role") in ("user", "assistant")][-(max_exchanges * 2):]
    if not turns:
        return "(no earlier messages in this conversation)"
    return "\n".join(f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'].strip()}"
                     for m in turns)


def answer(question: str, history: list[dict] | None = None, collections=("signed", "notes", "incoming"),
           where: dict | None = None, llm=None):
    """Retrieve, then answer. Returns (answer_text, sources, unsupported_citation_numbers)."""
    docs = search(question, collections=collections, where=where)
    prompt = ChatPromptTemplate.from_messages([("system", QA_SYSTEM), ("human", "{question}")])
    chain = prompt | (llm or build_llm(max_tokens=900)) | StrOutputParser()
    text = _strip_thinking(chain.invoke({"question": question, "context": format_context(docs),
                                         "history": format_history(history or [])}))
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    bad = sorted(n for n in cited if n < 1 or n > len(docs))
    return text, describe_sources(docs), bad
