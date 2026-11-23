import csv
import io
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

from policy import is_advice_or_performance, looks_like_pii

load_dotenv()
ROOT = Path(__file__).parent
SOURCES_FILE = ROOT / "data" / "sources.csv"
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = "text-embedding-3-small"
MAX_ANSWER_SENTENCES = 2

st.set_page_config(page_title="Groww | Mutual Fund Facts Assistant", page_icon="📘", layout="centered")
st.title("📘 Mutual Fund Facts Assistant")
st.caption("Groww milestone prototype · HDFC Mutual Fund facts corpus")
st.info(
    "**Facts-only. No investment advice.** Do not enter PAN, Aadhaar, folio/account numbers, "
    "OTPs, email addresses or phone numbers. This prototype does not accept or store investor PII."
)

EXAMPLES = [
    "What is the exit load for HDFC Nifty 50 Index Fund?",
    "What is the lock-in period for HDFC ELSS Tax Saver Fund?",
    "How can I request a capital-gains statement?",
]


def get_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("OPENAI_API_KEY", "")).strip()
    except Exception:
        return ""


def source_rows():
    with SOURCES_FILE.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def extract_page(row):
    """Download one approved public source and return clean text plus metadata."""
    url = row["url"]
    headers = {"User-Agent": "Mozilla/5.0 (compatible; FactsOnlyMFFAQ/1.0)"}
    response = requests.get(url, timeout=35, headers=headers)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()

    if ".pdf" in url.lower() or "application/pdf" in content_type:
        reader = PdfReader(io.BytesIO(response.content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        soup = BeautifulSoup(response.text, "html.parser")
        for node in soup(["script", "style", "noscript", "svg", "header", "footer", "nav"]):
            node.decompose()
        text = soup.get_text(" ", strip=True)

    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 250:
        raise ValueError("Page returned too little readable text")
    return {
        "source_id": row["source_id"],
        "title": row["title"],
        "publisher": row["publisher"],
        "url": url,
        "text": text,
    }


def split_text(text, chunk_chars=1800, overlap=250):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + chunk_chars // 2, end)
            if boundary > start:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


@st.cache_resource(show_spinner="Loading official public sources and building the RAG index…")
def build_corpus(api_key: str):
    client = OpenAI(api_key=api_key)
    rows = source_rows()
    loaded_sources, failures = [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = {pool.submit(extract_page, row): row for row in rows}
        for future in as_completed(jobs):
            row = jobs[future]
            try:
                loaded_sources.append(future.result())
            except Exception as exc:
                failures.append(f"{row['source_id']}: {type(exc).__name__}")

    chunks = []
    for source in loaded_sources:
        for piece in split_text(source["text"]):
            chunks.append({
                "text": piece,
                "title": source["title"],
                "publisher": source["publisher"],
                "url": source["url"],
                "source_id": source["source_id"],
            })
    if not chunks:
        raise RuntimeError("No official source pages could be read. Check the source URLs and deployment logs.")

    vectors = []
    batch_size = 64
    for offset in range(0, len(chunks), batch_size):
        batch = chunks[offset:offset + batch_size]
        result = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[item["text"] for item in batch],
        )
        vectors.extend(item.embedding for item in result.data)

    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)
    return {
        "client": client,
        "chunks": chunks,
        "matrix": matrix,
        "loaded_count": len(loaded_sources),
        "total_sources": len(rows),
        "failures": failures,
        "retrieved_at": datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC"),
    }


SYSTEM_PROMPT = """You are a facts-only mutual-fund scheme FAQ assistant.
The only schemes in scope are HDFC Large Cap Fund, HDFC Flexi Cap Fund,
HDFC ELSS Tax Saver Fund and HDFC Nifty 50 Index Fund. The approved corpus
contains public pages/documents from HDFC AMC, SEBI and AMFI.

Rules:
- Answer only from the supplied retrieved excerpt. Do not use outside knowledge.
- If the excerpt does not explicitly support the requested fact, say you cannot
  verify it from the indexed official source. Never fill gaps by guessing.
- Do not advise buying, selling, holding, switching, redeeming or choosing a fund.
- Do not calculate, compare, rank or predict investment performance or returns.
- Do not provide personalized portfolio, suitability, legal or tax advice.
- Do not ask for, repeat or process personal data.
- Use plain, neutral language and no more than two answer sentences.
- Do not generate URLs or citations; the application appends the exact source link.
- For changing values, state the source's publication/as-of date when available.
"""


def refusal(text, url, title):
    return f"{text}\n\nSource: [{title}]({url})"


def retrieve(question, corpus):
    query_vector = corpus["client"].embeddings.create(
        model=EMBEDDING_MODEL,
        input=question,
    ).data[0].embedding
    query = np.asarray(query_vector, dtype=np.float32)
    query /= max(float(np.linalg.norm(query)), 1e-12)
    scores = corpus["matrix"] @ query
    best_indices = np.argsort(scores)[::-1][:8]
    best = corpus["chunks"][int(best_indices[0])]
    # Cite and generate from chunks belonging to the same single official page.
    same_page = [
        corpus["chunks"][int(i)]
        for i in best_indices
        if corpus["chunks"][int(i)]["url"] == best["url"]
    ][:4]
    return best, same_page, float(scores[int(best_indices[0])])


def answer_question(question, corpus):
    if looks_like_pii(question):
        return refusal(
            "Please do not share personal or account information here. This assistant does not accept or process PAN, Aadhaar, folio/account numbers, OTPs, email addresses or phone numbers.",
            "https://investor.sebi.gov.in/", "SEBI Investor",
        )

    if is_advice_or_performance(question):
        return refusal(
            "I can provide verified scheme facts, but I cannot recommend buying, selling, holding or switching funds, or compare or predict returns. For neutral scheme information, consult the official HDFC Mutual Fund factsheets.",
            "https://www.hdfcfund.com/mutual-funds/factsheets", "HDFC Mutual Fund factsheets",
        )

    best, evidence, score = retrieve(question, corpus)
    context = "\n\n".join(
        f"Source title: {item['title']}\nPublisher: {item['publisher']}\nExcerpt: {item['text']}"
        for item in evidence
    )
    completion = corpus["client"].chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nRetrieved official source excerpts:\n{context}\n\nAnswer only when the excerpts support the fact. Maximum two sentences."},
        ],
    )
    answer = (completion.choices[0].message.content or "").strip()
    answer = re.sub(r"https?://\S+", "", answer)
    answer = re.sub(r"Source:\s*\[[^\]]*\]\([^)]+\)", "", answer, flags=re.I).strip()
    sentences = re.split(r"(?<=[.!?])\s+", answer)
    answer = " ".join(sentences[:MAX_ANSWER_SENTENCES]).strip()
    if not answer:
        answer = "I couldn't verify this fact in the retrieved official source."
    return f"{answer}\n\nSource: [{best['title']}]({best['url']})"


def set_example(value):
    st.session_state["question"] = value


with st.expander("Try an example question", expanded=True):
    for example in EXAMPLES:
        st.button(example, use_container_width=True, on_click=set_example, args=(example,))

st.text_input(
    "Ask a factual question",
    key="question",
    max_chars=500,
    placeholder="e.g. What is the benchmark of HDFC Nifty 50 Index Fund?",
)

ask_col, refresh_col = st.columns(2)
with ask_col:
    ask = st.button("Get factual answer", type="primary", use_container_width=True)
with refresh_col:
    refresh = st.button("Refresh official sources", use_container_width=True)

if refresh:
    build_corpus.clear()
    st.success("Source cache cleared. Ask a question to retrieve and re-index the official pages.")

if ask:
    question = st.session_state.get("question", "").strip()
    if not question:
        st.warning("Enter a factual question first.")
    elif len(question) > 500:
        st.warning("Keep the question under 500 characters.")
    else:
        api_key = get_api_key()
        if not api_key:
            st.error("The OpenAI API key is not configured. Add OPENAI_API_KEY to your local .env file or Streamlit Cloud Secrets.")
        else:
            try:
                with st.spinner("Retrieving official source information…"):
                    corpus = build_corpus(api_key)
                    answer = answer_question(question, corpus)
                st.subheader("Answer")
                st.markdown(answer)
                st.caption(f"Last updated from sources: {corpus['retrieved_at']} (time the public source pages were fetched by this app; document publication dates may be older).")
                st.caption(f"Official pages successfully loaded: {corpus['loaded_count']} of {corpus['total_sources']} listed sources.")
                if corpus["failures"]:
                    with st.expander("Sources that could not be loaded"):
                        st.write("\n".join(corpus["failures"]))
            except Exception as exc:
                st.error(f"The assistant could not complete this request: {type(exc).__name__}. Check API billing, source availability and deployment logs.")

st.divider()
st.caption(
    "Scope: HDFC Mutual Fund · 4 schemes · Public HDFC AMC, SEBI and AMFI sources only. "
    "Educational facts-only prototype; not investment, legal or tax advice."
)
