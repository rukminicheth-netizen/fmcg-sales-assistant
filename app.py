"""
PART 4 – THE INTERFACE (Streamlit chat on top of the RAG pipeline)

Run it with:
    streamlit run app.py
"""

import os
import re

import streamlit as st
from dotenv import load_dotenv

import config
from rag import Generator, Retriever, answer_question

load_dotenv()  # reads GEMINI_API_KEY from the .env file on your laptop

st.set_page_config(page_title="FMCG Sales Assistant", page_icon="🛒", layout="wide")

EXAMPLE_QUESTIONS = [
    "Can a salesperson accept gifts from a distributor?",
    "What are Marico's key brands?",
    "How did ITC's FMCG business perform in Q2 FY2026?",
    "What is the price of a 70g Maggi pack?",
]


# ---------------------------------------------------------------------------
# Load things once and reuse them (Streamlit re-runs this file on every click)
# ---------------------------------------------------------------------------
def get_api_key():
    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("GEMINI_API_KEY", "")  # used on Streamlit Cloud
        except Exception:
            key = ""
    return key


@st.cache_resource(show_spinner="Loading the document index...")
def load_retriever():
    return Retriever()


@st.cache_resource
def load_generator(api_key):
    return Generator(api_key)


def build_index_from_app():
    from ingest import build_index
    status = st.status("Building the index from the PDFs...", expanded=True)
    build_index(progress=status.write)
    status.update(label="Index built.", state="complete")
    load_retriever.clear()


# ---------------------------------------------------------------------------
# Showing an answer with its sources
# ---------------------------------------------------------------------------
def show_result(result):
    if not result["found"]:
        st.warning(result["answer"])
    else:
        st.markdown(result["answer"])

    if result.get("model"):
        st.caption(f"Answered by {result['model']}")

    chunks = result["chunks"]
    if not chunks:
        return
    cited = result["cited"]

    if cited:
        st.markdown("**Sources**")
        for n in cited:
            c = chunks[n - 1]
            with st.expander(f"[{n}] {c['title']} — page {c['page']}"):
                st.caption(f"File: {c['source']}")
                st.write(c["text"])

    others = [i for i in range(1, len(chunks) + 1) if i not in cited]
    if others:
        with st.expander(f"Other chunks retrieved but not used ({len(others)})"):
            for n in others:
                c = chunks[n - 1]
                st.markdown(f"**[{n}] {c['title']} — page {c['page']}**")
                st.caption(c["text"][:400] + ("..." if len(c["text"]) > 400 else ""))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("🛒 FMCG Sales Assistant")
st.caption("Ask questions about product, policy and company documents. "
           "Every answer shows the PDF and page it came from.")

try:
    retriever = load_retriever()
except FileNotFoundError:
    st.info("The document index hasn't been built yet.")
    if st.button("Build index now", type="primary"):
        build_index_from_app()
        st.rerun()
    st.stop()

with st.sidebar:
    st.header("Settings")
    api_key = get_api_key()
    if not api_key:
        api_key = st.text_input("Gemini API key", type="password",
                                help="Get one free at aistudio.google.com")

    all_sources = retriever.sources()
    chosen = st.multiselect("Search only in these documents", all_sources,
                            help="Leave empty to search all documents")
    mode_label = st.radio("Search method",
                          ["Hybrid (meaning + words)", "Meaning only", "Words only"],
                          help="Hybrid usually works best. The others are here to compare.")
    mode = {"Hybrid (meaning + words)": "hybrid", "Meaning only": "meaning", "Words only": "words"}[mode_label]
    top_k = st.slider("Chunks sent to the AI", 3, 10, config.TOP_K)

    st.divider()
    st.markdown("**Try a question**")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state.pending = q

    st.divider()
    st.caption(f"{len(all_sources)} documents · {len(retriever.chunks)} chunks indexed")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            show_result(msg["content"])

question = st.chat_input("Ask about a product, scheme or policy...")
if not question and st.session_state.get("pending"):
    question = st.session_state.pop("pending")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not api_key:
            st.error("Add your Gemini API key in the sidebar (or in the .env file) to get answers.")
            st.stop()
        with st.spinner("Searching the documents..."):
            try:
                result = answer_question(question, retriever, load_generator(api_key),
                                         top_k=top_k, mode=mode, allowed_sources=chosen or None)
            except Exception as error:
                message = re.sub(r"key=[\w-]+", "key=***", str(error))  # never show the key on screen
                st.error(f"Something went wrong: {message}")
                st.stop()
        show_result(result)
    st.session_state.messages.append({"role": "assistant", "content": result})
