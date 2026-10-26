"""
FLOW B – QUERY (runs for every question)

    question -> search by meaning (ChromaDB) + search by words (BM25)
             -> merge both lists, keep the best 5 chunks
             -> ask Gemini to answer ONLY from those chunks
             -> answer with citations, or "I couldn't find this in the documents."
"""

import json
import os
import re
import time

import config
from ingest import embed_texts

# Very common words that carry no meaning for keyword search
STOPWORDS = set("""
a an and are as at be by can could do does for from has have how i in is it its
me my of on or our that the their them there these this to was we what when where
which who why will with you your
""".split())


def tokenize(text):
    """Lowercase words and numbers, without very common words. Used for keyword search."""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


class Retriever:
    """Finds the chunks that best match a question, using meaning AND exact words."""

    def __init__(self):
        import chromadb
        from rank_bm25 import BM25Okapi

        if not config.CHUNKS_FILE.exists():
            raise FileNotFoundError("Index not found. Run `python ingest.py` first.")

        self.chunks = json.loads(config.CHUNKS_FILE.read_text(encoding="utf-8"))
        self.by_id = {c["id"]: c for c in self.chunks}

        # meaning search: the vector store built by ingest.py
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self.collection = client.get_collection(config.COLLECTION_NAME)

        # word search: BM25 keyword index, built in memory from the same chunks
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])

    def sources(self):
        """All file names in the index (used for the filter in the app)."""
        return sorted({c["source"] for c in self.chunks})

    # --- 2a. search by meaning --------------------------------------------
    def search_meaning(self, question, n, allowed_sources=None):
        where = {"source": {"$in": list(allowed_sources)}} if allowed_sources else None
        result = self.collection.query(
            query_embeddings=embed_texts([question]),
            n_results=n,
            where=where,
        )
        return result["ids"][0]

    # --- 2b. search by exact words ----------------------------------------
    def search_words(self, question, n, allowed_sources=None):
        scores = self.bm25.get_scores(tokenize(question))
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        ids = []
        for i in ranked:
            if scores[i] <= 0:
                break  # no matching words at all
            if allowed_sources and self.chunks[i]["source"] not in allowed_sources:
                continue
            ids.append(self.chunks[i]["id"])
            if len(ids) == n:
                break
        return ids

    # --- 3. merge both lists (Reciprocal Rank Fusion) ----------------------
    def retrieve(self, question, top_k=config.TOP_K, mode="hybrid", allowed_sources=None):
        """
        mode = "hybrid" (both), "meaning" (vector only) or "words" (keyword only).
        Reciprocal Rank Fusion: a chunk gets 1/(60 + rank) points from each list
        it appears in. Chunks that rank high in BOTH lists end up on top.
        """
        n = config.CANDIDATES_PER_SEARCH
        lists = []
        if mode in ("hybrid", "meaning"):
            lists.append(self.search_meaning(question, n, allowed_sources))
        if mode in ("hybrid", "words"):
            lists.append(self.search_words(question, n, allowed_sources))

        scores = {}
        for ranked_ids in lists:
            for rank, chunk_id in enumerate(ranked_ids, start=1):
                scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (config.RRF_K + rank)

        best = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [dict(self.by_id[i], score=round(scores[i], 4)) for i in best]


# ---------------------------------------------------------------------------
# 4. Ask Gemini – answer ONLY from the retrieved chunks
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = f"""You are an assistant for the field sales team of an FMCG company.
You answer questions using ONLY the numbered sources you are given.

Rules:
- Use only facts that appear in the sources. Do not use outside knowledge.
- Salespeople use everyday words, so the question may use different words from the sources
  (for example "presents" for "gifts", "dealers" or "retailers" for "distributors", "rate" for "price").
  If a source covers the same topic in different words, use it to answer.
- After every fact, cite the source number in square brackets, like [1] or [2][3].
- Quote numbers (prices, percentages, amounts, dates) exactly as written in the sources.
- Only if none of the sources contain the information needed, reply with exactly: "{config.NOT_FOUND_MESSAGE}"
- If the sources only partly answer the question, give the part they support and say what is missing.
- Keep the answer short and practical for a salesperson: a few sentences or bullet points.
"""


def build_prompt(question, chunks):
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        blocks.append(f"[{number}] {chunk['title']} (page {chunk['page']})\n{chunk['text']}")
    return "SOURCES:\n\n" + "\n\n".join(blocks) + f"\n\nQUESTION: {question}"


class Generator:
    """
    Talks to Gemini. Tries the models in config.GEMINI_MODELS in order:
    - if a model is busy (503/429) or too slow (504/timeout), wait a little and retry
    - if it stays busy, or the model name has been retired (404), move to the next model
    """

    def __init__(self, api_key):
        from google import genai
        from google.genai import types
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=config.GEMINI_TIMEOUT_SECONDS * 1000,
                retry_options=types.HttpRetryOptions(attempts=1),  # no hidden retries; we handle them below
            ),
        )
        self.models = [os.getenv("GEMINI_MODEL")] if os.getenv("GEMINI_MODEL") else list(config.GEMINI_MODELS)
        self.model = self.models[0]   # the model that answered last time (shown in logs)

    def _settings(self):
        from google.genai import types
        settings = dict(system_instruction=SYSTEM_PROMPT, temperature=config.TEMPERATURE)
        level = config.GEMINI_THINKING.lower()
        if level != "default":
            settings["thinking_config"] = types.ThinkingConfig(thinking_level=getattr(types.ThinkingLevel, level.upper()))
        return types.GenerateContentConfig(**settings)

    def generate(self, prompt):
        last_error = None
        for model in self.models:
            for attempt in range(1, config.GEMINI_BUSY_RETRIES + 2):
                try:
                    response = self.client.models.generate_content(
                        model=model, contents=prompt, config=self._settings()
                    )
                    self.model = model
                    return (response.text or "").strip()
                except Exception as error:
                    text, lower = str(error), str(error).lower()
                    last_error = text
                    if "404" in text or "not found" in lower:
                        print(f"{model} is not available, trying the next model")
                        break                                    # retired model: go to the next one
                    busy = any(code in text for code in ("503", "429", "504", "UNAVAILABLE",
                                                         "RESOURCE_EXHAUSTED", "DEADLINE_EXCEEDED"))
                    if busy or "timeout" in lower or "timed out" in lower:
                        if attempt <= config.GEMINI_BUSY_RETRIES:
                            wait = 3 * attempt                   # 3s, 6s, 9s
                            print(f"{model} is busy, retrying in {wait}s (attempt {attempt})")
                            time.sleep(wait)
                            continue
                        print(f"{model} is still busy, trying the next model")
                        break
                    raise                                         # any other error: show it
        raise RuntimeError("Gemini is busy right now. Please try again in a minute. "
                           f"(last error: {(last_error or '')[:120]})")


# ---------------------------------------------------------------------------
# The whole query flow in one call
# ---------------------------------------------------------------------------
def answer_question(question, retriever, generator, top_k=config.TOP_K, mode="hybrid", allowed_sources=None):
    chunks = retriever.retrieve(question, top_k=top_k, mode=mode, allowed_sources=allowed_sources)
    if not chunks:
        return {"answer": config.NOT_FOUND_MESSAGE, "found": False, "chunks": [], "cited": [], "model": None}

    answer = generator.generate(build_prompt(question, chunks))
    found = "couldn't find" not in answer.lower() and "could not find" not in answer.lower()

    # which source numbers did the answer actually cite?
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 1 <= int(n) <= len(chunks)})
    return {"answer": answer, "found": found, "chunks": chunks, "cited": cited, "model": generator.model}
