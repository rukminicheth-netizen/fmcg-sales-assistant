"""
All the settings for the FMCG Sales Assistant, in one place.

Changing a number here changes it everywhere, so it is easy to experiment
(for example, trying a different chunk size and re-running the evaluation).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
PDF_DIR = BASE_DIR / "data"            # the 10 FMCG PDFs live here
INDEX_DIR = BASE_DIR / "index"         # everything ingest.py builds goes here
CHROMA_DIR = INDEX_DIR / "chroma"      # the vector store (ChromaDB)
CHUNKS_FILE = INDEX_DIR / "chunks.json"  # all chunks as plain text (for keyword search)

COLLECTION_NAME = "fmcg_docs"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
# ~1000 characters is roughly a long paragraph: big enough to hold a full
# idea, small enough that the search finds the exact part that answers a question.
CHUNK_SIZE = 1000
# The last 200 characters of one chunk are repeated at the start of the next,
# so a sentence that falls on the boundary is not lost.
CHUNK_OVERLAP = 200
# Pages with less text than this (cover pages, blank pages, photo pages) are skipped.
MIN_PAGE_CHARS = 40

# ---------------------------------------------------------------------------
# Embeddings (turning text into numbers that capture meaning)
# ---------------------------------------------------------------------------
# BGE-small: a small, free, well-known English embedding model.
# It runs on the laptop (no API calls, no cost) and gives 384 numbers per chunk.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# ---------------------------------------------------------------------------
# Retrieval (finding the right chunks)
# ---------------------------------------------------------------------------
CANDIDATES_PER_SEARCH = 20   # how many chunks each search (meaning / words) returns
TOP_K = 5                    # how many chunks finally go to the LLM
RRF_K = 60                   # standard constant for Reciprocal Rank Fusion (merging two ranked lists)

# ---------------------------------------------------------------------------
# LLM (writing the answer)
# ---------------------------------------------------------------------------
# Models are tried in this order. "-latest" names always point to Google's current version,
# so the code keeps working when Google retires an older model name.
GEMINI_MODELS = [
    "gemini-flash-latest",       # main model: best quality on the free plan
    "gemini-flash-lite-latest",  # backup: used only if the main model is busy
]
TEMPERATURE = 0.1                  # low = factual and consistent, not creative
# How much Gemini "thinks" before answering: "default" (Gemini decides, most careful),
# or "high", "medium", "low", "minimal" (faster, but may miss details on complex questions).
# Accuracy matters more than speed here, so we keep Gemini's default.
GEMINI_THINKING = "default"
GEMINI_TIMEOUT_SECONDS = 120       # wait up to 2 minutes for one answer
GEMINI_BUSY_RETRIES = 3            # if a model is busy, retry this many times (waiting a little longer each time) before using the backup

NOT_FOUND_MESSAGE = "I couldn't find this in the documents."
