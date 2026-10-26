"""
FLOW A – INGESTION (runs once, or whenever PDFs are added)

    PDF  ->  extract text page by page  ->  chunk (with overlap)
         ->  tag with file name + page  ->  embed  ->  store

Run it with:
    python ingest.py
"""

import json
import re
import shutil

from pypdf import PdfReader

import config


# ---------------------------------------------------------------------------
# Step 1 – Extract text, page by page
# ---------------------------------------------------------------------------
def clean_text(text):
    """Tidy up text that comes out of a PDF."""
    text = text.replace("\x00", " ")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)   # join words broken across lines: "distri-\nbutor"
    text = re.sub(r"\s+", " ", text)               # collapse newlines / multiple spaces
    return text.strip()


def read_pdf(path):
    """Return a list of (page_number, text) for one PDF. Page numbers start at 1."""
    pages = []
    reader = PdfReader(str(path))
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = clean_text(page.extract_text() or "")
        except Exception as error:  # a few PDF pages are badly formed; skip them, don't crash
            print(f"   ! could not read page {page_number}: {error}")
            continue
        if len(text) >= config.MIN_PAGE_CHARS:
            pages.append((page_number, text))
    return pages


# ---------------------------------------------------------------------------
# Step 2 – Chunk the text (with overlap)
# ---------------------------------------------------------------------------
def split_into_chunks(text, size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    """
    Cut text into pieces of about `size` characters.
    Each new piece starts `overlap` characters before the previous one ended.
    We try to end each piece at a sentence end (". ") or at least at a space,
    so words and sentences are not cut in half.
    """
    if len(text) <= size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            # prefer to cut at the last full stop in the second half of the window
            cut = window.rfind(". ", size // 2)
            if cut == -1:
                cut = window.rfind(" ", size // 2)   # otherwise at the last space
            if cut != -1:
                end = start + cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        next_start = end - overlap
        # move the start forward to the beginning of a word
        space = text.find(" ", next_start)
        if space != -1 and space < end:
            next_start = space + 1
        start = max(next_start, start + 1)  # always move forward
    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Step 3 – Tag each chunk with where it came from (metadata)
# ---------------------------------------------------------------------------
def pretty_name(path):
    """'Hul-Code-of-Business-Principles.pdf' -> 'Hul Code of Business Principles'"""
    return re.sub(r"[-_]+", " ", path.stem).strip()


def build_chunks():
    """Read every PDF in the data folder and return a list of tagged chunks."""
    pdf_files = sorted(config.PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise SystemExit(f"No PDFs found in {config.PDF_DIR}. Put your PDFs there and run again.")

    all_chunks = []
    for path in pdf_files:
        pages = read_pdf(path)
        before = len(all_chunks)
        for page_number, page_text in pages:
            for i, chunk_text in enumerate(split_into_chunks(page_text)):
                all_chunks.append({
                    "id": f"{path.stem}::p{page_number}::c{i}",
                    "text": chunk_text,
                    "source": path.name,            # exact file name
                    "title": pretty_name(path),     # readable name for citations
                    "page": page_number,
                })
        print(f" - {path.name}: {len(pages)} pages with text -> {len(all_chunks) - before} chunks")
    return all_chunks


# ---------------------------------------------------------------------------
# Step 4 – Embed (turn text into numbers that capture meaning)
# ---------------------------------------------------------------------------
_embedder = None


def get_embedder():
    """Load the embedding model once and reuse it (the first run downloads it, ~130 MB)."""
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding
        _embedder = TextEmbedding(model_name=config.EMBEDDING_MODEL)
    return _embedder


def embed_texts(texts):
    """Return one list of 384 numbers for each text."""
    return [vector.tolist() for vector in get_embedder().embed(texts, batch_size=64)]


# ---------------------------------------------------------------------------
# Step 5 – Store in the vector store (ChromaDB) and save chunks for keyword search
# ---------------------------------------------------------------------------
def build_index(progress=print):
    """Run the whole ingestion flow. `progress` lets the Streamlit app show messages too."""
    progress("Reading and chunking PDFs...")
    chunks = build_chunks()
    progress(f"{len(chunks)} chunks created. Embedding them (this can take a few minutes)...")

    # start fresh, so re-running never leaves old or duplicate chunks behind
    if config.INDEX_DIR.exists():
        shutil.rmtree(config.INDEX_DIR)
    config.INDEX_DIR.mkdir(parents=True)

    import chromadb
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collection = client.create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # compare meanings by angle (cosine similarity)
    )

    batch = 256
    for start in range(0, len(chunks), batch):
        part = chunks[start:start + batch]
        collection.add(
            ids=[c["id"] for c in part],
            documents=[c["text"] for c in part],
            embeddings=embed_texts([c["text"] for c in part]),
            metadatas=[{"source": c["source"], "title": c["title"], "page": c["page"]} for c in part],
        )
        progress(f"   stored {min(start + batch, len(chunks))}/{len(chunks)}")

    # the keyword index (BM25) is rebuilt from this file whenever the app starts
    config.CHUNKS_FILE.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    progress(f"Done. Index saved in '{config.INDEX_DIR.name}/'.")
    return len(chunks)


if __name__ == "__main__":
    build_index()
