"""
Rag_insertion_pipeline.py
------------------
Ingests SEC filings (10-Q / 10-K text, e.g. exported from EDGAR's "View Filing
Data" viewer) into a vector DB with strong per-company / per-filing metadata
partitioning, so retrieval for Company A never bleeds into Company B.

Design:
  1. Parse raw filing text into logical SECTIONS using header patterns
     (e.g. "CONDENSED CONSOLIDATED BALANCE SHEETS", "Revenue", "Debt", ...).
  2. Build a deterministic doc_id from (CIK, form_type, fiscal_period_end,
     accession_number) -- NOT a random UUID -- so re-ingesting the same
     filing is idempotent.
  3. Chunk each section (token-bounded, with overlap) and prefix every chunk
     with a short "context header" (company/ticker/period/section) before
     embedding. This bakes disambiguating signal directly into the vector,
     on top of metadata filtering.
  4. Attach a full metadata record to every chunk (cik, ticker, company,
     form_type, fiscal period, section, source url, etc).
  5. Store in ChromaDB (local, file-backed, zero-infra). Swap `VectorStore`
     for Pinecone/Qdrant/pgvector by only touching that one class.
  6. Query helper that requires (or infers) a company filter before doing
     semantic search, since raw embeddings of numeric tables collide badly
     across companies (e.g. every 10-Q has a "Total assets" row).

Install:
    pip install chromadb pdfplumber python-dotenv sentence-transformers --break-system-packages

    Also requires Ollama running locally with an embedding model pulled:
        ollama pull nomic-embed-text
        ollama serve   # usually already running as a background service

Config (.env):
    SEC_FILING_PATH=/absolute/path/to/aapl_10q.pdf   # only for the CLI entry point
    CHROMA_PERSIST_DIR=./chroma_sec_filings          # optional, has a default
    OLLAMA_BASE_URL=http://localhost:11434           # optional, has a default
    EMBED_MODEL=nomic-embed-text                     # optional, has a default
    RERANK_ENABLED=true                              # optional, set false to skip reranking
    RERANK_MODEL=BAAI/bge-reranker-base              # optional, has a default

Usage:
    # From code (what the Streamlit upload flow uses):
    from Rag_Agent.Rag_insertion_pipeline import ingest_file
    meta = ingest_file("uploads/aapl.pdf")

    # From the command line:
    python Rag_insertion_pipeline.py
    # -> reads SEC_FILING_PATH from .env and ingests it.

    from Rag_insertion_pipeline import VectorStore
    store = VectorStore()
    store.query("What was gross margin?", ticker="AAPL")
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config -- pulled from environment / .env, no CLI flags.
# ---------------------------------------------------------------------------
SEC_FILING_PATH = os.environ.get("SEC_FILING_PATH", "")
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_sec_filings")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

# Reranking: retrieve many candidates by vector similarity, then re-score each
# (query, chunk) pair with a cross-encoder and keep only the best few.
RERANK_ENABLED = os.environ.get("RERANK_ENABLED", "true").lower() in ("1", "true", "yes")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-base")
RERANK_MAX_LENGTH = int(os.environ.get("RERANK_MAX_LENGTH", "512"))


# ---------------------------------------------------------------------------
# 0. Document loading (PDF / text). PDFs are a binary format -- plain
#    Path.read_text() returns raw PDF object streams, not the document's
#    words. pdfplumber decodes glyph/font tables and reconstructs reading
#    order per page.
# ---------------------------------------------------------------------------
def extract_text_from_pdf(path: str | Path) -> str:
    try:
        import pdfplumber
    except ImportError as e:
        raise ImportError(
            "pdfplumber is required to read PDF files. "
            "Install with: pip install pdfplumber --break-system-packages"
        ) from e

    pages_text: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages_text.append(page.extract_text() or "")
    return "\n".join(pages_text)


def load_document_text(path: str | Path) -> str:
    """Dispatches based on file extension. Add more branches (.html, .htm)
    here if you later ingest EDGAR's native HTML filings."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError(
        f"Unsupported file type '{suffix}' for {path}. "
        f"Supported: .pdf, .txt, .md"
    )


# ---------------------------------------------------------------------------
# 0b. EDGAR PDF clean-up.
#     The EDGAR viewer's "Print to PDF" output wraps statement titles across
#     two lines ("CONDENSED CONSOLIDATED BALANCE" / "SHEETS (Unaudited)") and
#     pdfplumber garbles at least one of them. The single-line header regexes
#     below can't match those, which used to make the cover page and all
#     primary financial statements fall before the first detected header and
#     get silently dropped.
#     Fix: insert a clean one-line title ABOVE each wrapped title. The original
#     lines stay in the body, so column headers like "9 Months Ended" and the
#     period dates are preserved for retrieval.
# ---------------------------------------------------------------------------
_TITLE_FIXES = [
    (r"^Cover Page Cover Page[^\n]*",
     "COVER PAGE"),
    (r"^CONDENSED CONSOLIDATED STATEMENTS[^\n]*\nOF OPERATIONS[^\n]*",
     "CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS"),
    (r"^CONDENSED CONSOLIDATED STATEMENTS[^\n]*\nOF COMPREHENSIVE INCOME[^\n]*",
     "CONDENSED CONSOLIDATED STATEMENTS OF COMPREHENSIVE INCOME"),
    (r"^CONDENSED CONSOLIDATED BALANCE\nSHEETS[^\n]*Parenthetical[^\n]*",
     "CONDENSED CONSOLIDATED BALANCE SHEETS (Parenthetical)"),
    (r"^CONDENSED CONSOLIDATED BALANCE\nSHEETS \(Unaudited\) - USD[^\n]*",
     "CONDENSED CONSOLIDATED BALANCE SHEETS"),
    (r"^CONDENSED CONSOLIDATED STATEMENTS[^\n]*\nOF SHAREHOLDER[^\n]*",
     "CONDENSED CONSOLIDATED STATEMENTS OF SHAREHOLDERS' EQUITY"),
    (r"^CONDENSED CONSOLIDATED STATEMENTS[^\n]*\nOF CASH FLOWS[^\n]*",
     "CONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS"),
]


def add_clean_titles(raw_text: str) -> str:
    for pattern, title in _TITLE_FIXES:
        raw_text = re.sub(
            pattern,
            lambda m, t=title: f"{t}\n{m.group(0)}",
            raw_text,
            flags=re.MULTILINE,
        )
    return raw_text


# Page header ("8/29/26, 12:52 PM View Filing Data") and footer (viewer URL)
# repeat on every page; they are noise inside chunks.
_CHROME_RE = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2} [AP]M View Filing Data"
    r"|https://www\.sec\.gov/cgi-bin/viewer\?.*)$",
    re.MULTILINE,
)


def strip_page_chrome(raw_text: str) -> str:
    return _CHROME_RE.sub("", raw_text)


# ---------------------------------------------------------------------------
# 1. Section splitting
# ---------------------------------------------------------------------------
# These header patterns match the kind of section titles EDGAR's XBRL viewer
# / most 10-Q / 10-K text exports use. Extend this list per filing type.
SECTION_HEADER_PATTERNS = [
    r"^COVER PAGE.*$",
    r"^CONDENSED CONSOLIDATED STATEMENTS? OF OPERATIONS.*$",
    r"^CONDENSED CONSOLIDATED STATEMENTS? OF COMPREHENSIVE INCOME.*$",
    r"^CONDENSED CONSOLIDATED BALANCE SHEETS?.*$",
    r"^CONDENSED CONSOLIDATED STATEMENTS? OF SHAREHOLDERS.? EQUITY.*$",
    r"^CONDENSED CONSOLIDATED STATEMENTS? OF CASH FLOWS.*$",
    r"^Summary of Significant Accounting Policies.*$",
    r"^Revenue$",
    r"^Earnings Per Share.*$",
    r"^Financial Instruments.*$",
    r"^Condensed Consolidated Financial Statement Details.*$",
    r"^Debt$",
    r"^Shareholders'? Equity$",
    r"^Share-Based Compensation.*$",
    r"^Commitments and Contingencies.*$",
    r"^Segment Information.*$",
    r"^Insider Trading Arrangements.*$",
    r"^Risk Factors.*$",
    r"^Management.?s Discussion and Analysis.*$",
    r"^Legal Proceedings.*$",
]
_HEADER_RE = re.compile("|".join(f"({p})" for p in SECTION_HEADER_PATTERNS), re.MULTILINE)


def slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    return re.sub(r"[\s_]+", "_", text)[:60]


def split_into_sections(raw_text: str) -> list[tuple[str, str]]:
    """Returns list of (section_title, section_text). Falls back to one
    'full_document' section if no headers are detected."""
    matches = list(_HEADER_RE.finditer(raw_text))
    if not matches:
        return [("full_document", raw_text)]

    sections: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        title = m.group(0).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        body = raw_text[start:end].strip()
        if body:
            sections.append((title, body))
    return sections


# ---------------------------------------------------------------------------
# 2. Chunking
# ---------------------------------------------------------------------------
# Financial tables must never be split mid-row/mid-table -- a chunk boundary
# landing between a header row ("Americas Europe Greater China ...") and its
# data row ("$45,781 $29,395 $18,816 ...") makes the numbers unattributable.
# So: detect whether a section is table-like, and if so keep the WHOLE
# section as a single chunk (chunked "by section", not by word count).
# Only prose-heavy sections (MD&A, risk factors, accounting policy text)
# fall back to word-bounded chunking.

_DOLLAR_OR_NUMBER_RE = re.compile(r"\$\s?-?\(?\d[\d,]*\)?|\b\d{2,}\b")


def is_table_like(text: str, density_threshold: float = 0.08) -> bool:
    """Heuristic: sections dominated by dollar amounts / numeric tokens are
    financial tables (balance sheet, income statement, segment tables,
    XBRL 'Details' sections, etc). Density = numeric tokens / total words."""
    words = text.split()
    if not words:
        return False
    numeric_hits = len(_DOLLAR_OR_NUMBER_RE.findall(text))
    return (numeric_hits / len(words)) >= density_threshold


def chunk_text(text: str, max_words: int = 250, overlap_words: int = 40) -> list[str]:
    """Word-bounded chunking for prose sections only (tables bypass this --
    see build_chunks)."""
    words = text.split()
    if len(words) <= max_words:
        return [text.strip()]
    chunks = []
    step = max_words - overlap_words
    for start in range(0, len(words), step):
        piece = " ".join(words[start:start + max_words])
        chunks.append(piece.strip())
        if start + max_words >= len(words):
            break
    return chunks


# ---------------------------------------------------------------------------
# 3. Filing metadata + deterministic IDs
# ---------------------------------------------------------------------------
@dataclass
class FilingMeta:
    cik: str                     # canonical company identifier -- NEVER changes
    ticker: str
    company_name: str
    form_type: str                # "10-Q", "10-K", "8-K", ...
    fiscal_period_end: str        # ISO date, e.g. "2026-06-27"
    accession_number: str         # SEC accession number, unique per filing
    source_url: str = ""
    filed_date: str = ""
    fiscal_quarter_label: str = ""  # e.g. "Q3 FY2026"
    is_latest: bool = True          # flip False when superseded by an amendment

    def doc_id(self) -> str:
        cik = self.cik.zfill(10)
        return f"{cik}_{self.form_type}_{self.fiscal_period_end}_{self.accession_number}"


@dataclass
class Chunk:
    chunk_id: str
    text: str            # embed THIS (includes context prefix)
    display_text: str    # store this for showing to the user (no prefix)
    metadata: dict[str, Any] = field(default_factory=dict)


def build_context_prefix(meta: FilingMeta, section_title: str) -> str:
    period = meta.fiscal_quarter_label or meta.fiscal_period_end
    return (
        f"[{meta.company_name} ({meta.ticker}) — {meta.form_type} — "
        f"{period} (period ended {meta.fiscal_period_end}) — {section_title}]\n"
    )


# ---------------------------------------------------------------------------
# 3b. Auto-extract filing metadata from the cover page text, so the user
#     only has to point SEC_FILING_PATH at a file. This is a best-effort
#     parse of the standard EDGAR viewer cover page fields -- always
#     sanity-check the printed summary against the actual filing.
# ---------------------------------------------------------------------------
_COVER_PATTERNS = {
    "form_type": r"Document Type\s+(\S+)",
    "company_name": r"Entity Registrant Name\s+([^\n]+)",
    "ticker": r"Trading Symbol\s+([A-Z]+)",
    "cik": r"Entity Central Index Key\s+(\d+)",
    "period_end_raw": r"Document Period End Date\s+([A-Za-z]+\.?\s+\d{1,2},\s+\d{4})",
    "fiscal_year": r"Document Fiscal Year Focus\s+(\d{4})",
    "fiscal_quarter": r"Document Fiscal Period Focus\s+(\S+)",
    "accession_number": r"accession_number=([\d\-]+)",
}


def _parse_cover_date(raw: str) -> str:
    """'Jun. 27, 2026' -> '2026-06-27'. Raises ValueError if unparseable."""
    from datetime import datetime
    cleaned = raw.replace(".", "").strip()
    return datetime.strptime(cleaned, "%b %d, %Y").strftime("%Y-%m-%d")


def extract_filing_meta(raw_text: str, source_url: str = "") -> tuple[FilingMeta, list[str]]:
    """Best-effort extraction of FilingMeta from an EDGAR viewer cover page.
    Returns (meta, warnings) -- warnings lists any field that couldn't be
    found, so the caller can decide whether to prompt for it."""
    found: dict[str, str] = {}
    warnings: list[str] = []
    for field_name, pattern in _COVER_PATTERNS.items():
        m = re.search(pattern, raw_text)
        if m:
            found[field_name] = m.group(1).strip()
        else:
            warnings.append(field_name)

    fiscal_period_end = ""
    if "period_end_raw" in found:
        try:
            fiscal_period_end = _parse_cover_date(found["period_end_raw"])
        except ValueError:
            warnings.append("period_end_raw (unparseable date format)")

    fiscal_quarter_label = ""
    if "fiscal_quarter" in found and "fiscal_year" in found:
        fiscal_quarter_label = f"{found['fiscal_quarter']} FY{found['fiscal_year']}"

    if not source_url:
        url_match = re.search(r"https://www\.sec\.gov/cgi-bin/viewer\?[^\s]+", raw_text)
        if url_match:
            source_url = url_match.group(0)

    meta = FilingMeta(
        cik=found.get("cik", ""),
        ticker=found.get("ticker", ""),
        company_name=found.get("company_name", ""),
        form_type=found.get("form_type", ""),
        fiscal_period_end=fiscal_period_end,
        accession_number=found.get("accession_number", ""),
        source_url=source_url,
        fiscal_quarter_label=fiscal_quarter_label,
    )
    return meta, warnings


def build_chunks(raw_text: str, meta: FilingMeta) -> list[Chunk]:
    """Chunks primarily BY SECTION: a table-like section becomes exactly one
    chunk (chunk_0), no matter how long, so no row/column ever gets severed
    across a chunk boundary. Prose sections still get word-bounded chunking
    if they're long, since narrative text doesn't have this row-integrity
    problem and oversized prose chunks hurt retrieval precision."""
    doc_id = meta.doc_id()
    out: list[Chunk] = []
    seen_slugs: dict[str, int] = {}

    for section_title, section_body in split_into_sections(raw_text):
        section_slug = slugify(section_title)

        # Two sections can share a title (e.g. "Commitments and Contingencies
        # Disclosure" appears twice). Without this, they'd get the same chunk_id
        # and the second upsert would silently overwrite the first.
        occurrence = seen_slugs.get(section_slug, 0)
        seen_slugs[section_slug] = occurrence + 1
        if occurrence:
            section_slug = f"{section_slug}_{occurrence}"

        table_like = is_table_like(section_body)

        pieces = [section_body.strip()] if table_like else chunk_text(section_body)

        for i, piece in enumerate(pieces):
            prefix = build_context_prefix(meta, section_title)
            embed_text = prefix + piece
            chunk_id = f"{doc_id}::{section_slug}::chunk_{i}"
            md = {
                "doc_id": doc_id,
                "cik": meta.cik.zfill(10),
                "ticker": meta.ticker.upper(),
                "company_name": meta.company_name,
                "form_type": meta.form_type,
                "fiscal_period_end": meta.fiscal_period_end,
                "fiscal_quarter_label": meta.fiscal_quarter_label,
                "accession_number": meta.accession_number,
                "filed_date": meta.filed_date,
                "source_url": meta.source_url,
                "section": section_title,
                "section_slug": section_slug,
                "chunk_index": i,
                "chunk_type": "table" if table_like else "text",
                "is_latest": meta.is_latest,
            }
            out.append(Chunk(chunk_id=chunk_id, text=embed_text,
                              display_text=piece, metadata=md))
    return out


# ---------------------------------------------------------------------------
# 4. Vector store (Chroma). Swap this class for Pinecone/Qdrant/pgvector.
#    Embeddings run locally through Ollama -- free, no API key, no per-token
#    billing, and the model stays out of your Python process's memory.
#    Requires Ollama running (default http://localhost:11434) with the
#    embedding model pulled first: `ollama pull nomic-embed-text`
#    Swap EMBED_MODEL in .env for another local embedding model, e.g.
#    "mxbai-embed-large" or "all-minilm".
# ---------------------------------------------------------------------------
class VectorStore:
    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR,
                 collection_name: str = "sec_filings",
                 embed_model: str = EMBED_MODEL,
                 ollama_base_url: str = OLLAMA_BASE_URL):
        import chromadb
        from chromadb.utils import embedding_functions

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embed_fn = embedding_functions.OllamaEmbeddingFunction(
            url=ollama_base_url,
            model_name=embed_model,
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embed_fn
        )

    def upsert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self.collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )

    def query(self, query_text: str, k: int = 8,
              cik: str | None = None, ticker: str | None = None,
              form_type: str | None = None, section_slug: str | None = None,
              latest_only: bool = True) -> list[dict]:
        conditions = []
        if cik:
            conditions.append({"cik": {"$eq": cik.zfill(10)}})
        if ticker:
            conditions.append({"ticker": {"$eq": ticker.upper()}})
        if form_type:
            conditions.append({"form_type": {"$eq": form_type}})
        if section_slug:
            conditions.append({"section_slug": {"$eq": section_slug}})
        if latest_only:
            conditions.append({"is_latest": {"$eq": True}})

        where = None
        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        res = self.collection.query(
            query_texts=[query_text], n_results=k, where=where,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for doc, md, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            out.append({"text": doc, "metadata": md, "distance": dist})
        return out

    def query_reranked(self, query_text: str, k: int = 6, fetch_k: int = 30,
                       rerank: bool | None = None, **filters) -> list[dict]:
        """Two-stage retrieval: pull `fetch_k` candidates by vector similarity,
        then let a cross-encoder reorder them and keep the best `k`.
        `filters` are the same as query() (cik, ticker, form_type, ...).
        rerank=None follows RERANK_ENABLED; True/False overrides it per call."""
        use_rerank = RERANK_ENABLED if rerank is None else rerank
        if not use_rerank:
            return self.query(query_text, k=k, **filters)

        candidates = self.query(query_text, k=max(fetch_k, k), **filters)
        return get_reranker().rerank(query_text, candidates, top_k=k)


# ---------------------------------------------------------------------------
# 4b. Reranker. A bi-encoder (the embedding model) scores query and chunk
#     separately, so near-identical numeric tables ("Total assets" in every
#     statement) often score almost the same. A cross-encoder reads the query
#     and the chunk TOGETHER and is much better at picking the right one.
#     It is slower, so it only runs on the ~30 vector-search candidates.
#
#     Needs:  pip install sentence-transformers
#     The model downloads from Hugging Face on first use. If it can't load
#     (not installed, offline, ...) we log a warning and fall back to the
#     plain vector order instead of failing the question.
#
#     RERANK_MODEL options: BAAI/bge-reranker-base (default, better quality),
#     cross-encoder/ms-marco-MiniLM-L-6-v2 (much smaller and faster).
#     Chunks longer than RERANK_MAX_LENGTH tokens are truncated for scoring
#     only; the full chunk text is still what's returned.
# ---------------------------------------------------------------------------
class Reranker:
    def __init__(self, model_name: str = RERANK_MODEL,
                 max_length: int = RERANK_MAX_LENGTH):
        self.model_name = model_name
        self.max_length = max_length
        self._model = None
        self._load_failed = False

    def _load(self):
        if self._model is not None:
            return self._model
        if self._load_failed:
            return None
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, max_length=self.max_length)
        except Exception as e:  # missing package, no network, bad model name...
            self._load_failed = True
            print(f"[reranker] disabled: could not load '{self.model_name}' ({e}). "
                  f"Falling back to vector-search order.")
            return None
        return self._model

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        if not candidates:
            return []
        model = self._load()
        if model is None:
            return candidates[:top_k]

        scores = model.predict([(query, c["text"]) for c in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda p: float(p[1]), reverse=True)
        return [{**c, "rerank_score": float(s)} for c, s in ranked[:top_k]]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    """One shared instance per process, so the model loads once and stays in
    memory across Streamlit reruns instead of reloading on every question."""
    return Reranker()


# ---------------------------------------------------------------------------
# 5. Ingest entry points.
#    ingest_file(path)   -- used by the Streamlit upload flow. Raises
#                           exceptions (never sys.exit) so callers can handle them.
#    ingest_from_env()   -- CLI convenience; reads SEC_FILING_PATH from .env.
# ---------------------------------------------------------------------------
def ingest_file(path: str | Path) -> FilingMeta:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    raw_text = load_document_text(file_path)
    print(f"Loaded {len(raw_text)} chars from {file_path}")

    # Metadata comes from the RAW text: the accession number lives in the
    # page-footer URL, which strip_page_chrome() removes below.
    meta, warnings = extract_filing_meta(raw_text)
    if warnings:
        raise ValueError(
            f"Could not auto-detect {', '.join(warnings)} from the cover page. "
            f"This file may not be an EDGAR viewer export."
        )

    print(f"Detected: {meta.company_name} ({meta.ticker}) | {meta.form_type} | "
          f"{meta.fiscal_period_end} | CIK {meta.cik}")

    clean_text = strip_page_chrome(add_clean_titles(raw_text))
    chunks = build_chunks(clean_text, meta)
    print(f"doc_id: {meta.doc_id()}")
    print(f"Parsed {len(chunks)} chunks across "
          f"{len({c.metadata['section_slug'] for c in chunks})} sections.")

    store = VectorStore(persist_dir=CHROMA_PERSIST_DIR)
    store.upsert(chunks)
    print(f"Upserted into Chroma collection at '{CHROMA_PERSIST_DIR}'.")
    return meta


def ingest_from_env() -> None:
    if not SEC_FILING_PATH:
        raise SystemExit(
            "SEC_FILING_PATH is not set. Add it to your .env file, e.g.:\n"
            "  SEC_FILING_PATH=/absolute/path/to/aapl_10q.pdf"
        )
    try:
        ingest_file(SEC_FILING_PATH)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"ERROR: {e}")


if __name__ == "__main__":
    ingest_from_env()