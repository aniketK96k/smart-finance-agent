# 📈 Smart Finance Agent

An AI-powered multi-agent financial assistant that combines **LangGraph, Google Gemini, Alpaca paper trading, natural-language SQL, RAG-based SEC filing analysis, portfolio risk management, and persistent conversational memory**.

A router decides whether a request needs deep financial analysis (answered from an uploaded SEC filing via RAG) or general financial assistance (SQL, portfolio, trading, ERP tools).

---

## 🚀 Features

- **Router-based multi-agent architecture** (LangGraph + Gemini)
- **Upload a 10-Q / 10-K PDF in the UI** and ask questions grounded in that filing
- **RAG pipeline for SEC filings**: section-aware parsing, metadata partitioning, local embeddings (Ollama), ChromaDB, and **cross-encoder reranking**
- **Natural-language SQL** over a local financial database
- **Alpaca paper trading** through controlled Python-level tools
- **Portfolio risk checks** and diversification analysis
- **Human-in-the-loop approval** for risk-flagged trades
- **Persistent conversation memory** (LangGraph SQLite checkpointer)

### Example

```
"What is my AAPL balance?"
        ↓
   Router Agent
        ↓
   General Agent → SQL Tool → Response
```

```
"What was Apple's gross margin this quarter?"
        ↓
   Router Agent
        ↓
  Analysis Agent
        ↓
  Vector search (30 candidates) → Reranker (top 6) → Gemini → Response
```

---

# 🏗️ Architecture

```
               ┌─────────────────┐
               │      User       │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │    Streamlit    │◄── Sidebar: PDF upload
               │    Frontend     │
               └────────┬────────┘
                        │  message (+ pdf_path on a new upload)
                        ▼
               ┌─────────────────┐
               │  Router Agent   │
               │     Gemini      │
               └────────┬────────┘
                        │
          ┌─────────────┴─────────────┐
          │                           │
          ▼                           ▼
┌───────────────────┐         ┌─────────────────┐
│  Analysis Agent   │         │  General Agent  │
│                   │         │                 │
│ Ingest PDF        │         │ Financial Q&A   │
│ (only on upload)  │         │ SQL             │
│ Retrieve + rerank │         │ Trading         │
│ Gemini analysis   │         │ Portfolio / Risk│
└─────────┬─────────┘         │ ERP             │
          │                   └────────┬────────┘
          ▼                            ▼
┌───────────────────┐         ┌─────────────────┐
│   RAG Pipeline    │         │    Tool Node    │
│ ChromaDB + Ollama │         └────────┬────────┘
│ embeddings        │                  │
│ Cross-encoder     │                  │
│ reranker          │                  │
└─────────┬─────────┘                  │
          └─────────────┬──────────────┘
                        ▼
               ┌─────────────────┐
               │    Response     │
               └─────────────────┘

       ┌──────────────────────────┐
       │   SQLite Checkpointer    │
       │     Persistent Memory    │
       └──────────────────────────┘
```

---

# 🧠 Agents

## 1. Router Agent

The entry point of the graph. Gemini classifies the query and returns a JSON list of the agents needed (`["analysis_agent"]`, `["agent"]`, or both). Output parsing tolerates code fences and list-style responses, and falls back to the general agent if the output is unusable.

**Analysis requests** (fundamental/technical analysis, financial health, filing questions) go to the **Analysis Agent**.
**Everything else** (balances, holdings, trades, transactions, diversification, ERP) goes to the **General Finance Agent**.

## 2. Analysis Agent

Answers questions about a company using SEC filing content retrieved with RAG.

```
Uploaded PDF? ── yes ──► Ingest into ChromaDB (once) ──┐
      │                                                 │
      no ─────────────────────────────────────────────►│
                                                        ▼
                                     Retrieve 30 candidates by vector search
                                                        ▼
                                     Rerank with a cross-encoder → top 6
                                                        ▼
                                              Gemini answers using
                                              ONLY the retrieved excerpts
```

Key behaviours:

- **Ingestion runs only when a PDF is uploaded.** The UI passes the saved file path into the graph state; the analysis agent ingests it and then clears `pdf_path`, so later turns never re-ingest.
- **Retrieval is scoped to the uploaded company.** The ticker is read from the filing's cover page and stored in state as `active_ticker`, so a question about Apple never retrieves another company's chunks.
- **Grounded answers.** Gemini is instructed to answer only from the retrieved excerpts and to say so when they don't contain the answer.
- **Clear failures.** If a PDF can't be indexed (for example it isn't an EDGAR filing export), the error is shown in chat instead of crashing the app.

## 3. General Finance Agent

A tool-calling Gemini agent for everything that isn't filing analysis.

Available tool categories:

- SQL financial database
- Alpaca account information (read-only tools)
- Trading tools (controlled, Python-level)
- ERP information
- Portfolio risk and diversification

---

# 🧾 RAG Pipeline (`Rag_Agent/Rag_insertion_pipeline.py`)

Built for SEC filings exported from EDGAR's **"View Filing Data"** viewer (printed to PDF).

```
PDF
 ↓  pdfplumber text extraction
 ↓  cover-page metadata extraction (ticker, CIK, form type, period, accession no.)
 ↓  clean-up: fix wrapped statement titles, strip page headers/footers
 ↓  section splitting (balance sheet, income statement, cash flows, notes, ...)
 ↓  chunking (tables kept whole, prose word-bounded with overlap)
 ↓  context prefix on every chunk: [Company (TICKER) — 10-Q — Q3 FY2026 — Section]
 ↓  embeddings via Ollama (nomic-embed-text)
 ↓  ChromaDB (persistent, metadata-filtered)
```

### Design decisions

- **Deterministic document IDs** from `(CIK, form type, period end, accession number)`, so re-ingesting the same filing is idempotent.
- **Table-aware chunking.** Numeric tables are never split mid-row; each table-like section stays a single chunk so values remain attributable to their column headers.
- **Context prefix before embedding** bakes company/period/section into every vector, on top of metadata filters.
- **Metadata partitioning** (`cik`, `ticker`, `form_type`, `section_slug`, `is_latest`) so retrieval for one company never bleeds into another. Every 10-Q has a "Total assets" row, so raw embeddings alone collide badly across companies.
- **EDGAR PDF clean-up.** The viewer's PDF output wraps statement titles across two lines, which would otherwise cause the cover page and all primary financial statements to be dropped. Clean one-line titles are inserted above each wrapped title, and repeated page headers/footers are removed.
- **Unique chunk IDs** even when two sections share a title.

### Two-stage retrieval with reranking

```
Question
   ↓
Stage 1 — vector search (bi-encoder, fast, rough)
   ChromaDB returns the 30 closest chunks, filtered by ticker
   ↓
Stage 2 — reranker (cross-encoder, slower, precise)
   Scores each (question, chunk) pair together and sorts
   ↓
Top 6 chunks → Gemini
```

The embedding model scores the question and each chunk separately, so near-identical financial tables often score almost the same. A cross-encoder reads the question and chunk together and is much better at picking the right one.

- Default model: `BAAI/bge-reranker-base` (downloads from Hugging Face on first use).
- Lighter option: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- If the reranker can't load, the pipeline logs a warning and falls back to plain vector order.
- Chunks longer than `RERANK_MAX_LENGTH` tokens are truncated for **scoring only**; the full chunk is still returned.

---

# 💰 Trading Safety

Trading operations are intentionally separated from unrestricted LLM tool calling.

```
LLM
 ↓
Trading Tool
 ↓
Risk Validation
 ↓
Alpaca
 ↓
Local Database Update
```

Raw order-placement tools (`place_stock_order`, `place_crypto_order`, `place_option_order`) are never exposed to the LLM. Business rules and safety checks stay in deterministic Python code.

# 🛡️ Portfolio Risk Management

Before a trade is executed, configurable limits are checked:

- Maximum individual position concentration
- Maximum sector concentration

```
Buy AAPL → Risk Check → Would exceed limit? → needs_confirmation → user approves or rejects
```

Risk-flagged trades pause the graph with a LangGraph interrupt; the Streamlit UI shows **Approve / Reject** buttons and resumes the run.

# 📈 Portfolio Diversification

The diversification tool calculates sector allocation, portfolio concentration, and the largest position.

---

# 🧠 Persistent Conversation Memory

LangGraph's SQLite checkpointer (`AsyncSqliteSaver`) persists conversation state per `thread_id`, so conversations survive Streamlit refreshes and application restarts. Chat history can be deleted from the app.

Because state is checkpointed, one-shot flags like `pdf_path` are cleared by the node that consumes them.

---

# 🗃️ Databases and Storage

| Store | Location | Purpose |
|---|---|---|
| Financial database | `data/finance.db` | Wallet, holdings, ledger, pending orders, sectors |
| Conversation memory | `checkpoints.db` | LangGraph checkpoints and chat history |
| Vector database | `chroma_sec_filings/` | Embedded SEC filing chunks |
| Uploaded PDFs | `uploads/` | Files saved by the Streamlit uploader |

---

# 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph |
| LLM | Google Gemini |
| Multi-agent routing | LangGraph + Gemini |
| PDF parsing | pdfplumber |
| Embeddings | Ollama (`nomic-embed-text`), local |
| Vector store | ChromaDB (persistent) |
| Reranking | sentence-transformers cross-encoder (`BAAI/bge-reranker-base`) |
| Trading | Alpaca MCP |
| Tool integration | LangChain Tools / MCP |
| Database | SQLite |
| Financial queries | LangChain SQLDatabaseToolkit |
| Frontend | Streamlit |
| Memory | LangGraph AsyncSqliteSaver |
| Environment management | python-dotenv |

---

# 📁 Project Structure

```
smart-finance-agent/
│
├── agent.py
│   └── LangGraph orchestration: router, agents, tools, chat/resume/history
│
├── app.py
│   └── Streamlit frontend (chat, approvals, PDF upload sidebar)
│
├── Rag_Agent/
│   ├── analysis_agent.py
│   │   └── Analysis agent: ingest-on-upload, retrieval, grounded answer
│   │
│   └── Rag_insertion_pipeline.py
│       └── PDF → sections → chunks → embeddings → ChromaDB, plus reranker
│
├── MCP/
│   ├── mcpclient.py        └── Alpaca MCP client
│   └── trade_tool.py       └── Controlled trading tools
│
├── sanitizerGemini/
│   └── gemini_schema_patch.py   └── Gemini-compatible MCP schemas
│
├── sql/
│   ├── sqlDB.py            └── SQL database tools
│   └── risk.py             └── Portfolio risk / diversification tools
│
├── data/
│   └── finance.db          └── Financial application database
│
├── uploads/                └── Uploaded PDFs (git-ignored)
├── chroma_sec_filings/     └── Vector DB (git-ignored)
├── checkpoints.db          └── LangGraph conversation memory
├── setup_trading_tables.py └── Database initialization
├── requirements.txt
└── README.md
```

---

# ⚙️ Setup

## Prerequisites

- Python 3.10+
- Alpaca paper trading account
- Google Gemini API key
- [Ollama](https://ollama.com) installed and running (for embeddings)

## Installation

```bash
git clone https://github.com/aniketK96k/smart-finance-agent.git
cd smart-finance-agent
pip install -r requirements.txt
```

The RAG features need these packages (add them to `requirements.txt` if they aren't there):

```
pdfplumber
chromadb
python-dotenv
sentence-transformers
```

## Pull the embedding model

```bash
ollama pull nomic-embed-text
```

Ollama listens on `http://localhost:11434` by default. Open that address in a browser to confirm it's running.

---

# 🔑 Environment Variables

Create a `.env` file in the project root:

```env
# --- LLM (Gemini) ---
GOOGLE_API_KEY=your_google_api_key

# --- Alpaca paper trading ---
AlpacaKey=your_alpaca_api_key
AlpacaSecret=your_alpaca_secret_key

# --- RAG: embeddings (Ollama, runs locally) ---
OLLAMA_BASE_URL=http://localhost:11434
EMBED_MODEL=nomic-embed-text

# --- RAG: vector store ---
CHROMA_PERSIST_DIR=./chroma_sec_filings

# --- RAG: reranker ---
RERANK_ENABLED=true
RERANK_MODEL=BAAI/bge-reranker-base
RERANK_MAX_LENGTH=512

# --- Command-line ingestion only (not needed for Streamlit uploads) ---
# SEC_FILING_PATH=/absolute/path/to/aapl.pdf
```

Only `GOOGLE_API_KEY`, `AlpacaKey` and `AlpacaSecret` are required. Everything else has a default.

Never commit your `.env` file. Add this to `.gitignore`:

```
.env
uploads/
chroma_sec_filings/
```

> **Changing `EMBED_MODEL`?** Delete `chroma_sec_filings/` and re-upload your filings. Different embedding models produce different vector sizes, and ChromaDB will error if one collection mixes them.

---

# 🗄️ Initialize Database

```bash
python setup_trading_tables.py
```

Seed the wallet with an initial paper balance:

```bash
python -c "
import sqlite3
conn = sqlite3.connect('data/finance.db')
conn.execute('UPDATE wallet SET balance = ? WHERE id = 1', (100000,))
conn.commit()
conn.close()
"
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

## Using filing analysis

1. Download a 10-Q or 10-K from EDGAR's **View Filing Data** viewer as a PDF (print the page to PDF).
2. In the Streamlit sidebar, upload the PDF under **Filing for analysis**.
3. Ask an analysis question, for example *"What was Apple's gross margin for the nine months ended June 27, 2026?"*

The file is indexed the first time an analysis question is asked after the upload. If your next message is routed to the general agent instead (for example *"show my holdings"*), the upload stays queued until the analysis agent runs.

The first question after starting the app is slower because the reranker model loads (and downloads, the very first time).

You can also ingest from the command line:

```bash
python Rag_Agent/Rag_insertion_pipeline.py   # reads SEC_FILING_PATH from .env
```

---

# 💡 Example Queries

### Portfolio

```
What's my current wallet balance?
Show me my holdings.
How diversified is my portfolio?
```

### SQL

```
Show my last 5 transactions.
What's my AAPL position worth?
```

### Trading

```
Buy 10 shares of AAPL.
Sell 5 shares of MSFT.
```

### Filing analysis (after uploading a PDF)

```
What was net sales for the nine months ended June 27, 2026?
What were total assets and total liabilities?
How much cash did operations generate?
Break down revenue by product category.
Summarise the segment operating income by region.
```

---

# ⚠️ Known Limitations

1. **EDGAR viewer exports only.** Metadata (ticker, CIK, period, accession number) is read from the "View Filing Data" cover page. A PDF without those fields is rejected with an error message. Supporting arbitrary PDFs would need a metadata fallback.
2. **Layout-specific section detection.** Statement titles are normalised for Apple's filing layout. Other companies may word titles differently, so check the section count after the first upload of a new company.
3. **Long table chunks are truncated for reranking.** The reranker scores about the first 512 tokens of each chunk. The full chunk is still sent to Gemini.
4. **One active company per thread.** Each upload overwrites `active_ticker`, so questions target the most recently uploaded company. Without any upload, retrieval searches all ingested filings.
5. **First-time position risk estimation.** The risk system may not calculate an accurate trade value for a symbol with no existing local position. A live price lookup can fix this.
6. **Sector information.** New holdings may not automatically have sector data.
7. **Pending orders.** Orders placed while markets are closed may need synchronisation before appearing in the local ledger.
8. **Single-user architecture.** The wallet and holdings database is designed for one user.
9. **Paper trading only by default.**

---

# 🔮 Future Improvements

- Metadata fallback so arbitrary filing PDFs can be ingested
- Ticker picker in the UI for switching between ingested filings
- Native HTML ingestion of EDGAR filings
- Hybrid search (BM25 + vectors) before reranking
- Automatic stock-price retrieval for risk calculations
- Automatic sector classification
- Background pending-order synchronisation
- Multi-user authentication
- Earnings, news, and technical-indicator agents
- Portfolio optimisation agent
- Real-time market alerts
- Production-grade vector database
- Observability and agent tracing
- RAG evaluation (retrieval and generation metrics)

---

# 📌 Project Goal

To demonstrate how **LLMs, tool calling, RAG with reranking, multi-agent orchestration, financial databases, risk controls, and external APIs** can be combined into a practical financial AI system, keeping important financial operations deterministic and controlled while using LLMs for understanding, routing, reasoning, and interaction.

---

# ⚠️ Disclaimer

This project is for educational and experimental purposes. It is **not financial advice**.

Trading functionality is designed for **Alpaca paper trading** by default. Enabling live trading introduces financial risk and should only be done after appropriate testing and safeguards.
