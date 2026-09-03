# 📈 Smart Finance Agent

An AI-powered multi-agent financial assistant that combines **LangGraph, Google Gemini, Alpaca paper trading, natural-language SQL, RAG-based financial analysis, portfolio risk management, and persistent conversational memory**.

The system uses a **router-based multi-agent architecture** to determine whether a user request requires financial analysis or general financial assistance.

---

## 🚀 Features

### 🤖 Multi-Agent Architecture

The application uses a router to intelligently direct user queries to the appropriate agent.

* **Router Agent** — determines which agent should handle the request.
* **General Finance Agent** — handles financial queries, SQL queries, portfolio information, trading, ERP data, and other tools.
* **Analysis Agent** — handles fundamental and technical stock analysis using the RAG pipeline.
* **Tool Node** — executes tools selected by the general finance agent.
* **Persistent Memory** — maintains conversations using LangGraph's SQLite checkpointer.

Example:

```text
"What is my AAPL balance?"
        ↓
   Router Agent
        ↓
   General Agent
        ↓
   SQL Tool
        ↓
     Response
```

While:

```text
"Analyze AAPL fundamentally"
        ↓
   Router Agent
        ↓
  Analysis Agent
        ↓
   RAG Pipeline
        ↓
 Financial Analysis
```

---

# 🏗️ Architecture

```text
                         ┌─────────────────┐
                         │     User        │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │    Streamlit    │
                         │    Frontend     │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  Router Agent   │
                         │     Gemini      │
                         └────────┬────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
          ┌─────────────────┐         ┌─────────────────┐
          │ Analysis Agent  │         │  General Agent  │
          │                 │         │                 │
          │ Fundamental     │         │ Financial Q&A   │
          │ Technical       │         │ SQL             │
          │ RAG             │         │ Trading         │
          │ Documents       │         │ Portfolio       │
          └────────┬────────┘         │ ERP             │
                   │                  │ Risk             │
                   ▼                  └────────┬────────┘
          ┌─────────────────┐                  │
          │   RAG Pipeline  │                  ▼
          │                 │         ┌─────────────────┐
          │ Documents       │         │    Tool Node    │
          │ Chunking        │         └────────┬────────┘
          │ Embeddings      │                  │
          │ Vector Store    │                  │
          │ Retrieval       │                  │
          └────────┬────────┘                  │
                   │                           │
                   └─────────────┬─────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │     Response    │
                        └─────────────────┘

                 ┌──────────────────────────┐
                 │   SQLite Checkpointer    │
                 │     Persistent Memory    │
                 └──────────────────────────┘
```

---

# 🧠 Agent Responsibilities

## 1. Router Agent

The router is the entry point for user requests.

It uses Gemini to classify the query and determine which agents are required.

Current routing logic:

```text
User Query
    ↓
Router Agent
    ↓
┌──────────────────────────────┐
│                              │
▼                              ▼
Analysis Agent             General Agent
```

The router identifies requests such as:

### Analysis requests

```text
"Analyze Apple fundamentally"
"Give me technical analysis of NVDA"
"Analyze the financial health of AAPL"
"Is this stock fundamentally strong?"
```

These are routed to the **Analysis Agent**.

### General requests

```text
"What is my wallet balance?"
"Show my holdings"
"Buy 10 shares of AAPL"
"Show my last 5 transactions"
"How diversified is my portfolio?"
```

These are routed to the **General Finance Agent**.

---

# 🔍 2. Analysis Agent

The Analysis Agent is responsible for deeper financial analysis.

It can use the RAG pipeline to retrieve relevant information from financial documents before generating an analysis.

Typical workflow:

```text
User Query
    ↓
Analysis Agent
    ↓
Document Retrieval
    ↓
Relevant Financial Documents
    ↓
LLM Analysis
    ↓
Final Response
```

The analysis agent is intended for:

* Fundamental analysis
* Technical analysis
* Financial report analysis
* Company document analysis
* SEC/financial filing analysis
* RAG-powered financial research

---

# 💬 3. General Finance Agent

The General Finance Agent handles queries that don't require the specialized analysis agent.

It is a tool-calling Gemini agent.

```text
User Query
    ↓
General Agent
    ↓
Gemini Tool Calling
    ↓
Tool Node
    ↓
Tool Execution
    ↓
General Agent
    ↓
Final Response
```

Available tool categories include:

* SQL financial database
* Alpaca account information
* Trading tools
* ERP information
* Portfolio risk
* Portfolio diversification

---

# 🛠️ Tools

## SQL Database

The agent can interact with the local financial database using natural language.

Examples:

```text
"What's my AAPL position worth?"

"Show my last 5 transactions"

"What's my current wallet balance?"

"Show all my holdings"
```

The SQL tool converts the request into an appropriate database query.

---

## 📊 Alpaca

The application integrates with the Alpaca MCP server for paper trading and account information.

Capabilities include:

* Account information
* Positions
* Market information
* Orders
* Trading operations
* Portfolio information

Raw order-placement tools are not directly exposed to the LLM.

Instead, trading operations are wrapped in controlled Python-level tools.

---

# 💰 Trading Safety

Trading operations are intentionally separated from unrestricted LLM tool calling.

The architecture prevents the model from directly deciding how multiple financial operations should be chained.

Instead:

```text
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

This allows business rules and safety checks to remain in deterministic Python code rather than relying on the LLM.

---

# 🛡️ Portfolio Risk Management

Before a trade is executed, the system can perform portfolio risk checks.

Current configurable limits include:

* Maximum individual position concentration
* Maximum sector concentration

Example:

```text
Buy AAPL
    ↓
Risk Check
    ↓
Would exceed position limit?
    ↓
YES
    ↓
needs_confirmation
```

The user can then explicitly confirm the trade.

---

# 📈 Portfolio Diversification

The application provides a portfolio diversification tool that can calculate:

* Sector allocation
* Portfolio concentration
* Largest position
* Portfolio diversification information

Example:

```text
"How diversified is my portfolio?"
```

The agent can retrieve the current portfolio and provide a sector-level breakdown.

---

# 🧾 RAG Pipeline

The analysis agent uses a Retrieval-Augmented Generation workflow.

The general pipeline is:

```text
Financial Documents
        ↓
     Loading
        ↓
     Chunking
        ↓
    Embeddings
        ↓
   Vector Store
        ↓
    Retrieval
        ↓
 Analysis Agent
        ↓
      Gemini
        ↓
    Response
```

This allows the analysis agent to ground its answers in company financial documents instead of relying entirely on the LLM's internal knowledge.

---

# 🧠 Persistent Conversation Memory

LangGraph's SQLite checkpointer is used to persist conversation state.

```text
User
 ↓
LangGraph
 ↓
State
 ↓
SQLite Checkpointer
```

Each conversation is associated with a `thread_id`.

This allows conversations to survive:

* Streamlit refreshes
* Application restarts
* New requests within the same conversation

Chat history can also be deleted using the application's history-management functionality.

---

# 🗃️ Database Architecture

The project uses SQLite databases for different purposes.

### Financial Database

```text
data/finance.db
```

Stores financial application data such as:

* Wallet
* Holdings
* Ledger
* Pending orders
* Sector information

### Conversation Database

```text
checkpoints.db
```

Stores LangGraph checkpoint state and conversation history.

---

# 🔄 Complete Request Flow

A typical request goes through the following pipeline:

```text
                    USER
                      │
                      ▼
                 Streamlit
                      │
                      ▼
                Router Agent
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
   Analysis Agent           General Agent
          │                       │
          ▼                       ▼
      RAG Tools              Tool Calling
          │                       │
          │                 ┌─────┴──────┐
          │                 │            │
          │                 ▼            ▼
          │             Tool Node    SQL/Alpaca/
          │                 │        Risk/ERP/etc.
          │                 │            │
          └─────────────────┴────────────┘
                            │
                            ▼
                         Response
```

---

# 🔐 Why Use a Multi-Agent Architecture?

A single large agent can become difficult to maintain as the number of tools and responsibilities increases.

The multi-agent design separates responsibilities.

### Router Agent

Responsible for:

```text
"What type of request is this?"
```

### General Agent

Responsible for:

```text
"Which financial tool should I use?"
```

### Analysis Agent

Responsible for:

```text
"How should I analyze this company/stock?"
```

This provides better separation of concerns and makes the system easier to extend.

---

# 🧰 Tech Stack

| Layer                  | Technology                      |
| ---------------------- | ------------------------------- |
| Agent orchestration    | LangGraph                       |
| LLM                    | Google Gemini                   |
| Multi-agent routing    | LangGraph + Gemini              |
| RAG                    | LangChain / custom RAG pipeline |
| Trading                | Alpaca MCP                      |
| Tool integration       | LangChain Tools / MCP           |
| Database               | SQLite                          |
| Financial queries      | LangChain SQLDatabaseToolkit    |
| Frontend               | Streamlit                       |
| Memory                 | LangGraph AsyncSqliteSaver      |
| Environment management | python-dotenv                   |

---

# 📁 Project Structure

```text
smart-finance-agent/
│
├── agent.py
│   └── Main LangGraph multi-agent orchestration
│
├── app.py
│   └── Streamlit frontend
│
├── Rag_Agent/
│   └── analysis_agent.py
│       └── Financial analysis agent / RAG workflow
│
├── MCP/
│   ├── mcpclient.py
│   │   └── Alpaca MCP client
│   │
│   └── trade_tool.py
│       └── Controlled trading tools
│
├── sanitizerGemini/
│   └── gemini_schema_patch.py
│       └── Gemini-compatible MCP schemas
│
├── sql/
│   ├── sqlDB.py
│   │   └── SQL database tools
│   │
│   └── risk.py
│       └── Portfolio risk/diversification tools
│
├── data/
│   └── finance.db
│       └── Financial application database
│
├── checkpoints.db
│   └── LangGraph conversation memory
│
├── setup_trading_tables.py
│   └── Database initialization
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Setup

## Prerequisites

* Python 3.10+
* `uv`
* Alpaca paper trading account
* Google Gemini API key

---

## Installation

Clone the repository:

```bash
git clone https://github.com/aniketK96k/smart-finance-agent.git
```

Move into the project:

```bash
cd smart-finance-agent
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key

AlpacaKey=your_alpaca_api_key
AlpacaSecret=your_alpaca_secret_key
```

Never commit your `.env` file to GitHub.

Add it to `.gitignore`:

```text
.env
```

---

# 🗄️ Initialize Database

Run:

```bash
python setup_trading_tables.py
```

This initializes the required financial tables.

You can seed the wallet with an initial paper balance:

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

Start Streamlit:

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 💡 Example Queries

### Portfolio

```text
What's my current wallet balance?
```

```text
Show me my holdings.
```

```text
How diversified is my portfolio?
```

### SQL

```text
Show my last 5 transactions.
```

```text
What's my AAPL position worth?
```

### Trading

```text
Buy 10 shares of AAPL.
```

```text
Sell 5 shares of MSFT.
```

### Financial Analysis

```text
Analyze AAPL fundamentally.
```

```text
Give me a technical analysis of TSLA.
```

```text
Analyze the financial health of this company.
```

---

# ⚠️ Known Limitations

### 1. First-time position risk estimation

The current risk system may not be able to calculate an accurate trade value for a symbol that has no existing local position.

A live market-price lookup can be added to improve this.

### 2. Sector information

New holdings may not automatically have sector information.

A fundamentals/market-data integration can be added to automatically populate sectors.

### 3. Pending orders

Orders placed while markets are closed may require synchronization before being reflected in the local ledger.

### 4. Single-user architecture

The current wallet and holdings database is designed primarily for a single-user application.

Multi-user authentication and user-specific portfolios can be added later.

### 5. Paper trading

The project is intended to use Alpaca paper trading by default.

---

# 🔮 Future Improvements

Potential improvements include:

* Automatic stock-price retrieval for risk calculations
* Automatic sector classification
* Background pending-order synchronization
* Multi-user authentication
* Long-term user memory
* More specialized financial agents
* Earnings-analysis agent
* News-analysis agent
* Technical-indicator agent
* Fundamental-analysis agent
* Portfolio optimization agent
* Real-time market alerts
* WhatsApp notifications
* Production-grade vector database
* Human-in-the-loop approval for trades
* Observability and agent tracing

A future architecture could look like:

```text
                         User
                           │
                           ▼
                     Router Agent
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
   Trading Agent     Analysis Agent      Research Agent
        │                  │                  │
        ▼                  ▼                  ▼
   Risk Engine             RAG              Web/News
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
                     Final Response
```

---

# 📌 Project Goal

The goal of Smart Finance Agent is to demonstrate how **LLMs, tool calling, RAG, multi-agent orchestration, financial databases, risk controls, and external APIs** can be combined into a practical financial AI system.

The project focuses on keeping important financial operations deterministic and controlled while using LLMs for natural-language understanding, routing, reasoning, and user interaction.

---

# ⚠️ Disclaimer

This project is intended for educational and experimental purposes.

It is **not financial advice**.

Trading functionality is designed for **Alpaca paper trading** by default. Enabling live trading introduces financial risk and should only be done after appropriate testing and safeguards.
