from typing import TypedDict, List, Annotated, Optional
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from Rag_Agent.Rag_insertion_pipeline import ingest_file, VectorStore
import os

load_dotenv()


class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    retrieval_context: str
    pdf_path: Optional[str]         # set by the UI only when a PDF was uploaded
    active_ticker: Optional[str]    # ticker of the most recently ingested filing


llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    max_output_tokens=1024,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


# -------------------------
# RAG Insertion Layer
# -------------------------
def rag_insertion(pdf_path: str) -> str:
    """Ingest the uploaded PDF into the vector DB. Returns the filing's ticker."""
    meta = ingest_file(pdf_path)
    print(f"RAG Insertion: {meta.ticker} {meta.form_type} {meta.fiscal_period_end}")
    return meta.ticker


# -------------------------
# RAG Retrieval Layer
# -------------------------
def rag_retrieval(query: str, ticker: Optional[str]) -> str:
    store = VectorStore()

    # Two-stage retrieval: 30 candidates by vector similarity, reranked by a
    # cross-encoder down to the best 6. ticker=None searches every filing.
    results = store.query_reranked(
        query_text=query,
        k=6,
        fetch_k=30,
        ticker=ticker,
        latest_only=True,
    )

    return "\n\n---\n\n".join(result["text"] for result in results)


# -------------------------
# Analysis Agent
# -------------------------
def analysis_agent(state: State):
    updates = {}
    ticker = state.get("active_ticker")

    # Ingest ONLY when a PDF was uploaded
    pdf_path = state.get("pdf_path")
    if pdf_path:
        try:
            ticker = rag_insertion(pdf_path)
            updates["active_ticker"] = ticker
        except Exception as e:
            return {
                "messages": [AIMessage(content=f"I couldn't index that PDF: {e}")],
                "pdf_path": None,   # clear it so a bad file isn't retried every turn
            }
        updates["pdf_path"] = None  # clear it so later turns don't re-ingest

    query = state["messages"][-1].content
    context = rag_retrieval(query, ticker)

    if not context.strip():
        updates["messages"] = [AIMessage(content=(
            "I couldn't find any indexed filing content for this question. "
            "Upload a 10-Q/10-K PDF in the sidebar first."
        ))]
        updates["retrieval_context"] = ""
        return updates

    response = llm.invoke([
        SystemMessage(content=(
            "You are a financial analyst. Answer using ONLY the SEC filing "
            "excerpts below. If they don't contain the answer, say so.\n\n"
            + context
        )),
        HumanMessage(content=query),
    ])

    updates["messages"] = [response]
    updates["retrieval_context"] = context
    return updates