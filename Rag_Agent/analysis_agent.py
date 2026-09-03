from typing import TypedDict, List, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from Rag_insertion_pipeline import  ingest_from_env
from langchain_core.messages import SystemMessage, HumanMessage
from Rag_insertion_pipeline import VectorStore
from langchain_google_genai import ChatGoogleGenerativeAI


class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# -------------------------
# RAG Insertion Layer
# -------------------------
def rag_insertion(state: State):
    """
    Takes information from the current state and
    inserts it into the vector database.
    """
    messages = state["messages"]

    # Get the latest message
    latest_message = messages[-1].content
    ingest_from_env()

    print("RAG Insertion:", latest_message)

    return state



class State(TypedDict):
    messages: list
    retrieval_context: str


def rag_retrieval(state: State):
    # Get the user's latest question
    query = state["messages"][-1].content

    # Connect to your existing Chroma DB
    store = VectorStore()

    # Retrieve relevant SEC filing chunks
    results = store.query(
        query_text=query,
        k=8,
        ticker="AAPL",       
        latest_only=True
    )

    #Convert retrieved chunks into context for the LLM
    context = "\n\n---\n\n".join(
        result["text"] for result in results
    )

    return {
        "retrieval_context": context,
        "messages": state["messages"],
    }

# -------------------------
# RAG Retrieval Layer
# -------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    max_output_tokens=1024,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


# -------------------------
# Analysis Agent
# -------------------------
def analysis_agent(state: State):

    # RAG insertion layer
    state = rag_insertion(state)

    # RAG retrieval layer
    state = rag_retrieval(state)
    response = llm.invoke(state["retrieval_context"])
    return {"messages": [response]}


    # Analysis logic here
    # ...

    return state