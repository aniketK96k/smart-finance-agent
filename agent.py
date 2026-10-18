from typing import TypedDict, List, Annotated, Dict, Any, Optional
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from MCP.mcpclient import get_alpaca_tools
from sanitizerGemini.gemini_schema_patch import sanitize_tools_for_gemini
from MCP.trade_tool import build_trade_tools
from sql.sqlDB import sql_tools
from dotenv import load_dotenv
import os
import asyncio
import streamlit as st
import nest_asyncio
from sql.risk import get_portfolio_diversification
from Rag_Agent.analysis_agent import analysis_agent
import json


nest_asyncio.apply()

DB_PATH = "checkpoints.db"


@st.cache_resource
def load_alpaca_tools():
    toolss = asyncio.run(get_alpaca_tools())
    return sanitize_tools_for_gemini(toolss)


ALPACATools = load_alpaca_tools()

TRADE_TOOLS = build_trade_tools(ALPACATools)
EXCLUDED_FOR_LLM = {"place_stock_order", "place_crypto_order", "place_option_order"}
alpaca_readonly_raw = [t for t in ALPACATools if t.name not in EXCLUDED_FOR_LLM]
ALPACA_READONLY = sanitize_tools_for_gemini(alpaca_readonly_raw)

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")


# --------------------------------------------------
# STATE
# --------------------------------------------------

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    needed_agents: List[str]
    pdf_path: Optional[str]         # set only when the user uploads a PDF; cleared by analysis_agent
    active_ticker: Optional[str]    # ticker of the most recently ingested filing


# --------------------------------------------------
# TOOLS
# --------------------------------------------------

sql_query_tool = next(t for t in sql_tools if t.name == "sql_db_query")


@tool
def database_query(query: str):
    """
    Execute SQL query on finance database.
    Use this for retrieving financial data.
    """
    result = sql_query_tool.invoke(query)
    if result is None or result == "":
        return "No records found"
    return str(result)


@tool
def erp_tool(query: str) -> Dict[str, Any]:
    """
    Fetch ERP information like vendor cost, purchase orders,
    inventory, and supplier details.
    """
    result = {
        "query": query,
        "vendor_cost": 300000,
        "currency": "INR",
        "status": "success",
    }
    return result


@tool
def rag_tool(query: str):
    """Fetch company documents"""
    return {"docs": ["Q2 report", "CEO notes"]}


RISK_TOOLS = sanitize_tools_for_gemini([get_portfolio_diversification])
tools = [erp_tool] + sql_tools + ALPACA_READONLY + TRADE_TOOLS + RISK_TOOLS


# --------------------------------------------------
# LLM WITH TOOLS BOUND
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    max_output_tokens=1024,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)
llm_with_tools = llm.bind_tools(tools)


def agent(state: State):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools)

ROUTER_PROMPT = """
Given this user query, decide which of these agents are needed: agent,analysis_agent.
analysis_agent :needed if user ask to analysis this stock fundamentally or technically
agent:any other query that is not related to fundamental analysis or technical analysis
Return a JSON list, e.g. ["analysis_agent", "agent"].

Query: {query}
"""

VALID_AGENTS = {"agent", "analysis_agent"}


def parse_json_list(text: str):
    """Parse the router's JSON list. Tolerates ```json fences and falls back to
    the general agent if the output is unusable."""
    try:
        cleaned = text.strip()
        cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError, AttributeError):
        return ["agent"]

    if not isinstance(parsed, list):
        return ["agent"]
    valid = [a for a in parsed if a in VALID_AGENTS]
    return valid or ["agent"]


def router_agent(state: State):
    query = state["messages"][-1].content

    response = llm.invoke([
        SystemMessage(content="You are a routing assistant. Return only JSON."),
        HumanMessage(content=ROUTER_PROMPT.format(query=query))
    ])
    # Gemini can return content as a list of blocks, so normalise to text first
    needed = parse_json_list(extract_text(response.content))
    return {"needed_agents": needed}


def route_decision(state: State):
    return state["needed_agents"]


# --------------------------------------------------
# GRAPH BUILD (compiled per-call with a fresh checkpointer connection)
# --------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(State)
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)
    graph.add_node("analysis_agent", analysis_agent)
    graph.add_node("router_agent", router_agent)

    # Router is the entry point again, so analysis_agent is reachable
    graph.add_edge(START, "router_agent")
    graph.add_conditional_edges(
        "router_agent",
        route_decision,
        {
            "analysis_agent": "analysis_agent",
            "agent": "agent",
        },
    )

    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("analysis_agent", END)
    return graph


graph = build_graph()


def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


# --------------------------------------------------
# Shared interrupt/result handling
# --------------------------------------------------

def _handle_result(result: dict):
    if "__interrupt__" in result:
        interrupt_obj = result["__interrupt__"][0]
        return {
            "status": "interrupted",
            "payload": interrupt_obj.value,
        }
    return {
        "status": "done",
        "text": extract_text(result["messages"][-1].content),
    }


# --------------------------------------------------
# CHAT (with SQLite-backed memory)
# --------------------------------------------------

async def chat_async(
    user_message: str,
    thread_id: str = "default",
    pdf_path: Optional[str] = None,
):
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        app = graph.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        inputs = {
            "messages": [HumanMessage(content=user_message)],
            "needed_agents": [],
        }
        # Only send pdf_path when there is a new upload. Never send None here:
        # state is checkpointed, so None would wipe a still-pending upload.
        if pdf_path:
            inputs["pdf_path"] = pdf_path

        result = await app.ainvoke(inputs, config=config)
        return _handle_result(result)


def chat(user_message: str, thread_id: str = "default", pdf_path: Optional[str] = None):
    return asyncio.run(chat_async(user_message, thread_id, pdf_path))


# --------------------------------------------------
# RESUME (after human approves/rejects an interrupt)
# --------------------------------------------------

async def resume_async(resume_value, thread_id: str = "default"):
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        app = graph.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        result = await app.ainvoke(
            Command(resume=resume_value),
            config=config,
        )
        return _handle_result(result)


def resume_chat(resume_value, thread_id: str = "default"):
    return asyncio.run(resume_async(resume_value, thread_id))


# --------------------------------------------------
# DELETE CHAT HISTORY (removes thread from checkpoints.db)
# --------------------------------------------------

async def delete_thread_async(thread_id: str):
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        await checkpointer.adelete_thread(thread_id)


def delete_thread(thread_id: str):
    asyncio.run(delete_thread_async(thread_id))


# --------------------------------------------------
# LOAD HISTORY (to repopulate Streamlit display after refresh)
# --------------------------------------------------

async def get_history_async(thread_id: str):
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        app = graph.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        state = await app.aget_state(config)
        if not state or not state.values.get("messages"):
            return []
        history = []
        for msg in state.values["messages"]:
            role = "user" if msg.type == "human" else "assistant"
            text = extract_text(msg.content)
            if text:
                history.append({"role": role, "content": text})
        return history


def get_history(thread_id: str):
    return asyncio.run(get_history_async(thread_id))