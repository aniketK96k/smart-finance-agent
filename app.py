from typing import TypedDict, List, Annotated, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    BaseMessage,
    SystemMessage,
)

from MCP.mcpclient import get_alpaca_tools
from sanitizerGemini.gemini_schema_patch import sanitize_tools_for_gemini
from MCP.trade_tool import build_trade_tools
from sql.sqlDB import sql_tools
from sql.risk import get_portfolio_diversification

from Rag_Agent.analysis_agent import analysis_agent

from dotenv import load_dotenv
import os
import asyncio
import streamlit as st
import nest_asyncio
import json


# --------------------------------------------------
# Setup
# --------------------------------------------------

nest_asyncio.apply()

DB_PATH = "checkpoints.db"

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")


# --------------------------------------------------
# State
# --------------------------------------------------

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

    # Router decision
    needed_agents: List[str]


# --------------------------------------------------
# Alpaca Tools
# --------------------------------------------------

@st.cache_resource
def load_alpaca_tools():

    toolss = asyncio.run(get_alpaca_tools())

    return sanitize_tools_for_gemini(toolss)


ALPACATools = load_alpaca_tools()


# --------------------------------------------------
# Trade Tools
# --------------------------------------------------

TRADE_TOOLS = build_trade_tools(ALPACATools)


# Don't expose actual order-placement tools
# directly to the LLM unless you intentionally
# want autonomous trading.
# --------------------------------------------------

EXCLUDED_FOR_LLM = {
    "place_stock_order",
    "place_crypto_order",
    "place_option_order",
}


alpaca_readonly_raw = [
    t
    for t in ALPACATools
    if t.name not in EXCLUDED_FOR_LLM
]

ALPACA_READONLY = sanitize_tools_for_gemini(
    alpaca_readonly_raw
)


# --------------------------------------------------
# SQL Tool
# --------------------------------------------------

sql_query_tool = next(
    t
    for t in sql_tools
    if t.name == "sql_db_query"
)


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


# --------------------------------------------------
# ERP Tool
# --------------------------------------------------

@tool
def erp_tool(query: str) -> Dict[str, Any]:
    """
    Fetch ERP information like vendor cost,
    purchase orders, inventory, and supplier details.
    """

    result = {
        "query": query,
        "vendor_cost": 300000,
        "currency": "INR",
        "status": "success",
    }

    return result


# --------------------------------------------------
# RAG Tool
# --------------------------------------------------

@tool
def rag_tool(query: str):
    """
    Fetch relevant company documents using RAG.
    """

    # TODO:
    # Replace this with your actual RAG retrieval function.

    return {
        "docs": [
            "Q2 report",
            "CEO notes"
        ]
    }


# --------------------------------------------------
# Risk Tools
# --------------------------------------------------

RISK_TOOLS = sanitize_tools_for_gemini(
    [get_portfolio_diversification]
)


# --------------------------------------------------
# All Tools
# --------------------------------------------------

tools = (
    [erp_tool, database_query, rag_tool]
    + sql_tools
    + ALPACA_READONLY
    + TRADE_TOOLS
    + RISK_TOOLS
)


# --------------------------------------------------
# LLM
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    max_output_tokens=1024,
    google_api_key=api_key,
)


llm_with_tools = llm.bind_tools(tools)


# --------------------------------------------------
# Main Agent
# --------------------------------------------------

def agent(state: State):

    messages = state["messages"]

    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response]
    }


# --------------------------------------------------
# Tool Node
# --------------------------------------------------

tool_node = ToolNode(tools)


# --------------------------------------------------
# Router Prompt
# --------------------------------------------------

ROUTER_PROMPT = """
Given the user's query, decide which agents are needed.

Available agents:

1. analysis_agent
   Use this when the user asks for:
   - fundamental analysis
   - technical analysis
   - stock analysis
   - company financial analysis
   - valuation
   - financial ratios
   - earnings analysis
   - price/technical indicators

2. agent
   Use this for:
   - general financial questions
   - database queries
   - portfolio questions
   - stock prices
   - trading actions
   - ERP queries
   - RAG/document questions
   - anything else

Return ONLY a valid JSON list.

Examples:

["analysis_agent"]

["agent"]

["analysis_agent", "agent"]

User query:
{query}
"""


# --------------------------------------------------
# Parse Router Output
# --------------------------------------------------

def parse_json_list(text: str):

    try:
        return json.loads(text)

    except json.JSONDecodeError:

        # fallback
        return ["agent"]


# --------------------------------------------------
# Router Agent
# --------------------------------------------------

def router_agent(state: State):

    # Get the latest user message
    query = state["messages"][-1].content

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a routing assistant. "
                    "Return ONLY valid JSON."
                )
            ),
            HumanMessage(
                content=ROUTER_PROMPT.format(
                    query=query
                )
            ),
        ]
    )

    needed = parse_json_list(
        response.content
    )

    return {
        "needed_agents": needed
    }


# --------------------------------------------------
# Router Decision
# --------------------------------------------------

def route_decision(state: State):

    return state["needed_agents"]


# --------------------------------------------------
# Build Graph
# --------------------------------------------------

def build_graph():

    graph = StateGraph(State)

    # Nodes
    graph.add_node(
        "router_agent",
        router_agent
    )

    graph.add_node(
        "agent",
        agent
    )

    graph.add_node(
        "tools",
        tool_node
    )

    graph.add_node(
        "analysis_agent",
        analysis_agent
    )

    # Entry point
    graph.set_entry_point(
        "router_agent"
    )

    # Router -> Agent
    graph.add_conditional_edges(
        "router_agent",
        route_decision,
        {
            "analysis_agent": "analysis_agent",
            "agent": "agent",
        },
    )

    # Agent -> tools OR END
    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )

    # Tools -> Agent
    graph.add_edge(
        "tools",
        "agent"
    )

    # Analysis agent -> END
    graph.add_edge(
        "analysis_agent",
        END
    )

    return graph


# --------------------------------------------------
# Create Graph
# --------------------------------------------------

graph = build_graph()


# --------------------------------------------------
# Extract Message Text
# --------------------------------------------------

def extract_text(content) -> str:

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        parts = []

        for block in content:

            if (
                isinstance(block, dict)
                and block.get("type") == "text"
            ):
                parts.append(
                    block.get("text", "")
                )

            elif isinstance(block, str):
                parts.append(block)

        return "".join(parts)

    return str(content)


# --------------------------------------------------
# Async Chat
# --------------------------------------------------

async def chat_async(
    user_message: str,
    thread_id: str = "default"
):

    async with AsyncSqliteSaver.from_conn_string(
        DB_PATH
    ) as checkpointer:

        app = graph.compile(
            checkpointer=checkpointer
        )

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        result = await app.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=user_message
                    )
                ],
                "needed_agents": [],
            },
            config=config,
        )

        return extract_text(
            result["messages"][-1].content
        )


# --------------------------------------------------
# Sync Chat
# --------------------------------------------------

def chat(
    user_message: str,
    thread_id: str = "default"
):

    return asyncio.run(
        chat_async(
            user_message,
            thread_id
        )
    )


# --------------------------------------------------
# Delete Conversation
# --------------------------------------------------

async def delete_thread_async(
    thread_id: str
):

    async with AsyncSqliteSaver.from_conn_string(
        DB_PATH
    ) as checkpointer:

        await checkpointer.adelete_thread(
            thread_id
        )


def delete_thread(
    thread_id: str
):

    asyncio.run(
        delete_thread_async(thread_id)
    )


# --------------------------------------------------
# Get Conversation History
# --------------------------------------------------

async def get_history_async(
    thread_id: str
):

    async with AsyncSqliteSaver.from_conn_string(
        DB_PATH
    ) as checkpointer:

        app = graph.compile(
            checkpointer=checkpointer
        )

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        state = await app.aget_state(
            config
        )

        if (
            not state
            or not state.values.get("messages")
        ):
            return []

        history = []

        for msg in state.values["messages"]:

            role = (
                "user"
                if msg.type == "human"
                else "assistant"
            )

            text = extract_text(
                msg.content
            )

            if text:

                history.append(
                    {
                        "role": role,
                        "content": text,
                    }
                )

        return history


def get_history(
    thread_id: str
):

    return asyncio.run(
        get_history_async(thread_id)
    )