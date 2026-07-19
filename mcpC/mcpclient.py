from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()



client = MultiServerMCPClient({
      
    "alpaca": {
        "transport": "stdio",
      "command": "uvx",
      "args": ["alpaca-mcp-server"],
      "env": {
        "ALPACA_API_KEY": os.getenv("AlpacaKey"),
        "ALPACA_SECRET_KEY": os.getenv("AlpacaSecret")
      }
    }
  
    })


   
async def get_alpaca_tools():
    return await client.get_tools()
