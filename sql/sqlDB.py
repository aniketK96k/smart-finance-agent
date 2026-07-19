from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.llms import Ollama
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv()




db = SQLDatabase.from_uri(
    "sqlite:///F:/GenAi/ProjectFInancial/data/finance.db"
)
print(db.get_usable_table_names())

print(db.get_table_info())


llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    max_output_tokens=1024,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

toolkit = SQLDatabaseToolkit(
    db=db,
    llm=llm
)

sql_tools = toolkit.get_tools()
for tool in sql_tools:
    print(tool.name)