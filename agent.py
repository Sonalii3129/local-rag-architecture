from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from duckduckgo_search import DDGS

# Math tools
def add_numbers(a: float, b: float) -> float:
    return a + b

def multiply_numbers(a: float, b: float) -> float:
    return a * b

# Web search tool
def duckduckgo_search(query: str) -> str:
    with DDGS() as ddgs:
        results = ddgs.text(query)
        for result in results:
            return result["body"]

# Word count tool
def count_words(text: str) -> int:
    return len(text.split())

# Custom tool: reverse text
def reverse_text(text: str) -> str:
    return text[::-1]

# Convert all functions to tools
add_tool = FunctionTool.from_defaults(fn=add_numbers)
multiply_tool = FunctionTool.from_defaults(fn=multiply_numbers)
search_tool = FunctionTool.from_defaults(fn=duckduckgo_search)
wordcount_tool = FunctionTool.from_defaults(fn=count_words)
reverse_tool = FunctionTool.from_defaults(fn=reverse_text)

tools = [add_tool, multiply_tool, search_tool, wordcount_tool, reverse_tool]

# Set LLM and agent
Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)
agent = ReActAgent.from_tools(tools, llm=Settings.llm, verbose=True)

# Test
response = agent.query("Reverse the sentence: Streamlit is powerful")
print(response)
