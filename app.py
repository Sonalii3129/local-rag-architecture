import streamlit as st
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from duckduckgo_search import DDGS

# Tool functions
def add_numbers(a: float, b: float) -> float:
    return a + b

def multiply_numbers(a: float, b: float) -> float:
    return a * b

def duckduckgo_search(query: str) -> str:
    with DDGS() as ddgs:
        results = ddgs.text(query)
        for result in results:
            return result["body"]

def count_words(text: str) -> int:
    return len(text.split())

# Custom tool
def reverse_text(text: str) -> str:
    return text[::-1]

# Register tools
add_tool = FunctionTool.from_defaults(fn=add_numbers)
multiply_tool = FunctionTool.from_defaults(fn=multiply_numbers)
search_tool = FunctionTool.from_defaults(fn=duckduckgo_search)
wordcount_tool = FunctionTool.from_defaults(fn=count_words)
reverse_tool = FunctionTool.from_defaults(fn=reverse_text)

tools = [add_tool, multiply_tool, search_tool, wordcount_tool, reverse_tool]

# Set LLM and agent
Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)
agent = ReActAgent.from_tools(tools, llm=Settings.llm, verbose=True)

# Streamlit UI
st.set_page_config(page_title="Agentic AI Web App", layout="centered")
st.title("Agentic AI Assistant")
st.write("Ask your question below or choose a sample query:")

sample_prompt = st.selectbox("Try a sample question:", [
    "",
    "What is 4.5 times 3.2 then add 7?",
    "What's the latest news about NASA?",
    "How many words are in: 'Agentic AI is useful in many ways'?",
    "Reverse the sentence: 'Lab 5 is now complete'"
])

query = st.text_input("Or type your own question:", sample_prompt)

if query:
    response = agent.query(query)
    st.subheader("Answer:")
    st.write(response)
