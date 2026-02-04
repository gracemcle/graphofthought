from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
import os
from langchain_ollama import OllamaLLM

os.environ['NO_PROXY'] = 'milan0.ftpn.ornl.gov,localhost,127.0.0.1'

llm = OllamaLLM(
    model="gpt-oss:20b",
    temperature=0.7
)

class State(TypedDict):
    kernel: str
    characteristics: str

def query(state: State):
    
    pass

# Build the graph
graph = StateGraph(State)

# Add nodes
graph.add_node("query", query)

# Set entry point
graph.set_entry_point("query")

# Compile the graph
app = graph.compile()

# Run it!
result = app.invoke({
    "kernel": "",
})

print(result["analysis"])