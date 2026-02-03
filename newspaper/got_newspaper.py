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

class NewspaperState(TypedDict):
    topic: str
    headline: str
    gossip: str
    jobs: str
    comics: str
    weather: str
    score: int
    newspaper: str

def generate_headline(state: NewspaperState) -> NewspaperState:
    topic = state["topic"]
    prompt = f"Write a catchy, attention-grabbing newspaper headline about: {topic}"
    headline = llm.invoke(prompt)
    return {"headline": headline}

def generate_gossip(state: NewspaperState) -> NewspaperState:
    topic = state["topic"]
    prompt = f"Write a funny gossip column entry about: {topic}. Make it satirical and entertaining."
    gossip = llm.invoke(prompt)
    return {"gossip": gossip}

def generate_jobs(state: NewspaperState) -> NewspaperState:
    topic = state["topic"]
    prompt = f"Write 2-3 job listings related to: {topic}."
    jobs = llm.invoke(prompt)
    return {"jobs": jobs}

def generate_comics(state: NewspaperState) -> NewspaperState:
    topic = state["topic"]
    prompt = f"Write 3 short jokes or one-liners about: {topic}. Make them punny and clever."
    comics = llm.invoke(prompt)
    return {"comics": comics}

def generate_weather(state: NewspaperState) -> NewspaperState:
    topic = state["topic"]
    prompt = f"Write a forecast for the newstopic: {topic}."
    weather = llm.invoke(prompt)
    return {"weather": weather}

def evaluate_content(state: NewspaperState) -> NewspaperState:
    # Score each section
    topic = state["topic"]
    
    eval_prompt = f"""
    Evaluate these newspaper sections about "{topic}" on a scale of 1-10:
    
    Headline: {state['headline']}
    Gossip: {state['gossip']}
    Jobs: {state['jobs']}
    Comics: {state['comics']}
    
    If all of the sections were in a newspaper, rate how consistent and coherent it would be. Report only the integer (1-10).
    """
    
    score = llm.invoke(eval_prompt)
    
    return {"score": score}

def synthesize_newspaper(state: NewspaperState) -> NewspaperState:
    newspaper_prompt = f"If the evaluation score is above a 6, synthesize and format each of the news sections into a newsletter in markdown. Ensure consistent tone and add connective tissue. Else, do not synthesize and report that the score was bad, and needs to go back to editing."
    newspaper = llm.invoke(newspaper_prompt)
    return {"newspaper": newspaper}

# Build the graph
workflow = StateGraph(NewspaperState)

# Add nodes
workflow.add_node("generate_headline", generate_headline)
workflow.add_node("generate_gossip", generate_gossip)
workflow.add_node("generate_jobs", generate_jobs)
workflow.add_node("generate_comics", generate_comics)
workflow.add_node("generate_weather", generate_weather)
workflow.add_node("evaluate_content", evaluate_content)
workflow.add_node("synthesize_newspaper", synthesize_newspaper)

# Set entry point
workflow.set_entry_point("generate_headline")

# Add edges (parallel generation)
workflow.add_edge("generate_headline", "generate_gossip")
workflow.add_edge("generate_headline", "generate_jobs")
workflow.add_edge("generate_headline", "generate_comics")
workflow.add_edge("generate_headline", "generate_weather")

# All parallel tasks feed into evaluation
workflow.add_edge("generate_gossip", "evaluate_content")
workflow.add_edge("generate_jobs", "evaluate_content")
workflow.add_edge("generate_comics", "evaluate_content")
workflow.add_edge("generate_weather", "evaluate_content")

# Evaluation feeds into synthesis
workflow.add_edge("evaluate_content", "synthesize_newspaper")

# Synthesis is the end
workflow.add_edge("synthesize_newspaper", END)

# Compile the graph
app = workflow.compile()

# Run it!
result = app.invoke({
    "topic": "A big terrible storm destroys nearly everything.",
    "headline": "",
    "gossip": "",
    "jobs": "",
    "comics": "",
    "evaluation_scores": {},
    "final_newspaper": ""
})

print(result["final_newspaper"])