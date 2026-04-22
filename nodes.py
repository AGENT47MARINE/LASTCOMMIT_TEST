import os
from typing import Optional
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from state import AgentState
from dotenv import load_dotenv
from utils import rule_based_route, semantic_route

load_dotenv()

class UniversalOutput(BaseModel):
    task: str
    status: str
    result: dict
    confidence: float
    error: Optional[str] = None

# --- CLOUD MODELS (Groq) ---
llm_70b = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm_8b = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# --- NODES ---

def classifier_node(state: AgentState):
    query = state["input"]
    rule_intent = rule_based_route(query)
    if rule_intent: return {"intent": rule_intent, "confidence": 1.0, "steps": ["Rule-routed"]}
    
    sem_intent, sem_score = semantic_route(query)
    if sem_intent and sem_score > 0.85: return {"intent": sem_intent, "confidence": sem_score, "steps": ["Semantic-routed"]}
    
    prompt = ChatPromptTemplate.from_template(
        "Classify the user intent into one of: SUMMARIZE, ENTITY, RAG, CODE, ANOMALY, STRUCTURED.\n"
        "Output ONLY the keyword.\n"
        "Input: {input}"
    )
    response = llm_70b.invoke(prompt.format(input=query))
    content = response.content.strip().upper()
    
    # Extract the first matching keyword
    for choice in ["SUMMARIZE", "ENTITY", "RAG", "CODE", "ANOMALY", "STRUCTURED"]:
        if choice in content:
            return {"intent": choice, "confidence": 0.9, "steps": [f"Groq-70B identified {choice}"]}
            
    return {"intent": "AMBIGUOUS", "confidence": 0.0, "steps": ["Groq failed to classify"]}

def code_solver_node(state: AgentState):
    """
    STRICT SCORING MODE:
    Forces the model to output exactly 'The [type] is [value].'
    """
    prompt = ChatPromptTemplate.from_template(
        "You are a math solver for a competition. "
        "Rule: You MUST respond in a single sentence following this exact pattern: 'The [sum/result/value] is [answer].' "
        "For '10 + 15', you MUST say: 'The sum is 25.' "
        "Problem: {input}"
    )
    response = llm_70b.invoke(prompt.format(input=state["input"]))
    return {"result": {"solution": response.content}, "steps": ["High-precision math solution generated"]}

def summarizer_node(state: AgentState):
    prompt = ChatPromptTemplate.from_template("Summarize concisely: {input}")
    response = llm_8b.invoke(prompt.format(input=state["input"]))
    return {"result": {"summary": response.content}, "steps": ["Groq-8B summarized"]}

def entity_extractor_node(state: AgentState):
    prompt = ChatPromptTemplate.from_template("Extract entities (JSON): {input}")
    response = llm_8b.invoke(prompt.format(input=state["input"]))
    return {"result": {"entities": response.content}, "steps": ["Groq-8B extracted entities"]}

def structured_processor_node(state: AgentState):
    prompt = ChatPromptTemplate.from_template("Analyze metrics: {input}")
    response = llm_8b.invoke(prompt.format(input=state["input"]))
    return {"result": {"analysis": response.content}, "steps": ["Groq-8B processed data"]}

def anomaly_detector_node(state: AgentState):
    prompt = ChatPromptTemplate.from_template("Find anomalies: {input}")
    response = llm_70b.invoke(prompt.format(input=state["input"]))
    return {"result": {"anomalies": response.content}, "steps": ["Groq-70B detected anomalies"]}

def ambiguity_node(state: AgentState):
    return {"result": {"question": "Please clarify."}, "steps": ["Ambiguity detected"]}

def rag_node(state: AgentState):
    return {"result": {"answer": "Cloud RAG is disabled in Lite mode."}, "steps": ["RAG bypassed"]}

def validator_node(state: AgentState):
    if not state["result"]: return {"error": "Processing failed", "steps": ["Validation failed"]}
    output = UniversalOutput(task=state.get("intent", "MULTI"), status="success", result=state["result"], confidence=state.get("confidence", 0.0))
    return {"result": output.model_dump(), "steps": ["Final validation passed"]}
