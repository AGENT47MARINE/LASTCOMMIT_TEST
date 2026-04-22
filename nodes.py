import os
import re
from typing import Optional
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from state import AgentState
from dotenv import load_dotenv

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


def _solve_score_comparison(query: str) -> Optional[str]:
    """Deterministically solve simple '<name> scored <num>' comparison questions."""
    pairs = re.findall(
        r"\b([A-Za-z][A-Za-z'\-]*)\s+(?:scored|got|earned|has|have|had)\s+(-?\d+(?:\.\d+)?)\b",
        query,
        flags=re.IGNORECASE,
    )
    if len(pairs) < 2:
        return None

    best_name = None
    best_score = None
    tie = False

    for name, score_raw in pairs:
        score = float(score_raw)
        if best_score is None or score > best_score:
            best_score = score
            best_name = name
            tie = False
        elif score == best_score:
            tie = True

    if tie:
        return "Equal"

    return best_name if best_name else None


def _solve_numeric_comparison(query: str) -> Optional[str]:
    """Handle standalone numeric comparison questions such as 'Compare: 40, 20'. """
    numbers = re.findall(r"-?\d+(?:\.\d+)?", query)
    if len(numbers) < 2:
        return None

    q = query.lower()
    if not any(word in q for word in ["compare", "greater", "higher", "bigger", "more", "larger", "highest", "max", "largest", "bigger"]):
        if not any(word in q for word in ["smaller", "lower", "less", "least", "minimum", "min"]):
            return None

    values = [float(n) for n in numbers[:2]]
    if values[0] == values[1]:
        return "Equal"

    if any(word in q for word in ["smaller", "lower", "less", "least", "minimum", "min"]):
        winner = numbers[0] if values[0] < values[1] else numbers[1]
    else:
        winner = numbers[0] if values[0] > values[1] else numbers[1]

    # Preserve integer formatting when possible.
    return str(int(float(winner))) if float(winner).is_integer() else winner


def _canonicalize_to_input_token(answer: str, query: str) -> str:
    """Match the answer back to the casing used in the query when possible."""
    if not answer:
        return answer

    token_map = {}
    for token in re.findall(r"[A-Za-z][A-Za-z'\-]*", query):
        token_map.setdefault(token.lower(), token)

    return token_map.get(answer.lower(), answer)


def _normalize_answer(raw: str) -> str:
    """Normalize LLM output to a strict single-line answer string."""
    text = (raw or "").strip()
    if not text:
        return text

    # Keep only the first non-empty line for strict evaluator outputs.
    for line in text.splitlines():
        line = line.strip()
        if line:
            text = line
            break

    # Remove common answer prefixes and trailing punctuation.
    text = re.sub(r"^(?:answer\s*:\s*|output\s*:\s*)", "", text, flags=re.IGNORECASE)
    text = text.strip().rstrip(".!,;:")
    return text

# --- NODES ---

def classifier_node(state: AgentState):
    query = state["input"]
    prompt = ChatPromptTemplate.from_template(
        "Classify the user intent into one of: SUMMARIZE, ENTITY, RAG, CODE, ANOMALY, STRUCTURED.\n"
        "Use CODE for arithmetic, comparisons, ranking, winner/loser questions, and tie checks.\n"
        "Use ENTITY for extracting a single fact, name, date, email, room, or event.\n"
        "Use SUMMARIZE for requests to summarize text.\n"
        "Use ANOMALY for requests to identify unusual patterns.\n"
        "Use STRUCTURED for requests involving structured data analysis or transformation.\n"
        "Use RAG only if the user asks for external knowledge lookup.\n"
        "Output ONLY the keyword, nothing else.\n"
        "Examples:\n"
        "Q: Alice scored 80, Bob scored 90. Who scored highest?\n"
        "A: CODE\n"
        "Q: Extract the email from this text.\n"
        "A: ENTITY\n"
        "Q: Summarize this paragraph.\n"
        "A: SUMMARIZE\n"
        "Input: {input}"
    )
    response = llm_70b.invoke(prompt.format(input=query))
    content = response.content.strip().upper()
    
    for choice in ["SUMMARIZE", "ENTITY", "RAG", "CODE", "ANOMALY", "STRUCTURED"]:
        if choice in content:
            return {"intent": choice, "confidence": 0.9, "steps": [f"Groq-70B identified {choice}"]}
            
    return {"intent": "ENTITY", "confidence": 0.5, "steps": ["Groq failed to classify, falling back to ENTITY"]}

def code_solver_node(state: AgentState):
    prompt = ChatPromptTemplate.from_template(
        "You are an API serving exact answers for an evaluator. Your goal is to match the expected answer string exactly.\n"
        "Respond with EXACTLY the answer string required for a 100% match. Follow these rules strictly:\n"
        "1. NO trailing punctuation.\n"
        "2. Preserve the casing used in the question when the answer is a name or entity.\n"
        "3. NO conversational filler or explanations.\n"
        "4. If the answer is a tie, return Equal.\n"
        "5. Do all reasoning inside the model and output only the final answer.\n\n"
        "Here are 3 examples to guide your output:\n"
        "Example 1:\n"
        "Q: Compare: 15, 25. Which is greater?\n"
        "A: 25\n\n"
        "Example 2:\n"
        "Q: Is an elephant bigger or a banana?\n"
        "A: Elephant\n\n"
        "Example 3:\n"
        "Q: Both Alice and Bob scored 90. Who scored higher?\n"
        "A: Equal\n\n"
        "Q: {input}\n"
        "A:"
    )
    response = llm_70b.invoke(prompt.format(input=state["input"]))
    return {
        "result": {"solution": _normalize_answer(response.content)},
        "steps": ["High-precision exact answer generated using 3 examples"]
    }

def summarizer_node(state: AgentState):
    prompt = ChatPromptTemplate.from_template(
        "Summarize the following text in exactly one concise sentence. Do not add any conversational filler.\n\nText: {input}"
    )
    response = llm_8b.invoke(prompt.format(input=state["input"]))
    return {"result": {"summary": response.content.strip()}, "steps": ["Groq-8B summarized"]}

def entity_extractor_node(state: AgentState):
    prompt = ChatPromptTemplate.from_template(
        "You are an exact string extractor. You MUST output ONLY the raw extracted entity value, with absolutely NO extra words, NO sentences, and NO punctuation at the end.\n"
        "Example 1: Extract date from 'Meeting on 12 March 2024'\nOutput: 12 March 2024\n"
        "Example 2: Extract email from 'Contact test@test.com'\nOutput: test@test.com\n\n"
        "Input: {input}\nOutput:"
    )
    response = llm_70b.invoke(prompt.format(input=state["input"]))
    return {"result": {"entities": response.content.strip()}, "steps": ["Groq-70B extracted entities"]}

def structured_processor_node(state: AgentState):
    prompt = ChatPromptTemplate.from_template(
        "You are an exact string extractor. You MUST output ONLY the raw extracted entity value, with absolutely NO extra words, NO sentences, and NO punctuation at the end.\n"
        "Example 1: Extract date from 'Meeting on 12 March 2024'\nOutput: 12 March 2024\n"
        "Example 2: Extract email from 'Contact test@test.com'\nOutput: test@test.com\n\n"
        "Input: {input}\nOutput:"
    )
    response = llm_70b.invoke(prompt.format(input=state["input"]))
    return {"result": {"analysis": response.content.strip()}, "steps": ["Groq-70B processed data"]}

def anomaly_detector_node(state: AgentState):
    prompt = ChatPromptTemplate.from_template("Identify any anomalies in one concise sentence. No conversational filler.\n\nData: {input}")
    response = llm_70b.invoke(prompt.format(input=state["input"]))
    return {"result": {"anomalies": response.content.strip()}, "steps": ["Groq-70B detected anomalies"]}

def ambiguity_node(state: AgentState):
    return {"result": {"question": "Please clarify your request."}, "steps": ["Ambiguity detected"]}

def rag_node(state: AgentState):
    return {"result": {"answer": "Cloud RAG is disabled in Lite mode."}, "steps": ["RAG bypassed"]}

def validator_node(state: AgentState):
    if not state.get("result"): return {"error": "Processing failed", "steps": ["Validation failed"]}
    output = UniversalOutput(task=state.get("intent", "MULTI"), status="success", result=state["result"], confidence=state.get("confidence", 0.0))
    return {"result": output.model_dump(), "steps": ["Final validation passed"]}
