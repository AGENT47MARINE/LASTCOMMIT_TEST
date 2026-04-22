import re

def rule_based_route(query: str):
    """Tier 0: Ultra-fast regex routing (No LLM, No Embeddings)"""
    q = query.lower()
    
    # Priority 1: Math/Logic (CODE)
    if any(word in q for word in ["calculate", "math", "code", "solve", "+", "-", "*", "/"]):
        return "CODE"
    
    # Priority 2: Summarization
    if any(word in q for word in ["summarize", "summary", "tl;dr"]):
        return "SUMMARIZE"
        
    # Priority 3: Facts/Entities
    if any(word in q for word in ["extract", "entities", "who is", "what is"]):
        return "ENTITY"
        
    return None

def semantic_route(query: str):
    return None, 0.0
