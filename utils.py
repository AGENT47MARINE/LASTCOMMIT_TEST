import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize a small, fast embedding model (MiniLM is perfect for CPU/Local)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Pre-defined clusters for semantic routing
INTENT_SAMPLES = {
    "SUMMARIZE": ["make it shorter", "summary of this", "tl;dr", "condense this", "brief me"],
    "ENTITY": ["who is", "find names", "extract dates", "list companies", "identify people"],
    "RAG": ["what does the doc say", "according to the text", "find answer in file", "search documentation"],
    "CODE": ["solve this math", "write a python script", "logic puzzle", "debug this code", "calculate"]
}

# Pre-calculate embeddings for samples
SAMPLE_EMBEDDINGS = {
    intent: model.encode(samples) for intent, samples in INTENT_SAMPLES.items()
}

def rule_based_route(query: str):
    """Tier 0: Deterministic Rule Layer (<5ms)"""
    q = query.lower()
    if any(x in q for x in ["summarize", "summary", "tl;dr", "shorten"]):
        return "SUMMARIZE"
    if any(x in q for x in ["extract", "who is", "names", "dates", "list of"]):
        return "ENTITY"
    if any(x in q for x in ["{", "[", "csv", "json", "data table", "metrics"]):
        return "STRUCTURED"
    if any(x in q for x in ["solve", "code", "python", "debug", "calculate", "math"]):
        return "CODE"
    return None

def semantic_route(query: str, threshold: float = 0.7):
    """Tier 1: Semantic Embedding Layer (~20-40ms)"""
    query_embedding = model.encode([query])
    
    best_intent = None
    max_sim = 0
    
    for intent, embeddings in SAMPLE_EMBEDDINGS.items():
        sim = np.max(cosine_similarity(query_embedding, embeddings))
        if sim > max_sim:
            max_sim = sim
            best_intent = intent
            
    if max_sim >= threshold:
        return best_intent, float(max_sim)
    return None, 0.0
