import chromadb
from sentence_transformers import SentenceTransformer
import os

# Initialize the embedding model (same one used for routing)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Persistent storage for your knowledge base
CHROMA_PATH = "./chroma_db"
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name="neuron_knowledge")

def add_to_knowledge_base(text: str, metadata: dict = None):
    """Adds a snippet of text to the local database."""
    doc_id = str(hash(text))
    embedding = embedding_model.encode([text])[0].tolist()
    
    collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[metadata] if metadata else None
    )
    return doc_id

def retrieve_context(query: str, n_results: int = 3):
    """Searches the local database for relevant facts."""
    query_embedding = embedding_model.encode([query])[0].tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    # Flatten the documents into a single string of context
    context = "\n".join(results['documents'][0]) if results['documents'] else ""
    return context

# Pre-load some system knowledge for the demo
if collection.count() == 0:
    add_to_knowledge_base("NEURON-12 is an autonomous AI orchestrator built by Yagye on a LangGraph framework.")
    add_to_knowledge_base("The system runs on an RTX 5050 Laptop GPU with 8GB of VRAM and 24GB of RAM.")
    add_to_knowledge_base("It uses Gemma-4 as the primary local LLM for all reasoning and task execution.")
