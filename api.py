import time
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from graph import app  # Your LangGraph workflow

load_dotenv()

# Initialize FastAPI - Competition Ready
api = FastAPI(title="NEURON-12 Cloud Gateway")

# --- COMPETITION SCHEMA COMPLIANCE ---
class EvaluationInput(BaseModel):
    query: str
    assets: Optional[List[str]] = []

class EvaluationOutput(BaseModel):
    output: str  # Flat string response as required by the evaluator

@api.get("/")
async def health():
    print(">>> Health check request received")
    return {"status": "online", "engine": "Groq-70B/8B Hybrid"}

@api.post("/v1/answer", response_model=EvaluationOutput)
async def process_for_competition(data: EvaluationInput):
    """
    Modular Endpoint for External Evaluation Engine.
    Currently hard-routed to Level 05.
    """
    print(f"\n--- New Request Received ---")
    print(f"Query: {data.query}")
    start_time = time.time()
    try:
        import importlib
        # Route to the current active level (05)
        agent = importlib.import_module("challenges.05.agent")
        
        # Execute the modular agent
        answer_str = agent.run(data.query)

        # --- POST-PROCESSING FOR 100% SCORE ---
        if isinstance(answer_str, str):
            # Remove trailing periods, quotes, and whitespace
            answer_str = answer_str.strip().strip('.').strip('"').strip("'").strip()
            
        duration = time.time() - start_time
        print(f"Response: {answer_str[:100]}...")
        print(f"Processing time: {duration:.2f}s")
        print(f"----------------------------\n")
        return {"output": answer_str}

    except Exception as e:
        # Standard error response for the evaluator
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Use environment variable for port (required by Render/Railway)
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(api, host="0.0.0.0", port=port)
