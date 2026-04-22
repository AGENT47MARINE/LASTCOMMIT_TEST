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

@api.post("/v1/process", response_model=EvaluationOutput)
async def process_for_competition(data: EvaluationInput):
    """
    Main Endpoint for External Evaluation Engine.
    """
    print(f"\n--- New Request Received ---")
    print(f"Query: {data.query}")
    start_time = time.time()
    try:
        # 1. Initialize State
        initial_state = {
            "input": data.query,
            "intent": None,
            "result": None,
            "confidence": 0.0,
            "error": None,
            "steps": [],
            "retries": 0
        }
        
        # 2. Invoke the Cloud Graph
        final_state = app.invoke(initial_state)
        
        # 3. Extract the primary result as a string for the 'output' field
        result_dict = final_state.get("result", {})
        
        # Flattening logic: deep-dives into the UniversalOutput structure
        worker_result = result_dict.get("result", {}) if isinstance(result_dict, dict) else {}
        
        # Priority mapping for different worker outputs
        search_keys = ["answer", "solution", "summary", "analysis", "entities", "anomalies", "question"]
        
        answer_str = None
        for key in search_keys:
            if key in worker_result:
                answer_str = worker_result[key]
                break
        
        # Fallback to top-level if not found in nested result
        if not answer_str:
            for key in search_keys:
                if key in result_dict:
                    answer_str = result_dict[key]
                    break
        
        # Ultimate fallback
        if not answer_str:
            answer_str = str(result_dict)

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
