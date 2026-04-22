import time
import logging
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from graph import app  # Your LangGraph workflow

load_dotenv()

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("interactions.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("evaluator")

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
    logger.info("Health check request received")
    return {"status": "online", "engine": "Groq-70B/8B Hybrid"}

@api.post("/v1/answer", response_model=EvaluationOutput)
async def process_for_competition(data: EvaluationInput):
    """
    Main Endpoint for External Evaluation Engine.
    With built-in retries and model fallbacks for rate-limit resilience.
    """
    logger.info(f"REQUEST START | Query: {data.query}")
    start_time = time.time()
    
    # Simple retry loop for rate limit resilience
    max_attempts = 2
    for attempt in range(max_attempts):
        try:
            # 1. Initialize State
            initial_state = {
                "input": data.query,
                "intent": None,
                "result": None,
                "confidence": 0.0,
                "error": None,
                "steps": [],
                "retries": attempt
            }
            
            # 2. Invoke the Cloud Graph
            final_state = app.invoke(initial_state)
            
            # 3. Extract the primary result as a string
            result_dict = final_state.get("result", {})
            worker_result = result_dict.get("result", {}) if isinstance(result_dict, dict) else {}
            
            search_keys = ["answer", "solution", "summary", "analysis", "entities", "anomalies", "question"]
            answer_str = None
            for key in search_keys:
                if key in worker_result:
                    answer_str = worker_result[key]
                    break
            
            if not answer_str:
                for key in search_keys:
                    if key in result_dict:
                        answer_str = result_dict[key]
                        break
            
            if not answer_str:
                answer_str = str(result_dict)

            duration = time.time() - start_time
            logger.info(f"REQUEST COMPLETE | Attempt: {attempt+1} | Duration: {duration:.4f}s | Response: '{answer_str}'")
            return {"output": answer_str}

        except Exception as e:
            logger.error(f"Error on attempt {attempt+1}: {str(e)}")
            if "rate_limit" in str(e).lower() and attempt < max_attempts - 1:
                time.sleep(2) # Short wait before retry
                continue
            if attempt == max_attempts - 1:
                # Ultimate fallback - return something better than 500
                return {"output": "Error: Service temporarily overloaded. Please retry."}
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Use environment variable for port (required by Render/Railway)
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(api, host="0.0.0.0", port=port)
