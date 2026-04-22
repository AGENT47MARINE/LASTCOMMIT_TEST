import os
import logging
from graph import app
from dotenv import load_dotenv

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
logger = logging.getLogger("orchestrator")

def run_orchestrator(query: str):
    logger.info(f"TEST QUERY: {query}")
    
    # Initialize state
    initial_state = {
        "input": query,
        "intent": None,
        "result": None,
        "confidence": 0.0,
        "error": None,
        "steps": [],
        "retries": 0
    }
    
    # Run the graph
    config = {"configurable": {"thread_id": "1"}}
    final_state = app.invoke(initial_state, config=config)
    
    import json
    logger.info(f"FINAL RESULT: {json.dumps(final_state['result'], indent=2)}")
    
    print("\n--- Execution Steps ---")
    for step in final_state["steps"]:
        print(f"- {step}")

if __name__ == "__main__":
    # Test cases
    test_queries = [
        "Process this JSON data: {'revenue': 5000, 'expenses': 2000, 'profit_margin': '60%'}",
        "Please summarize this: The NEURON-12 system is a multi-agent orchestrator built with LangGraph."
    ]
    
    for q in test_queries:
        run_orchestrator(q)
