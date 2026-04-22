import os
from graph import app
from dotenv import load_dotenv

load_dotenv()

# Ensure LangSmith is tracking if keys are provided
# export LANGCHAIN_TRACING_V2=true
# export LANGCHAIN_API_KEY=your_key_here

def run_orchestrator(query: str):
    print(f"\n--- Processing Query: {query} ---")
    
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
    
    print("\n--- Final Output (Strict JSON) ---")
    import json
    print(json.dumps(final_state["result"], indent=2))
    
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
