from graph import app

queries = [
    "Is 9 an odd number?",
    "Is 22 an even number?",
    "Is 13 an even number?"
]

for q in queries:
    initial_state = {
        "input": q,
        "intent": None,
        "result": None,
        "confidence": 0.0,
        "error": None,
        "steps": [],
        "retries": 0
    }
    print(f"Query: {q}")
    final_state = app.invoke(initial_state)
    print(f"Result: {final_state.get('result', {}).get('result', {})}")
    print("-" * 20)
