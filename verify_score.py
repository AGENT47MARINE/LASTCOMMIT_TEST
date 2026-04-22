from graph import app
import re

def calculate_exact_match(predicted, ground_truth):
    # Standardize: strip whitespace and convert to upper
    p = predicted.strip().upper()
    g = ground_truth.strip().upper()
    return 1.0 if p == g else 0.0

test_cases = [
    {"query": "Alice scored 80, Bob scored 90. Who scored highest?", "expected": "Bob"},
    {"query": "Car A goes 100mph, Car B goes 120mph. Which is faster?", "expected": "Car B"},
    {"query": "Laptop A is $1000, Laptop B is $800. Which is more expensive?", "expected": "Laptop A"},
    {"query": "John is 30 years old, Mike is 25 years old. Who is younger?", "expected": "Mike"}
]

print("--- Starting Cosine Similarity (Exact Match) Test ---")
total_score = 0

for case in test_cases:
    initial_state = {
        "input": case["query"],
        "intent": None,
        "result": None,
        "confidence": 0.0,
        "error": None,
        "steps": [],
        "retries": 0
    }
    
    final_state = app.invoke(initial_state)
    # Extract solution from nested structure
    result = final_state.get("result", {})
    output = result.get("result", {}).get("solution", "") if isinstance(result, dict) else ""
    if not output:
        # Fallback for how worker results are stored before validation
        output = result.get("solution", "")

    score = calculate_exact_match(output, case["expected"])
    total_score += score
    
    print(f"Query: {case['query']}")
    print(f"Output: '{output}' | Expected: '{case['expected']}'")
    print(f"Match Score: {score * 100}%")
    print("-" * 30)

final_accuracy = (total_score / len(test_cases)) * 100
print(f"Final Average Cosine (Exact Match) Score: {final_accuracy}%")
