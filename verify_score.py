from graph import app
import re

def calculate_exact_match(predicted, ground_truth):
    # Standardize: strip whitespace and convert to upper
    p = predicted.strip().upper()
    g = ground_truth.strip().upper()
    return 1.0 if p == g else 0.0

test_cases = [
    {"query": "Numbers: 2,5,8,11. Sum even numbers.", "expected": "10"},
    {"query": "Numbers: 1, 3, 5, 7. Sum even numbers.", "expected": "0"},
    {"query": "Numbers: 10, 15, 20. Sum numbers greater than 10.", "expected": "35"},
    {"query": "Numbers: 4, 9, 16. Sum odd numbers.", "expected": "9"}
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
