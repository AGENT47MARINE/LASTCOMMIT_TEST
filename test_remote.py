import requests
import json
import time

URL = "https://lastcommit-test.onrender.com/v1/answer"

test_cases = [
    {
        "query": "Who is taller: the Burj Khalifa or the Eiffel Tower?",
        "expected": "Burj Khalifa"
    },
    {
        "query": "Between a Tesla Model S and a Boeing 747, which is faster?",
        "expected": "Boeing 747"
    },
    {
        "query": "Is 10.5 greater than 10.05?",
        "expected": "10.5"
    }
]

print(f"Testing Render API: {URL}")
print("-" * 50)

for i, tc in enumerate(test_cases, 1):
    payload = {"query": tc["query"], "assets": []}
    start = time.time()
    try:
        response = requests.post(URL, json=payload, timeout=30)
        duration = time.time() - start
        
        if response.status_code == 200:
            result = response.json()
            output = result.get("output", "")
            print(f"Test {i}: {tc['query']}")
            print(f"Result: '{output}' (Expected: '{tc['expected']}')")
            print(f"Time: {duration:.2f}s")
        else:
            print(f"Test {i} FAILED with status {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Test {i} ERROR: {e}")
    print("-" * 50)
