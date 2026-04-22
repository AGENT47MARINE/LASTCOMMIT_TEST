from fastapi.testclient import TestClient
from api import api
import math
from collections import Counter

client = TestClient(api)

def cosine(a: str, b: str) -> float:
    v1 = Counter(a.lower().split())
    v2 = Counter(b.lower().split())
    common = set(v1) & set(v2)
    num = sum(v1[x] * v2[x] for x in common)
    den = (
        math.sqrt(sum(v**2 for v in v1.values()))
        * math.sqrt(sum(v**2 for v in v2.values()))
    )
    return num / den if den else 0.0

queries = [
    ("Company X made $1.5M, Company Y made $2.0M. Which company made more?", "Company Y"),
    ("The red balloon reached 50ft, the blue balloon reached 40ft. Which went higher?", "red balloon"),
    ("Laptop A has 16GB RAM, Laptop B has 32GB RAM. Which has less RAM?", "Laptop A"),
    ("Player One scored 1500 points, Player Two scored 1200 points. Who won?", "Player One"),
    ("City A is 500 miles away, City B is 800 miles away. Which is closer?", "City A")
]

print("--- Testing API Endpoint ---")
total_cos = 0.0
for q, expected in queries:
    response = client.post("/v1/answer", json={"query": q, "assets": []})
    
    if response.status_code == 200:
        got = response.json().get("output", "")
    else:
        got = "ERROR"
        
    cos = cosine(got, expected)
    total_cos += cos
    
    print(f"Q: {q}")
    print(f"Expected: {expected}")
    print(f"Got:      {got}")
    print(f"Score:    {cos * 100:.1f}%\n")

print(f"Average API Cosine Score: {(total_cos / len(queries)) * 100:.1f}%")
