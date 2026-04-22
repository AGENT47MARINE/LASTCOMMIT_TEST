"""
scratch/test_05.py
Smoke-test for challenges/05/agent.py
Run from project root: python -m scratch.test_05
"""
import math, importlib
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

# ── Load the agent ───────────────────────────────────────────────────────────
agent = importlib.import_module("challenges.05.agent")

# ── Helpers ──────────────────────────────────────────────────────────────────
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

def jaccard_ngram(a: str, b: str, n: int = 2) -> float:
    def ngrams(text):
        t = text.lower()
        return set(t[i : i + n] for i in range(len(t) - n + 1))
    s1, s2 = ngrams(a), ngrams(b)
    if not s1 and not s2:
        return 1.0
    return len(s1 & s2) / len(s1 | s2)

# ── Test cases ────────────────────────────────────────────────────────────────
# Format: (query, expected_output)
tests = [
    # Sample case from the challenge brief
    (
        "Alice scored 80, Bob scored 90. Who scored highest?",
        "Bob",
    ),
    # Additional edge cases
    (
        "The red balloon reached 50ft, the blue balloon reached 40ft. Which went higher?",
        "red balloon",
    ),
    (
        "Company X made $1.5M, Company Y made $2.0M. Which company made more?",
        "Company Y",
    )
]

# ── Runner ────────────────────────────────────────────────────────────────────
total_cos = 0.0
for q, expected in tests:
    got = agent.run(q)
    cos = cosine(got, expected)
    jac = jaccard_ngram(got, expected)
    total_cos += cos
    icon = "PASS" if cos > 0.85 else "WARN"
    print(f"{icon} [{cos:.3f} cos | {jac:.3f} jac]")
    print(f"   Q        : {q}")
    print(f"   Expected : {expected}")
    print(f"   Got      : {got}")
    print()

print(f"Average cosine: {total_cos / len(tests):.3f}")
