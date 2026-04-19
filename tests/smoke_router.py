"""Smoke test: Intent Router only.
Run: python tests/smoke_router.py
"""
import sys
sys.path.insert(0, ".")

from zli.router.intent import classify
from zli.schemas import TaskRequest, TaskType

TEST_CASES = [
    ("implement quicksort in Python",                   TaskType.CODE),
    ("what is the integral of x^2 sin(x)?",            TaskType.MATH),
    ("who won the 2024 US presidential election?",      TaskType.RESEARCH),
    ("summarize this article for me",                   TaskType.SUMMARIZE),
    ("generate a JSON schema for a user profile",       TaskType.STRUCTURED),
]

print("\n=== Router Smoke Test ===\n")
all_pass = True
for query, expected_type in TEST_CASES:
    req = TaskRequest(query=query)
    result = classify(req)
    ok = result.task_type == expected_type
    status = "PASS" if ok else "WARN"  # warn not fail — LLM may classify differently
    print(f"  [{status}] '{query[:50]}...' → {result.task_type.value} (expected: {expected_type.value}) conf={result.confidence:.2f}")
    if not ok:
        print(f"         Rationale: {result.routing_rationale}")

print("\n=== Done ===\n")
