"""End-to-end smoke test: full pipeline — single, parallel, and sequential.
Run: python tests/smoke_e2e.py
"""
import sys
sys.path.insert(0, ".")

import time
from msr.graph.graph import graph
from msr.schemas import FinalResponse, MSRState, TaskRequest

TEST_QUERIES = [
    # (query, label)
    ("What is the derivative of f(x) = x^3?",
     "single / math"),
    ("Write a Python function that reverses a linked list.",
     "single / code"),
    ("Summarise this text: Python is a high-level language known for readability. "
     "Also tell me what 2 + 2 is.",
     "parallel / summarize + math"),
]

print("\n=== End-to-End Smoke Tests ===\n")

for query, label in TEST_QUERIES:
    print(f"[{label}] {query[:70]}...")
    request = TaskRequest(query=query)

    initial: MSRState = {
        "request":            request.model_dump(mode="json"),
        "routed_plan":        None,
        "specialist_outputs": [],
        "verified_outputs":   [],
        "final_response":     None,
        "error":              None,
        "retry_subtask_ids":  [],
    }

    t0 = time.monotonic()
    result = graph.invoke(initial)
    elapsed = int((time.monotonic() - t0) * 1000)

    if result.get("error") and not result.get("final_response"):
        print(f"  FAIL — error: {result['error']}\n")
        continue

    if not result.get("final_response"):
        print(f"  FAIL — no final_response in state\n")
        continue

    fr = FinalResponse.model_validate(result["final_response"])
    print(f"  PASS — {elapsed}ms | tokens={fr.total_tokens} | models={fr.model_trace}")
    print(f"  Answer preview: {fr.answer[:200].strip()}")
    if fr.warnings:
        print(f"  Warnings: {fr.warnings}")
    print()

print("=== Done ===\n")
