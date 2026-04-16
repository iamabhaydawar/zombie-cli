"""Smoke test: Planner in isolation.
Run: python tests/smoke_planner.py
"""
import sys
sys.path.insert(0, ".")

from msr.planner.planner import plan
from msr.schemas import ComplexityLevel, RoutedTask, TaskRequest, TaskType

TEST_CASES = [
    # (query, task_type, complexity, expected_min_subtasks)
    ("What is 2+2?",                                                      TaskType.MATH,    ComplexityLevel.LOW,    1),
    ("Write a REST API in FastAPI with JWT auth and full test suite",      TaskType.CODE,    ComplexityLevel.HIGH,   2),
    ("Compare the economic policies of the US and EU in 2024",            TaskType.RESEARCH, ComplexityLevel.HIGH,  2),
]

print("\n=== Planner Smoke Tests ===\n")
for query, task_type, complexity, min_subtasks in TEST_CASES:
    routed = RoutedTask(
        request=TaskRequest(query=query),
        task_type=task_type,
        complexity=complexity,
        routing_rationale="test fixture",
        confidence=0.9,
    )
    result = plan(routed)
    ok = len(result.subtasks) >= min_subtasks
    status = "PASS" if ok else "WARN"
    print(f"  [{status}] '{query[:55]}...'")
    print(f"         subtasks={len(result.subtasks)} parallel={result.is_parallel}")
    for i, st in enumerate(result.subtasks):
        print(f"         {i+1}. [{st.task_type.value}] {st.prompt[:70]}")
    print()

print("=== Done ===\n")
