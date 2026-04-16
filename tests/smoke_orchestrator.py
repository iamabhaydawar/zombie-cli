"""Smoke test: Orchestrator in isolation — all three execution modes.
Run: python tests/smoke_orchestrator.py
"""
import sys
sys.path.insert(0, ".")

from msr.orchestrator.orchestrator import orchestrate
from msr.schemas import ExecutionMode, TaskRequest, TaskType

TEST_CASES = [
    # (query, expected_mode, min_subtasks, description)
    (
        "What is the derivative of x^3?",
        ExecutionMode.SINGLE,
        1,
        "single intent → single subtask",
    ),
    (
        "Summarise this CSV file and also fix my Python IndexError bug",
        ExecutionMode.PARALLEL,
        2,
        "two independent intents → parallel",
    ),
    (
        "Clean this CSV by removing null rows, then summarise the cleaned data",
        ExecutionMode.SEQUENTIAL,
        2,
        "dependent intents → sequential with depends_on",
    ),
    (
        "Write a Python function to fetch JSON from a URL, then write unit tests for it",
        ExecutionMode.SEQUENTIAL,
        2,
        "write-then-test → sequential",
    ),
]

print("\n=== Orchestrator Smoke Tests ===\n")
all_pass = True

for query, expected_mode, min_subtasks, description in TEST_CASES:
    req = TaskRequest(query=query)
    plan = orchestrate(req)

    mode_ok      = plan.execution_mode == expected_mode
    subtask_ok   = len(plan.subtasks) >= min_subtasks
    deps_ok      = True

    # For sequential mode, at least one subtask should have depends_on
    if expected_mode == ExecutionMode.SEQUENTIAL:
        deps_ok = any(len(st.depends_on) > 0 for st in plan.subtasks)

    ok = mode_ok and subtask_ok and deps_ok
    status = "PASS" if ok else "WARN"
    if not ok:
        all_pass = False

    print(f"  [{status}] {description}")
    print(f"         mode={plan.execution_mode.value} (expected={expected_mode.value})  "
          f"subtasks={len(plan.subtasks)}")
    print(f"         rationale: {plan.routing_rationale}")
    for st in plan.subtasks:
        dep_str = f" → depends_on={st.depends_on}" if st.depends_on else ""
        print(f"         • [{st.task_type.value}] conf={st.confidence:.2f}{dep_str}  {st.prompt[:70]}")

    if not mode_ok:
        print(f"         ⚠ expected mode={expected_mode.value}, got {plan.execution_mode.value}")
    if not deps_ok:
        print(f"         ⚠ expected at least one subtask with depends_on for sequential mode")
    print()

print("=== Done ===")
print(f"Overall: {'ALL PASSED' if all_pass else 'SOME UNEXPECTED — review WARN entries'}\n")
