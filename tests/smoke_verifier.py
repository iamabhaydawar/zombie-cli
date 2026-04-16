"""Smoke test: Verifier in isolation.
Run: python tests/smoke_verifier.py
"""
import sys
sys.path.insert(0, ".")

from msr.schemas import SpecialistOutput, TaskType, Verdict
from msr.verifier.verifier import verify

GOOD_CODE = """\
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
"""

BAD_MATH = "The derivative of x^3 is 2x."  # Wrong — should be 3x^2

ERRORED_OUTPUT = ""

print("\n=== Verifier Smoke Tests ===\n")

# Test 1: Good code — should PASS
out1 = SpecialistOutput(subtask_id="t1", task_type=TaskType.CODE, model_used="gpt-4o", content=GOOD_CODE, latency_ms=1000)
v1 = verify(out1)
print(f"  [{'PASS' if v1.verdict == Verdict.PASS else 'UNEXPECTED'}] Good code → verdict={v1.verdict.value} score={v1.score:.2f}")

# Test 2: Wrong math — should FAIL or RETRY
out2 = SpecialistOutput(subtask_id="t2", task_type=TaskType.MATH, model_used="deepseek", content=BAD_MATH, latency_ms=800)
v2 = verify(out2)
print(f"  [{'PASS' if v2.verdict in (Verdict.FAIL, Verdict.RETRY) else 'UNEXPECTED'}] Wrong math → verdict={v2.verdict.value} score={v2.score:.2f}")
if v2.issues:
    print(f"         Issues: {v2.issues[0]}")

# Test 3: Errored specialist output — should FAIL immediately
out3 = SpecialistOutput(subtask_id="t3", task_type=TaskType.CODE, model_used="gpt-4o", content=ERRORED_OUTPUT, latency_ms=0, error="API timeout")
v3 = verify(out3)
print(f"  [{'PASS' if v3.verdict == Verdict.FAIL else 'UNEXPECTED'}] Errored output → verdict={v3.verdict.value} (no LLM call needed)")

print("\n=== Done ===\n")
