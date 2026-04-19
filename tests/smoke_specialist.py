"""Smoke test: one specialist in isolation.
Run: python tests/smoke_specialist.py
     python tests/smoke_specialist.py --specialist math
"""
import sys
sys.path.insert(0, ".")

import argparse
from zli.schemas import SubTask, TaskType
from zli.specialists.code import CodeSpecialist
from zli.specialists.math import MathSpecialist
from zli.specialists.research import ResearchSpecialist
from zli.specialists.summarize import SummarizeSpecialist
from zli.specialists.structured import StructuredSpecialist

SPECIALISTS = {
    "code":       (CodeSpecialist(),       SubTask(task_type=TaskType.CODE,       prompt="Write a Python function that checks if a number is prime.")),
    "math":       (MathSpecialist(),       SubTask(task_type=TaskType.MATH,       prompt="Find the derivative of f(x) = x^3 * sin(x).")),
    "research":   (ResearchSpecialist(),   SubTask(task_type=TaskType.RESEARCH,   prompt="What is the capital city of Australia and what is its population?")),
    "summarize":  (SummarizeSpecialist(),  SubTask(task_type=TaskType.SUMMARIZE,  prompt="Summarize this text: Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected.")),
    "structured": (StructuredSpecialist(), SubTask(task_type=TaskType.STRUCTURED, prompt="Generate a JSON schema for a user profile with fields: id (uuid), name (string), email (string), age (integer, optional), created_at (datetime).")),
}

parser = argparse.ArgumentParser()
parser.add_argument("--specialist", default="all", choices=list(SPECIALISTS.keys()) + ["all"])
args = parser.parse_args()

to_run = SPECIALISTS if args.specialist == "all" else {args.specialist: SPECIALISTS[args.specialist]}

print("\n=== Specialist Smoke Tests ===\n")
for name, (specialist, subtask) in to_run.items():
    print(f"Testing [{name}]...")
    output = specialist.run(subtask)
    if output.error:
        print(f"  FAIL — error: {output.error}\n")
    else:
        print(f"  PASS — model={output.model_used} latency={output.latency_ms}ms tokens={output.token_usage}")
        print(f"  Output preview: {output.content[:200].strip()}\n")

print("=== Done ===\n")
