from __future__ import annotations

import json
import time

import litellm

from msr.config import settings
from msr.schemas import ComplexityLevel, ExecutionPlan, RoutedTask, SubTask, TaskType

_SYSTEM = """\
You are a task planner for a multi-specialist AI system.
Given a routed task, break it into the minimal set of atomic subtasks.
Each subtask will be handled by a single specialist: code, math, research, summarize, structured, factcheck, or general.

Rules:
- For LOW complexity: return exactly 1 subtask — do not over-decompose simple tasks.
- For MEDIUM complexity: 2-3 subtasks maximum.
- For HIGH complexity: up to 5 subtasks. Identify which can run in parallel.
- A subtask prompt must be self-contained — the specialist will not see the original query.
- Write each subtask prompt as a direct instruction to the specialist (imperative mood).
- Set depends_on to the IDs of subtasks that must complete before this one starts.
- Set parallel_ok to true if this subtask does not depend on another's output.

Return ONLY a JSON object with this shape:
{
  "subtasks": [
    {
      "id": "short unique string like 'st-1'",
      "task_type": one of ["code", "math", "research", "summarize", "structured", "factcheck", "general"],
      "prompt": "complete, self-contained instruction for the specialist",
      "depends_on": [],
      "parallel_ok": true
    }
  ],
  "is_parallel": true or false (true if ANY subtask has parallel_ok=true and no depends_on),
  "plan_rationale": "one sentence explaining the decomposition strategy"
}

No markdown, no explanation outside the JSON object.
"""


def plan(routed: RoutedTask) -> ExecutionPlan:
    """Decompose a RoutedTask into an ExecutionPlan of SubTasks."""

    # Low-complexity tasks get a fast path — no LLM planning needed
    if routed.complexity == ComplexityLevel.LOW:
        subtask = SubTask(
            id="st-1",
            task_type=routed.task_type,
            prompt=routed.request.query
            + (f"\n\nContext:\n{routed.request.context}" if routed.request.context else ""),
            depends_on=[],
            parallel_ok=True,
        )
        return ExecutionPlan(
            routed_task=routed,
            subtasks=[subtask],
            is_parallel=False,
            plan_rationale="Single subtask — low complexity fast path.",
        )

    user_content = (
        f"Task type: {routed.task_type.value}\n"
        f"Complexity: {routed.complexity.value}\n"
        f"Query: {routed.request.query}"
    )
    if routed.request.context:
        user_content += f"\n\nContext:\n{routed.request.context[:2000]}"

    response = litellm.completion(
        model=settings.planner_model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_content},
        ],
        timeout=settings.msr_default_timeout_s,
        temperature=0.0,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    data = json.loads(raw)

    subtasks = [
        SubTask(
            id=st["id"],
            task_type=TaskType(st["task_type"]),
            prompt=st["prompt"],
            depends_on=st.get("depends_on", []),
            parallel_ok=st.get("parallel_ok", True),
        )
        for st in data["subtasks"]
    ]

    return ExecutionPlan(
        routed_task=routed,
        subtasks=subtasks,
        is_parallel=data.get("is_parallel", False),
        plan_rationale=data.get("plan_rationale", ""),
    )
