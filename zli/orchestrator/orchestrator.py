from __future__ import annotations

import json

import litellm

from zli.config import settings
from zli.schemas import ExecutionMode, RoutedPlan, SubTask, TaskRequest, TaskType

# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM = """\
You are an intelligent orchestrator for a multi-specialist AI system.
Your job is to analyse a user request, detect whether it contains one or multiple intents,
decompose it into the minimal set of atomic subtasks, assign each to the right specialist,
and decide the correct execution mode.

Specialists available:
- code:       writing, debugging, refactoring, explaining code
- math:       arithmetic, algebra, calculus, proofs, formal reasoning
- research:   deep questions needing citations, academic depth, background knowledge
- summarize:  condensing documents, transcripts, articles (user has provided source text)
- structured: producing JSON, SQL, YAML, plans, schemas, tables
- factcheck:  verifying claims, current events, real-time data, truth-seeking
- general:    anything that does not clearly fit the above

Execution mode rules — apply these strictly:
- "single":     exactly one intent detected → exactly one subtask
- "parallel":   multiple INDEPENDENT intents (neither depends on the other's output) → run simultaneously
- "sequential": multiple DEPENDENT intents (task B needs task A's output) → run in DAG order

Examples:
  "summarise this CSV and fix my Python bug"
    → is_multi_intent=true, mode="parallel", two subtasks (summarize + code)
    → parallel because CSV summary does not need the bug fix result, and vice versa

  "clean this CSV then summarise it"
    → is_multi_intent=true, mode="sequential", two subtasks
    → st-1: code (clean the CSV), st-2: summarize (depends_on: ["st-1"])
    → sequential because the summary needs the cleaned output

  "what is 2+2"
    → is_multi_intent=false, mode="single", one subtask

Ambiguity rule:
  If the request is genuinely ambiguous about whether tasks are dependent,
  model them as sequential (conservative) and explain in routing_rationale.

Subtask prompt rules:
- Write each subtask prompt as a direct, self-contained instruction to the specialist.
- The specialist will NOT see the original query — the prompt must contain all context needed.
- For sequential subtasks that depend on a previous step, write:
  "Using the output from the previous step, [instruction]..."
  The system will automatically inject the actual prior output at runtime.
- Per-subtask confidence reflects how clearly this intent was expressed (0.0–1.0).

Return ONLY a JSON object — no markdown, no explanation outside the object:
{
  "is_multi_intent": boolean,
  "execution_mode": "single" | "parallel" | "sequential",
  "routing_rationale": "one sentence explaining the decomposition and mode choice",
  "subtasks": [
    {
      "id": "st-1",
      "task_type": "code" | "math" | "research" | "summarize" | "structured" | "factcheck" | "general",
      "prompt": "complete, self-contained instruction for this specialist",
      "confidence": 0.0–1.0,
      "depends_on": [],
      "parallel_ok": true
    }
  ]
}
"""

# ── Public API ─────────────────────────────────────────────────────────────────

def orchestrate(request: TaskRequest) -> RoutedPlan:
    """Detect intent(s), decompose, and return a fully-formed RoutedPlan.

    Replaces the old router → planner two-step with a single LLM call.
    Uses Claude Sonnet (planner_model) for high-quality decomposition.
    """
    user_content = request.query
    if request.context:
        user_content += f"\n\n[User has provided additional context — {len(request.context)} chars]\n{request.context[:3000]}"

    # Try primary model (Claude Sonnet), fall back to GPT-4o on error
    try:
        response = litellm.completion(
            model=settings.orchestrator_primary,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": user_content},
            ],
            timeout=settings.zli_default_timeout_s,
            temperature=0.0,
        )
    except Exception:
        response = litellm.completion(
            model=settings.orchestrator_fallback,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": user_content},
            ],
            timeout=settings.zli_default_timeout_s,
            temperature=0.0,
        )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if the model wraps the JSON anyway
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
            confidence=float(st.get("confidence", 1.0)),
            depends_on=st.get("depends_on", []),
            parallel_ok=st.get("parallel_ok", True),
        )
        for st in data["subtasks"]
    ]

    return RoutedPlan(
        original_request=request.query,
        is_multi_intent=bool(data["is_multi_intent"]),
        execution_mode=ExecutionMode(data["execution_mode"]),
        subtasks=subtasks,
        routing_rationale=data.get("routing_rationale", ""),
    )
