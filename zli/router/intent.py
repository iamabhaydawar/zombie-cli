from __future__ import annotations

import json
import time

import litellm

from zli.config import settings
from zli.schemas import ComplexityLevel, RoutedTask, TaskRequest, TaskType

_SYSTEM = """\
You are an intent router for a multi-model AI system.
Classify the user's request and return ONLY a JSON object with these fields:
{
  "task_type": one of ["code", "math", "research", "summarize", "structured", "factcheck", "general"],
  "complexity": one of ["low", "medium", "high"],
  "routing_rationale": "one sentence explaining the classification",
  "confidence": float between 0.0 and 1.0,
  "estimated_tokens": integer estimate of tokens the specialist will need
}

Classification guide:
- code:       writing, debugging, refactoring, explaining code
- math:       arithmetic, algebra, calculus, proofs, formal reasoning
- research:   deep questions needing citations, academic sources, background knowledge
- summarize:  condensing documents, transcripts, articles — input text provided
- structured: producing JSON, SQL, YAML, plans, schemas, tables
- factcheck:  verifying a specific claim, asking about CURRENT events or RECENT news, checking if something is true/false, questions about what is happening RIGHT NOW, real-time data (prices, scores, polls), or requests that benefit from live X/Twitter information
- general:    anything that does not clearly fit the above

Key distinction — research vs factcheck:
- research = "explain the history of X", "what caused Y", "summarize the literature on Z" → needs depth and citations
- factcheck = "is it true that X?", "what happened with Y yesterday?", "what is the current status of Z?" → needs real-time accuracy and a verdict

Complexity:
- low: single-step, one model, fast answer
- medium: may need planning or multi-step reasoning
- high: requires parallel subtasks or deep multi-step work

Return ONLY the JSON object. No markdown, no explanation.
"""


def classify(request: TaskRequest) -> RoutedTask:
    """Classify a TaskRequest into a RoutedTask using the router model."""
    user_content = request.query
    if request.context:
        user_content += f"\n\n[Context provided: {len(request.context)} chars]"

    t0 = time.monotonic()
    response = litellm.completion(
        model=settings.router_model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_content},
        ],
        timeout=settings.zli_default_timeout_s,
        temperature=0.0,
    )
    elapsed_ms = int((time.monotonic() - t0) * 1000)  # noqa: F841 — kept for future logging

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if the model wraps the JSON anyway
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    data = json.loads(raw)

    return RoutedTask(
        request=request,
        task_type=TaskType(data["task_type"]),
        complexity=ComplexityLevel(data["complexity"]),
        routing_rationale=data["routing_rationale"],
        confidence=float(data["confidence"]),
        estimated_tokens=data.get("estimated_tokens"),
    )
