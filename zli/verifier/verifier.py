from __future__ import annotations

import json

import litellm

from zli.config import settings
from zli.schemas import SpecialistOutput, TaskType, Verdict, VerifiedOutput

_SYSTEM = """\
You are a strict output verifier for an AI pipeline. A specialist model has produced an answer.
Your job is to score it honestly and catch errors before they reach the user.

You are a cross-model checker — you may be reviewing output from a different model family.
Be objective. Do not give benefit of the doubt. Surface real problems.

Evaluation criteria by task type:
- code:       Does it run? Is it correct? Are there syntax errors, logic bugs, or missing edge cases?
- math:       Are all steps shown? Is the final answer correct? Are there arithmetic or reasoning errors?
- research:   Are claims supported? Are sources plausible? Is anything likely fabricated?
- summarize:  Does it accurately reflect the source? Are key points missing? Is anything added that wasn't in the source?
- structured: Is the output syntactically valid and parsable? Does it match the requested schema or format?
- general:    Is the answer complete, accurate, and relevant to the question?

Return ONLY a JSON object:
{
  "verdict": one of ["pass", "fail", "retry"],
  "score": float 0.0-1.0 (0=completely wrong, 1=perfect),
  "issues": ["list of specific problems found — be precise, not vague"],
  "suggested_fix": "one sentence describing what the specialist should do differently on retry, or null"
}

Verdict guide:
- pass:  score >= 0.75 and no critical errors
- retry: score 0.4-0.74 OR a fixable critical error exists (wrong approach, missing key part)
- fail:  score < 0.4 OR unfixable (hallucinated sources, completely wrong answer, malformed output)

No markdown, no explanation outside the JSON.
"""


def verify(output: SpecialistOutput, retry_count: int = 0) -> VerifiedOutput:
    """Score a SpecialistOutput and return a VerifiedOutput with verdict."""

    # If the specialist itself errored, fail immediately without calling verifier
    if output.error:
        return VerifiedOutput(
            specialist_output=output,
            verdict=Verdict.FAIL,
            score=0.0,
            issues=[f"Specialist error: {output.error}"],
            retry_count=retry_count,
        )

    user_content = (
        f"Task type: {output.task_type.value}\n"
        f"Model used: {output.model_used}\n\n"
        f"--- Specialist Output ---\n{output.content}"
    )
    if output.structured_data:
        user_content += f"\n\n--- Parsed structured data ---\n{json.dumps(output.structured_data, indent=2)}"

    response = litellm.completion(
        model=settings.verifier_model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_content},
        ],
        timeout=settings.zli_default_timeout_s,
        temperature=0.0,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    data = json.loads(raw)
    verdict = Verdict(data["verdict"])

    # Enforce circuit breaker: after 2 retries, downgrade RETRY to FAIL
    if verdict == Verdict.RETRY and retry_count >= settings.zli_max_retries:
        verdict = Verdict.FAIL

    return VerifiedOutput(
        specialist_output=output,
        verdict=verdict,
        score=float(data["score"]),
        issues=data.get("issues", []),
        suggested_fix=data.get("suggested_fix"),
        retry_count=retry_count,
    )
