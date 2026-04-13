from __future__ import annotations

import time

import litellm

from msr.config import settings
from msr.schemas import FinalResponse, TaskRequest, Verdict, VerifiedOutput

_SYSTEM = """\
You are the final synthesis layer of a multi-specialist AI pipeline.
You receive one or more verified outputs from specialist models and must produce a single, clean, complete answer for the user.

Your job:
- Merge all specialist outputs into one coherent, well-structured response.
- Preserve technical accuracy from the specialist outputs — do not simplify to the point of incorrectness.
- Resolve any contradictions between specialists by preferring the higher-scored output or by surfacing the conflict.
- Remove redundancy, internal metadata, and specialist formatting artifacts (e.g., "Problem Type:" headers unless they aid the user).
- Match the format to the task: code answers should include code blocks, math answers should use LaTeX, research answers should list sources.
- Do not add new information that was not in the specialist outputs.
- Do not editorialize or pad the answer with praise or filler phrases.

Quality bar:
- The answer must be self-contained — the user should not need to read specialist outputs to understand it.
- If any specialist issued a warning or low-confidence result, surface it clearly at the end.
- If the answer is code, it must be complete and runnable unless a snippet was explicitly requested.
- If sources were found by the research specialist, include them at the end under a Sources section.

Tone:
- Professional, precise, and direct.
- Match the register of the original question (technical if the question is technical).
"""


def synthesize(
    verified_outputs: list[VerifiedOutput],
    request: TaskRequest,
) -> FinalResponse:
    """Merge verified specialist outputs into a single FinalResponse."""
    t0 = time.monotonic()

    # Build context from all verified outputs
    context_parts = []
    sources: list[str] = []
    model_trace: list[str] = []
    total_tokens = 0
    warnings: list[str] = []
    scores: list[float] = []

    for vo in verified_outputs:
        so = vo.specialist_output
        model_trace.append(so.model_used)
        scores.append(vo.score)

        if so.token_usage:
            total_tokens += so.token_usage.get("total_tokens", 0)

        if vo.verdict == Verdict.FAIL:
            warnings.append(
                f"{so.task_type.value} specialist ({so.model_used}) failed verification "
                f"(score: {vo.score:.2f}). Issues: {'; '.join(vo.issues)}"
            )

        context_parts.append(
            f"[{so.task_type.value.upper()} SPECIALIST — {so.model_used} — score {vo.score:.2f}]\n"
            f"{so.content}"
        )

    user_content = (
        f"Original question: {request.query}\n\n"
        + "\n\n---\n\n".join(context_parts)
    )

    try:
        response = litellm.completion(
            model=settings.synthesizer_model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_content},
            ],
            timeout=settings.msr_default_timeout_s,
        )
    except Exception:
        response = litellm.completion(
            model=settings.synthesizer_fallback,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_content},
            ],
            timeout=settings.msr_default_timeout_s,
        )

    answer = response.choices[0].message.content or ""
    if response.usage:
        total_tokens += response.usage.total_tokens or 0

    # Add synthesizer model to trace (deduped)
    if settings.synthesizer_model not in model_trace:
        model_trace.append(settings.synthesizer_model)

    # Dedup trace while preserving order
    seen: set[str] = set()
    deduped_trace: list[str] = []
    for m in model_trace:
        if m not in seen:
            seen.add(m)
            deduped_trace.append(m)

    avg_confidence = sum(scores) / len(scores) if scores else 0.0
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    return FinalResponse(
        request_id=request.id,
        answer=answer,
        sources=sources,
        model_trace=deduped_trace,
        total_latency_ms=elapsed_ms,
        total_tokens=total_tokens,
        confidence=round(avg_confidence, 3),
        warnings=warnings,
    )
