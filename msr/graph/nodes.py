from __future__ import annotations

from typing import Any

from langgraph.types import Send

from msr.orchestrator.orchestrator import orchestrate
from msr.schemas import (
    ExecutionMode,
    FinalResponse,
    MSRState,
    RoutedPlan,
    SpecialistOutput,
    SubTask,
    TaskRequest,
    TaskType,
    VerifiedOutput,
)
from msr.specialists.code import CodeSpecialist
from msr.specialists.factcheck import FactCheckSpecialist
from msr.specialists.math import MathSpecialist
from msr.specialists.research import ResearchSpecialist
from msr.specialists.structured import StructuredSpecialist
from msr.specialists.summarize import SummarizeSpecialist
from msr.synthesizer.synthesizer import synthesize
from msr.verifier.verifier import verify

# ── Specialist registry ────────────────────────────────────────────────────────

_SPECIALISTS: dict[str, Any] = {
    TaskType.CODE.value:       CodeSpecialist(),
    TaskType.MATH.value:       MathSpecialist(),
    TaskType.RESEARCH.value:   ResearchSpecialist(),
    TaskType.SUMMARIZE.value:  SummarizeSpecialist(),
    TaskType.STRUCTURED.value: StructuredSpecialist(),
    TaskType.FACTCHECK.value:  FactCheckSpecialist(),
}


# ── Orchestrate node (replaces route + plan) ──────────────────────────────────

def orchestrate_node(state: MSRState) -> dict:
    """Single-step orchestration: detect intent(s), decompose, set execution mode.

    Replaces the old route_node + plan_node two-step.
    Returns a RoutedPlan that drives everything downstream.
    """
    try:
        request = TaskRequest.model_validate(state["request"])
        routed_plan = orchestrate(request)
        return {"routed_plan": routed_plan.model_dump()}
    except Exception as exc:
        return {"error": f"Orchestrator failed: {exc}"}


# ── Dispatch node — DAG-aware fan-out ─────────────────────────────────────────

def dispatch_node(state: MSRState) -> list[Send]:
    """Fan-out: spawn one Send per ready subtask.

    'Ready' means: all entries in depends_on are already in verified_outputs.

    For parallel mode: all subtasks are ready immediately (no deps).
    For sequential mode: only fire the next wave whose deps are satisfied.
    For retry mode: only re-fire the subtasks in retry_subtask_ids.

    For sequential subtasks that DO have deps, the prior outputs are injected
    into the prompt so the specialist has access to upstream results.
    """
    plan = RoutedPlan.model_validate(state["routed_plan"])
    retry_ids: set[str] = set(state.get("retry_subtask_ids") or [])

    # Build a map of subtask_id → verified content for dep injection
    completed: dict[str, str] = {}
    for v in state.get("verified_outputs", []):
        vo = VerifiedOutput.model_validate(v)
        completed[vo.specialist_output.subtask_id] = vo.specialist_output.content

    sends: list[Send] = []

    for subtask in plan.subtasks:
        # If we are in retry mode, only re-dispatch requested subtasks
        if retry_ids and subtask.id not in retry_ids:
            continue

        # Skip subtasks already completed (in normal DAG-advance mode)
        if not retry_ids and subtask.id in completed:
            continue

        # Only dispatch if all dependencies are satisfied
        if not all(dep in completed for dep in subtask.depends_on):
            continue

        # Inject prior outputs into the prompt for sequential subtasks
        enriched_prompt = subtask.prompt
        if subtask.depends_on:
            dep_context = "\n\n".join(
                f"[Output from step {dep_id}]\n{completed[dep_id]}"
                for dep_id in subtask.depends_on
                if dep_id in completed
            )
            if dep_context:
                enriched_prompt = (
                    f"Previous step output(s) — use this as your input:\n\n"
                    f"{dep_context}\n\n"
                    f"---\n\n"
                    f"Your task: {subtask.prompt}"
                )

        enriched_subtask = subtask.model_copy(update={"prompt": enriched_prompt})
        node_name = f"specialist_{subtask.task_type.value}"
        sends.append(Send(node_name, {"subtask": enriched_subtask.model_dump()}))

    return sends


# ── Specialist nodes — factory pattern ────────────────────────────────────────

def _make_specialist_node(task_type_value: str):
    """Factory: returns a node function bound to the given task type specialist."""
    def specialist_node(state: dict) -> dict:
        subtask = SubTask.model_validate(state["subtask"])
        specialist = _SPECIALISTS.get(task_type_value)
        if specialist is None:
            specialist = _SPECIALISTS[TaskType.GENERAL.value]
        output: SpecialistOutput = specialist.run(subtask)
        return {"specialist_outputs": [output.model_dump()]}
    specialist_node.__name__ = f"specialist_{task_type_value}"
    return specialist_node


specialist_code_node        = _make_specialist_node(TaskType.CODE.value)
specialist_math_node        = _make_specialist_node(TaskType.MATH.value)
specialist_research_node    = _make_specialist_node(TaskType.RESEARCH.value)
specialist_summarize_node   = _make_specialist_node(TaskType.SUMMARIZE.value)
specialist_structured_node  = _make_specialist_node(TaskType.STRUCTURED.value)
specialist_factcheck_node   = _make_specialist_node(TaskType.FACTCHECK.value)
specialist_general_node     = _make_specialist_node(TaskType.GENERAL.value)


# ── Verify node ────────────────────────────────────────────────────────────────

def verify_node(state: MSRState) -> dict:
    """Verify all specialist outputs. Populate retry_subtask_ids for bad outputs."""
    raw_outputs = state.get("specialist_outputs", [])
    verified: list[dict] = []
    retry_ids: list[str] = []

    # Build retry-count map from previous verification passes
    prev_verified = state.get("verified_outputs", [])
    retry_count_map: dict[str, int] = {}
    for pv in prev_verified:
        pv_obj = VerifiedOutput.model_validate(pv)
        retry_count_map[pv_obj.specialist_output.subtask_id] = pv_obj.retry_count

    for raw in raw_outputs:
        output = SpecialistOutput.model_validate(raw)
        prev_retries = retry_count_map.get(output.subtask_id, 0)
        vo = verify(output, retry_count=prev_retries)
        verified.append(vo.model_dump())

        from msr.schemas import Verdict
        if vo.verdict == Verdict.RETRY:
            retry_ids.append(output.subtask_id)

    return {
        "verified_outputs": verified,
        "retry_subtask_ids": retry_ids,
    }


# ── Synthesize node ────────────────────────────────────────────────────────────

def synthesize_node(state: MSRState) -> dict:
    """Merge all verified outputs into the final response."""
    request = TaskRequest.model_validate(state["request"])
    raw_verified = state.get("verified_outputs", [])
    verified = [VerifiedOutput.model_validate(v) for v in raw_verified]
    final = synthesize(verified, request)
    return {"final_response": final.model_dump(mode="json")}


# ── Error handler ──────────────────────────────────────────────────────────────

def handle_error_node(state: MSRState) -> dict:
    """Convert a pipeline error into a graceful FinalResponse."""
    request = TaskRequest.model_validate(state["request"])
    error_msg = state.get("error", "An unknown error occurred.")
    final = FinalResponse(
        request_id=request.id,
        answer=f"Sorry, the pipeline encountered an error:\n\n{error_msg}",
        total_latency_ms=0,
        total_tokens=0,
        confidence=0.0,
        warnings=[error_msg],
    )
    return {"final_response": final.model_dump(mode="json")}
