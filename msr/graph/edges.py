from __future__ import annotations

from msr.schemas import ExecutionMode, MSRState, RoutedPlan, VerifiedOutput


def after_orchestrate(state: MSRState) -> str:
    """Route to error handler if orchestration failed, else proceed to dispatch."""
    if state.get("error"):
        return "handle_error"
    return "dispatch"


def after_verify(state: MSRState) -> str:
    """Decide what to do after a verification pass.

    Three possible outcomes, checked in order:
    1. Retries needed → back to dispatch (re-run bad subtasks)
    2. Sequential mode with more ready subtasks → back to dispatch (next DAG wave)
    3. All done → synthesize
    """
    # ── 1. Retry check ────────────────────────────────────────────────────────
    retry_ids = state.get("retry_subtask_ids") or []
    if retry_ids:
        return "dispatch"

    # ── 2. Sequential DAG advance ─────────────────────────────────────────────
    # Check if there are subtasks whose deps are now satisfied by the latest
    # verification pass. If so, loop back to dispatch to fire the next wave.
    routed_plan_dict = state.get("routed_plan")
    if routed_plan_dict:
        plan = RoutedPlan.model_validate(routed_plan_dict)

        if plan.execution_mode == ExecutionMode.SEQUENTIAL:
            # Build the set of completed subtask IDs
            completed_ids: set[str] = {
                VerifiedOutput.model_validate(v).specialist_output.subtask_id
                for v in state.get("verified_outputs", [])
            }
            all_ids = {st.id for st in plan.subtasks}
            remaining = all_ids - completed_ids

            # Check if any remaining subtask is now ready (all deps completed)
            for subtask in plan.subtasks:
                if subtask.id in remaining:
                    if all(dep in completed_ids for dep in subtask.depends_on):
                        return "dispatch"  # Next DAG wave is ready

    # ── 3. All done ───────────────────────────────────────────────────────────
    return "synthesize"
