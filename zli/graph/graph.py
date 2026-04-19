from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from zli.graph.edges import after_orchestrate, after_verify
from zli.graph.nodes import (
    dispatch_node,
    handle_error_node,
    orchestrate_node,
    specialist_code_node,
    specialist_factcheck_node,
    specialist_general_node,
    specialist_math_node,
    specialist_research_node,
    specialist_structured_node,
    specialist_summarize_node,
    synthesize_node,
    verify_node,
)
from zli.schemas import MSRState


def build_graph():
    """Assemble and compile the ZLI LangGraph StateGraph.

    Architecture:
        START → orchestrate → dispatch → specialist_X (parallel/sequential)
                                              ↓
                                           verify ─── retry/sequential ──► dispatch
                                              ↓
                                         synthesize → END

    The orchestrate node replaces the old route + plan two-step.
    dispatch is DAG-aware: fires only ready subtasks each pass.
    verify loops back to dispatch for retries and sequential next-waves.
    """
    g = StateGraph(MSRState)

    # ── Core orchestration nodes ───────────────────────────────────────────────
    g.add_node("orchestrate",  orchestrate_node)
    g.add_node("dispatch",     dispatch_node)
    g.add_node("verify",       verify_node)
    g.add_node("synthesize",   synthesize_node)
    g.add_node("handle_error", handle_error_node)

    # ── Specialist leaf nodes (one per task type) ──────────────────────────────
    g.add_node("specialist_code",        specialist_code_node)
    g.add_node("specialist_math",        specialist_math_node)
    g.add_node("specialist_research",    specialist_research_node)
    g.add_node("specialist_summarize",   specialist_summarize_node)
    g.add_node("specialist_structured",  specialist_structured_node)
    g.add_node("specialist_factcheck",   specialist_factcheck_node)
    g.add_node("specialist_general",     specialist_general_node)

    # ── Edges ──────────────────────────────────────────────────────────────────
    g.add_edge(START, "orchestrate")

    g.add_conditional_edges("orchestrate", after_orchestrate, {
        "dispatch":     "dispatch",
        "handle_error": "handle_error",
    })

    # dispatch → specialist fan-out via Send() in dispatch_node.
    # Each specialist fans back to verify via operator.add reducer.
    for specialist_node_name in [
        "specialist_code",
        "specialist_math",
        "specialist_research",
        "specialist_summarize",
        "specialist_structured",
        "specialist_factcheck",
        "specialist_general",
    ]:
        g.add_edge(specialist_node_name, "verify")

    g.add_conditional_edges("verify", after_verify, {
        "dispatch":   "dispatch",   # retry or sequential next-wave
        "synthesize": "synthesize",
    })

    g.add_edge("synthesize",   END)
    g.add_edge("handle_error", END)

    return g.compile()


# Singleton — import this wherever you need the compiled graph
graph = build_graph()
