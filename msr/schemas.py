from __future__ import annotations

import operator
import uuid
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field


# ── Enumerations ───────────────────────────────────────────────────────────────

class TaskType(str, Enum):
    CODE       = "code"
    MATH       = "math"
    RESEARCH   = "research"
    SUMMARIZE  = "summarize"
    STRUCTURED = "structured"
    FACTCHECK  = "factcheck"   # Grok: real-time info, current events, truth-seeking
    GENERAL    = "general"


class ComplexityLevel(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


class ExecutionMode(str, Enum):
    SINGLE     = "single"      # one clear intent → one specialist
    PARALLEL   = "parallel"    # multiple independent intents → run concurrently
    SEQUENTIAL = "sequential"  # dependent intents → run in DAG order


class Verdict(str, Enum):
    PASS  = "pass"
    FAIL  = "fail"
    RETRY = "retry"


# ── Layer 0: Entry ─────────────────────────────────────────────────────────────

class TaskRequest(BaseModel):
    """Raw user input. Created by the CLI."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    query: str
    context: str | None = None
    max_latency_ms: int = 30_000
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ── Layer 1 (legacy): Single-intent router output ─────────────────────────────
# Kept for backward compatibility with standalone router smoke tests.

class RoutedTask(BaseModel):
    """Legacy single-intent router output. Superseded by RoutedPlan."""
    request: TaskRequest
    task_type: TaskType
    complexity: ComplexityLevel
    routing_rationale: str
    confidence: float = Field(ge=0.0, le=1.0)
    estimated_tokens: int | None = None


# ── Layer 2: Orchestrator output (replaces RoutedTask + ExecutionPlan) ─────────

class SubTask(BaseModel):
    """One atomic unit of work sent to a single specialist."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_type: TaskType
    prompt: str                          # self-contained instruction for the specialist
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)  # per-subtask confidence
    depends_on: list[str] = Field(default_factory=list)      # SubTask IDs (DAG edges)
    parallel_ok: bool = True


class RoutedPlan(BaseModel):
    """Orchestrator output. A fully decomposed, mode-annotated plan.

    Replaces the old RoutedTask + ExecutionPlan two-step.
    The orchestrator detects compound intent in one LLM call and
    returns this directly — no separate planning step needed.
    """
    original_request: str
    is_multi_intent: bool
    execution_mode: ExecutionMode
    subtasks: list[SubTask]
    routing_rationale: str             # explains the decomposition decision


# ── Legacy: ExecutionPlan kept for backward compat with planner smoke tests ────

class ExecutionPlan(BaseModel):
    """Legacy planner output. Superseded by RoutedPlan."""
    routed_task: RoutedTask
    subtasks: list[SubTask]
    is_parallel: bool
    plan_rationale: str


# ── Layer 3: After Specialists ─────────────────────────────────────────────────

class SpecialistOutput(BaseModel):
    """Raw output from one specialist for one SubTask."""
    subtask_id: str
    task_type: TaskType
    model_used: str
    content: str
    structured_data: dict[str, Any] | None = None
    latency_ms: int
    token_usage: dict[str, int] | None = None
    error: str | None = None


# ── Layer 4: After Verifier ────────────────────────────────────────────────────

class VerifiedOutput(BaseModel):
    """Verifier's assessment of one SpecialistOutput."""
    specialist_output: SpecialistOutput
    verdict: Verdict
    score: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    suggested_fix: str | None = None
    retry_count: int = 0


# ── Layer 5: Final ─────────────────────────────────────────────────────────────

class FinalResponse(BaseModel):
    """Synthesizer output. What the user sees."""
    request_id: str
    answer: str
    sources: list[str] = Field(default_factory=list)
    model_trace: list[str] = Field(default_factory=list)
    total_latency_ms: int
    total_tokens: int
    confidence: float = Field(ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)


# ── LangGraph State ────────────────────────────────────────────────────────────

class MSRState(TypedDict, total=False):
    """Mutable state flowing through the LangGraph.

    specialist_outputs and verified_outputs use operator.add as reducer
    so parallel specialist branches fan-in correctly.

    routed_plan replaces the old routed_task + plan two-step fields.
    """
    request: dict
    routed_plan: dict | None           # RoutedPlan — set by orchestrate_node
    specialist_outputs: Annotated[list[dict], operator.add]
    verified_outputs: Annotated[list[dict], operator.add]
    final_response: dict | None
    error: str | None
    retry_subtask_ids: list[str]
