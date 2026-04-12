from __future__ import annotations

import time
from abc import ABC, abstractmethod

import litellm
from litellm.exceptions import APIConnectionError, APIError, Timeout

from msr.config import settings
from msr.schemas import SpecialistOutput, SubTask, TaskType


class BaseSpecialist(ABC):
    """Abstract base class for all specialists.

    Subclasses must set `task_type` and `system_prompt`, and may override
    `_build_messages` for custom message construction.
    """

    task_type: TaskType
    system_prompt: str

    def _get_models(self) -> tuple[str, str]:
        entry = settings.model_map.get(self.task_type.value, {})
        primary = entry.get("primary", "claude-sonnet-4-6")
        fallback = entry.get("fallback", "gpt-4o")
        return primary, fallback

    def _build_messages(self, subtask: SubTask) -> list[dict]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": subtask.prompt},
        ]

    def _call(self, model: str, messages: list[dict], **kwargs) -> tuple[str, dict[str, int]]:
        """Call litellm and return (content, token_usage)."""
        response = litellm.completion(
            model=model,
            messages=messages,
            timeout=settings.msr_default_timeout_s,
            **kwargs,
        )
        content = response.choices[0].message.content or ""
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }
        return content, usage

    def run(self, subtask: SubTask) -> SpecialistOutput:
        """Execute the subtask with primary model, falling back on error."""
        primary, fallback = self._get_models()
        messages = self._build_messages(subtask)
        model_used = primary

        t0 = time.monotonic()
        try:
            content, usage = self._call(primary, messages)
        except (Timeout, APIConnectionError, APIError) as exc:
            # Try fallback once
            try:
                model_used = fallback
                content, usage = self._call(fallback, messages)
            except Exception as exc2:
                latency_ms = int((time.monotonic() - t0) * 1000)
                return SpecialistOutput(
                    subtask_id=subtask.id,
                    task_type=self.task_type,
                    model_used=fallback,
                    content="",
                    latency_ms=latency_ms,
                    error=f"Primary ({primary}): {exc}. Fallback ({fallback}): {exc2}",
                )
        except Exception as exc:
            latency_ms = int((time.monotonic() - t0) * 1000)
            return SpecialistOutput(
                subtask_id=subtask.id,
                task_type=self.task_type,
                model_used=primary,
                content="",
                latency_ms=latency_ms,
                error=str(exc),
            )

        latency_ms = int((time.monotonic() - t0) * 1000)
        return SpecialistOutput(
            subtask_id=subtask.id,
            task_type=self.task_type,
            model_used=model_used,
            content=content,
            latency_ms=latency_ms,
            token_usage=usage or None,
        )
