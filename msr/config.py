from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Always resolve .env relative to the project root (one level above this file)
_PROJECT_ROOT = Path(__file__).parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        env_ignore_empty=True,   # treat empty OS env vars as unset → .env file wins
        extra="ignore",
    )

    anthropic_api_key: str = ""
    openai_api_key: str = ""
    groq_api_key: str = ""
    gemini_api_key: str = ""
    perplexity_api_key: str = ""
    xai_api_key: str = ""

    msr_default_timeout_s: int = 30
    msr_max_retries: int = 2
    msr_log_level: str = "INFO"

    # Model assignments — primary + fallback per task type
    # Keys match TaskType enum values
    model_map: dict = {
        "code":       {"primary": "claude-sonnet-4-6",                      "fallback": "gpt-4o"},
        "math":       {"primary": "groq/deepseek-r1-distill-llama-70b",     "fallback": "gpt-4o"},
        "research":   {"primary": "perplexity/sonar-pro",                   "fallback": "claude-sonnet-4-6"},
        "summarize":  {"primary": "gemini/gemini-2.5-flash",                "fallback": "claude-haiku-4-5"},
        "structured": {"primary": "gpt-4o",                                 "fallback": "gemini/gemini-2.5-flash"},
        "factcheck":  {"primary": "xai/grok-3",                             "fallback": "perplexity/sonar-pro"},
        "general":    {"primary": "claude-sonnet-4-6",                      "fallback": "gpt-4o"},
    }

    router_model: str = "gemini/gemini-2.5-flash"
    # Orchestrator: primary for best decomposition quality, fallback on credit/API errors
    orchestrator_primary: str = "claude-sonnet-4-6"
    orchestrator_fallback: str = "gpt-4o"
    # Keep planner_model as alias used by legacy planner module
    planner_model: str = "claude-sonnet-4-6"
    verifier_model: str = "gpt-4o"
    synthesizer_model: str = "claude-sonnet-4-6"
    synthesizer_fallback: str = "gpt-4o"


settings = Settings()

# Push loaded keys into os.environ so litellm can pick them up.
# litellm reads ANTHROPIC_API_KEY, OPENAI_API_KEY, etc. from the process
# environment directly. If the OS env had an empty string, pydantic-settings
# resolved the correct value from .env — but os.environ is unchanged, so we
# sync back here.
import os as _os
_KEY_MAP = {
    "ANTHROPIC_API_KEY":  settings.anthropic_api_key,
    "OPENAI_API_KEY":     settings.openai_api_key,
    "GROQ_API_KEY":       settings.groq_api_key,
    "GEMINI_API_KEY":     settings.gemini_api_key,
    "PERPLEXITY_API_KEY": settings.perplexity_api_key,
    "XAI_API_KEY":        settings.xai_api_key,
}
for _k, _v in _KEY_MAP.items():
    if _v:  # only set if we actually have a value
        _os.environ[_k] = _v
