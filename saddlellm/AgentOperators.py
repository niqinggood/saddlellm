"""Backward-compatible Agent imports.

Agent orchestration moved to :mod:`saddle_ml.agent`.  This module remains so
older integrations importing ``saddle_llm.AgentOperators`` keep working while
the LLM package owns only model-provider behavior.
"""

from saddle_ml.agent.operator import (  # noqa: F401
    AgentOperatorError,
    RESULT_SCHEMA_VERSION,
    SAFE_TOOLS,
    SCHEMA_VERSION,
    SUPPORTED_AGENT_OPERATORS,
    ToolRuntime,
    _json_safe,
    normalize_agent_config,
    run_agent_operator,
    safe_calculate,
)
try:
    from .AgentModelProvider import endpoint as _endpoint
except ImportError:  # pragma: no cover - compatibility for direct file loaders
    from saddle_llm.AgentModelProvider import endpoint as _endpoint


__all__ = [
    "AgentOperatorError",
    "RESULT_SCHEMA_VERSION",
    "SAFE_TOOLS",
    "SCHEMA_VERSION",
    "SUPPORTED_AGENT_OPERATORS",
    "ToolRuntime",
    "normalize_agent_config",
    "run_agent_operator",
    "safe_calculate",
]
