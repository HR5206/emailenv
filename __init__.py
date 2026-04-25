"""HelpdeskEnv — Multi-Agent IT Helpdesk OpenEnv Environment."""

from helpdeskenv_class import HelpdeskEnv
from models import (
    # Round 1 models (kept for backward compat per master prompt)
    Email,
    Observation,
    Action,
    State,
    Reward,
    TaskType,
    EmailTask,
    AgentAction,
    StepResult,
    EnvState,
    ResetResponse,
    ErrorResponse,
    # Round 2 enums
    TicketCategory,
    AgentRole,
    TicketPriority,
    SupportTier,
    # Round 2 models
    Ticket,
    HelpdeskAction,
    HelpdeskEnvState,
    HelpdeskResetResponse,
)

__all__ = [
    "HelpdeskEnv",
    # Round 1
    "Email",
    "Observation",
    "Action",
    "State",
    "Reward",
    "TaskType",
    "EmailTask",
    "AgentAction",
    "StepResult",
    "EnvState",
    "ResetResponse",
    "ErrorResponse",
    # Round 2
    "TicketCategory",
    "AgentRole",
    "TicketPriority",
    "SupportTier",
    "Ticket",
    "HelpdeskAction",
    "HelpdeskEnvState",
    "HelpdeskResetResponse",
]