"""Consolidated models for the EmailEnv environment.

Contains all Pydantic models used across the application:
- EmailEnv OpenEnv models (Email, Observation, Action, State, Reward)
- Task and grading models (EmailTask, AgentAction, StepResult, EnvState, etc.)
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Literal
from enum import Enum
from pydantic import BaseModel, Field

# ============================================================================
# OpenEnv Core Models (with fallback classes)
# ============================================================================

try:
    from openenv_core import Observation as OpenEnvObservation        
    from openenv_core import Action as OpenEnvAction
    from openenv_core import State as OpenEnvState
    from openenv_core import Reward as OpenEnvReward
except Exception:  # type: ignore
    class OpenEnvObservation(BaseModel):
        pass

    class OpenEnvAction(BaseModel):
        pass

    class OpenEnvState(BaseModel):
        pass

    class OpenEnvReward(BaseModel):
        pass


# ============================================================================
# Email and OpenEnv Models
# ============================================================================

class Email(BaseModel):
    """Represents an email message."""
    id: str
    subject: str
    body: str
    sender: str
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Observation(OpenEnvObservation):
    """Observation from the EmailEnv environment."""
    email: Optional[Email] = None
    task: Literal["spam_classification", "email_prioritization", "reply_generation"]
    step_index: int
    total_steps: int
    remaining_emails: int


class Action(OpenEnvAction):
    """Action submitted to the EmailEnv environment."""
    type: Literal["classify_spam", "set_priority", "generate_reply", "skip"]
    is_spam: Optional[bool] = None
    priority: Optional[Literal["low", "medium", "high"]] = None
    reply_text: Optional[str] = None


class State(OpenEnvState):
    """State snapshot of the EmailEnv environment."""
    current_email_index: int
    total_emails: int
    completed: bool
    task: Literal["spam_classification", "email_prioritization", "reply_generation"]


class Reward(OpenEnvReward):
    """Reward signal from the EmailEnv environment."""
    value: float
    task_scores: Dict[str, float] = Field(default_factory=dict)
    done: bool = False


# ============================================================================
# Task and Grading Models
# ============================================================================

class TaskType(str, Enum):
    """
    Defines the three task types in EmailEnv.
    Using str + Enum means the values are plain strings
    (e.g. "spam", "priority", "reply") - easy to use in JSON.
    """
    SPAM = "spam"
    PRIORITY = "priority"
    REPLY = "reply"


class EmailTask(BaseModel):
    """
    Represents one email scenario presented to the agent.
    This is the 'observation' - what the agent sees.
    """
    task_id: str = Field(..., description="Unique identifier for this task")
    task_type: TaskType = Field(..., description="Which of the 3 tasks is this")
    subject: str = Field(..., description="Email subject line")
    sender: str = Field(..., description="Email sender address")
    body: str = Field(..., description="Full email body text")
    context: Optional[str] = Field(
        None,
        description="Extra content for the agent, e.g. 'You are a customer support rep'"
    )
    ground_truth: Optional[str] = Field(
        None,
        description="The correct answer (hidden from agent, used by grader)"
    )


class AgentAction(BaseModel):
    """
    Represents the action submitted by the agent.

    For Task 1 (spam): action_value = "spam" or "not_spam"
    For Task 2 (priority): action_value = "high", "medium" or "low"
    For Task 3 (reply): action_value = the full drafted reply text
    """
    task_id: str = Field(..., description="Must match the current task's task_id")
    action_value: str = Field(..., description="The agent's answer or response")


class StepResult(BaseModel):
    """
    Returned by step() after the agent submits an action.
    Contains the reward, whether the episode is done,
    and optional feedback for the agent.
    """
    task_id: str = Field(..., description="The task that was just graded")
    reward: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score between 0.0 (wrong) and 1.0 (perfect)"
    )
    done: bool = Field(..., description="True if the episode is complete")
    feedback: Optional[str] = Field(
        None,
        description="Human-readable explanation of the score"
    )
    correct_answer: Optional[str] = Field(
        None,
        description="Revealed after grading so agent can learn"
    )


class EnvState(BaseModel):
    """
    A snapshot of the environment at any point in time.
    Returned by state() endpoint.
    """
    current_task: Optional[EmailTask] = Field(
        None,
        description="The email task that is currently active"
    )
    task_number: int = Field(
        default=0,
        description="Which task index we are on (0, 1, or 2)"
    )
    total_reward: float = Field(
        default=0.0,
        description="Cumulative reward across all tasks so far"
    )
    is_done: bool = Field(
        default=False,
        description="True when all 3 tasks are complete"
    )
    history: list[StepResult] = Field(
        default_factory=list,
        description="List of all past StepResults in this episode"
    )


class ResetResponse(BaseModel):
    """
    Returned when the agent calls reset() to start a new episode.
    Gives the agent its first task.
    """
    observation: EmailTask = Field(
        ...,
        description="The first email task for the agent to solve"
    )
    available_tasks: list[str] = Field(
        default=["spam_classification", "email_prioritization", "reply_generation"],
        description="List of available task types"
    )


class ErrorResponse(BaseModel):
    """
    Returned when something goes wrong (wrong task_id, bad input, etc.)
    """
    error: str = Field(..., description="Description of what went wrong")
    detail: Optional[str] = Field(None, description="Extra data if available")


# ============================================================================
# Export All Models
# ============================================================================

__all__ = [
    # OpenEnv models
    "Email",
    "Observation",
    "Action",
    "State",
    "Reward",
    # Task models
    "TaskType",
    "EmailTask",
    "AgentAction",
    "StepResult",
    "EnvState",
    "ResetResponse",
    "ErrorResponse",
]
