"""HelpdeskEnv — Multi-Agent IT Helpdesk (Round 2)."""

from knowledge_base import KnowledgeBase, KBEntry
from models import (
    Ticket, HelpdeskAction, HelpdeskEnvState,
    StepResult, ErrorResponse, AgentRole
)
from tasks import get_random_ticket_scenario, get_all_ticket_scenarios
from graders import (
    grade_reply, grade_priority, grade_triage,
    grade_efficiency, grade_kb_contribution
)


class HelpdeskEnv:
    """
    Multi-agent helpdesk environment.

    Episode flow:
    1. reset() → loads a batch of tickets
    2. For each ticket:
       a. Triage agent classifies category, priority, tier
       b. Assigned agent (L1/L2/L3) takes resolution actions
       c. Agent may escalate → next tier agent takes over
       d. When resolved → move to next ticket
    3. Episode ends when all tickets are resolved or SLA breached

    Knowledge Base persists across episodes (self-improvement).
    """

    def __init__(self):
        self._kb = KnowledgeBase()       # Created ONCE, shared across episodes
        self._state = HelpdeskEnvState()
        self._tickets: list[Ticket] = []

    def reset(self, seed: int = None, num_tickets: int = 3):
        """Start a new episode with a batch of tickets.
        NOTE: KB is NOT reset — it accumulates (self-improvement)."""
        # ... select tickets, reset episode state ...
        pass

    def step(self, action: HelpdeskAction):
        """Process an agent's action on the current ticket.
        Routes by agent role, grades the action, handles escalation."""
        # ... implement multi-agent routing ...
        pass

    def state(self) -> HelpdeskEnvState:
        return self._state

    def kb(self) -> KnowledgeBase:
        return self._kb
