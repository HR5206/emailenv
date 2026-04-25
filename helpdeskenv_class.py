"""HelpdeskEnv — Multi-Agent IT Helpdesk (Round 2)."""

import json
import random
from knowledge_base import KnowledgeBase, KBEntry
from models import (
    Ticket, HelpdeskAction, HelpdeskEnvState,
    StepResult, ErrorResponse, AgentRole, SupportTier
)
from tasks import get_all_ticket_scenarios
from graders import (
    grade_reply, grade_priority, grade_triage,
    grade_efficiency, grade_kb_contribution
)


TIER_TO_AGENT = {
    SupportTier.L1: AgentRole.L1,
    SupportTier.L2: AgentRole.L2,
    SupportTier.L3: AgentRole.L3,
}

ESCALATION_PATH = {
    AgentRole.L1: AgentRole.L2,
    AgentRole.L2: AgentRole.L3,
    AgentRole.L3: None,
}


class HelpdeskEnv:
    """
    Multi-agent helpdesk environment.

    Episode flow:
    1. reset() loads a batch of tickets
    2. For each ticket:
       a. Triage agent classifies category, priority, tier
       b. Assigned agent (L1/L2/L3) takes resolution actions
       c. Agent may escalate to next tier
       d. When resolved, move to next ticket
    3. Episode ends when all tickets are resolved or SLA breached

    Knowledge Base persists across episodes (self-improvement).
    """

    def __init__(self):
        self._kb = KnowledgeBase()
        self._state = HelpdeskEnvState()
        self._tickets: list[Ticket] = []
        self._current_ticket_idx: int = 0
        self._ticket_resolved: bool = False
        self._triage_done: bool = False

    def reset(self, seed: int = None, num_tickets: int = 3) -> dict:
        """Start a new episode. KB is NOT reset (self-improvement)."""
        all_scenarios = get_all_ticket_scenarios()
        if seed is not None:
            random.seed(seed)
        selected = random.sample(
            all_scenarios, min(num_tickets, len(all_scenarios))
        )

        self._tickets = selected
        self._current_ticket_idx = 0
        self._ticket_resolved = False
        self._triage_done = False

        prior_kb = self._state.kb_entries_added

        self._state = HelpdeskEnvState(
            current_ticket=self._tickets[0],
            current_agent=AgentRole.TRIAGE,
            ticket_number=0,
            total_tickets=len(self._tickets),
            total_reward=0.0,
            steps_on_current_ticket=0,
            is_done=False,
            history=[],
            kb_entries_added=prior_kb,
            escalation_count=0,
        )

        return {
            "ticket": self._tickets[0].model_dump(),
            "current_agent": AgentRole.TRIAGE.value,
            "total_tickets": len(self._tickets),
            "kb_size": self._kb.size(),
        }

    def step(self, action: HelpdeskAction) -> StepResult | ErrorResponse:
        """Process an agent action. Routes by role and action_type."""

        if not self._tickets:
            return ErrorResponse(
                error="Environment not initialized.",
                detail="Call reset() first.",
            )

        if self._state.is_done:
            return ErrorResponse(
                error="Episode already complete.",
                detail="Call reset() to start a new episode.",
            )

        ticket = self._state.current_ticket
        if action.ticket_id != ticket.ticket_id:
            return ErrorResponse(
                error="Ticket ID mismatch.",
                detail=(
                    f"Expected '{ticket.ticket_id}' "
                    f"got '{action.ticket_id}'"
                ),
            )

        self._state.steps_on_current_ticket += 1

        if action.agent_role == AgentRole.TRIAGE:
            result = self._handle_triage(ticket, action)
        elif action.action_type == "escalate":
            result = self._handle_escalation(ticket, action)
        elif action.action_type == "search_kb":
            result = self._handle_kb_search(ticket, action)
        elif action.action_type == "write_kb_entry":
            result = self._handle_kb_write(ticket, action)
        elif action.action_type == "respond_to_customer":
            result = self._handle_respond_to_customer(ticket, action)
        elif action.action_type in (
            "apply_solution",
            "apply_fix",
            "apply_complex_fix",
        ):
            result = self._handle_resolution(ticket, action)
        elif action.action_type in (
            "diagnose",
            "deep_diagnose",
            "request_info",
        ):
            result = self._handle_diagnostic(ticket, action)
        else:
            return ErrorResponse(
                error=f"Unknown action_type: {action.action_type}",
                detail=f"Agent: {action.agent_role.value}",
            )

        self._state.total_reward = round(
            self._state.total_reward + result.reward, 2
        )
        self._state.history.append(result)

        if self._ticket_resolved:
            self._advance_to_next_ticket(result)

        if (
            not self._ticket_resolved
            and self._state.steps_on_current_ticket >= ticket.sla_steps
        ):
            result.feedback = (
                (result.feedback or "")
                + f"\nSLA BREACHED after "
                + f"{self._state.steps_on_current_ticket} steps "
                + f"(limit {ticket.sla_steps})."
            )
            self._ticket_resolved = True
            self._advance_to_next_ticket(result)

        return result

    # ----- Triage -----

    def _handle_triage(
        self, ticket: Ticket, action: HelpdeskAction
    ) -> StepResult:
        result = grade_triage(ticket, action)

        try:
            parsed = json.loads(action.action_value)
            tier_str = parsed.get(
                "tier", ticket.ground_truth_tier.value
            )
        except Exception:
            tier_str = ticket.ground_truth_tier.value

        tier_map = {
            "l1": SupportTier.L1,
            "l2": SupportTier.L2,
            "l3": SupportTier.L3,
        }
        assigned = tier_map.get(
            tier_str.lower(), ticket.ground_truth_tier
        )
        self._state.current_agent = TIER_TO_AGENT[assigned]
        self._triage_done = True

        result.feedback = (
            (result.feedback or "")
            + f"\nRouted to {self._state.current_agent.value} agent."
        )
        return result

    # ----- Escalation -----

    def _handle_escalation(
        self, ticket: Ticket, action: HelpdeskAction
    ) -> StepResult:
        current = self._state.current_agent
        nxt = ESCALATION_PATH.get(current)

        if nxt is None:
            return StepResult(
                task_id=ticket.ticket_id,
                reward=0.0,
                done=False,
                feedback=(
                    f"{current.value} is the highest tier. "
                    "Cannot escalate further."
                ),
                correct_answer=None,
            )

        self._state.current_agent = nxt
        self._state.escalation_count += 1

        return StepResult(
            task_id=ticket.ticket_id,
            reward=0.1,
            done=False,
            feedback=(
                f"Escalated from {current.value} to {nxt.value}. "
                f"Reason: {action.action_value}"
            ),
            correct_answer=None,
        )

    # ----- KB Search -----

    def _handle_kb_search(
        self, ticket: Ticket, action: HelpdeskAction
    ) -> StepResult:
        results = self._kb.search(action.action_value)

        if results:
            lines = [
                f"- {r.title}: {r.solution[:80]}..."
                for r in results
            ]
            feedback = (
                f"Found {len(results)} KB entries:\n"
                + "\n".join(lines)
            )
            reward = 0.2
        else:
            feedback = "No matching KB entries found."
            reward = 0.05

        return StepResult(
            task_id=ticket.ticket_id,
            reward=reward,
            done=False,
            feedback=feedback,
            correct_answer=None,
        )

    # ----- KB Write -----

    def _handle_kb_write(
        self, ticket: Ticket, action: HelpdeskAction
    ) -> StepResult:
        kb_score = grade_kb_contribution(
            action.action_value, ticket
        )

        eid = f"agent_{ticket.ticket_id}_{self._kb.size() + 1}"
        new_entry = KBEntry(
            entry_id=eid,
            ticket_category=ticket.category.value,
            title=f"Solution for {ticket.subject[:50]}",
            problem_description=ticket.body[:200],
            solution=action.action_value,
            keywords=ticket.subject.lower().split()[:5],
            created_by=action.agent_role.value,
        )

        added = self._kb.add(new_entry)
        if added:
            self._state.kb_entries_added += 1
            feedback = (
                f"KB article created (id={eid}). "
                f"Quality score: {kb_score:.2f}"
            )
        else:
            feedback = f"KB article {eid} already exists."

        return StepResult(
            task_id=ticket.ticket_id,
            reward=round(kb_score * 0.5, 2),
            done=False,
            feedback=feedback,
            correct_answer=None,
        )

    # ----- Respond to Customer -----

    def _handle_respond_to_customer(
        self, ticket: Ticket, action: HelpdeskAction
    ) -> StepResult:
        # Create EmailTask-compatible wrapper for grade_reply
        temp_task = EmailTask(
            task_id=ticket.ticket_id,
            task_type="reply",
            subject=ticket.subject,
            sender=ticket.sender,
            body=ticket.body,
            context=ticket.context,
            ground_truth=ticket.ground_truth_resolution,
        )
        temp_action = AgentAction(
            task_id=ticket.ticket_id,
            action_value=action.action_value,
        )

        reply_result = grade_reply(temp_task, temp_action)

        eff = grade_efficiency(
            self._state.steps_on_current_ticket,
            ticket.sla_steps,
            self._state.escalation_count,
        )

        final = round(reply_result.reward * 0.7 + eff * 0.3, 2)
        final = min(1.0, max(0.0, final))

        self._ticket_resolved = True

        return StepResult(
            task_id=ticket.ticket_id,
            reward=final,
            done=False,
            feedback=(
                f"Customer response by {action.agent_role.value}.\n"
                f"  Reply quality: {reply_result.reward:.2f}\n"
                f"  Efficiency: {eff:.2f}\n"
                f"  Final reward: {final:.2f}\n"
                f"  {reply_result.feedback}"
            ),
            correct_answer=ticket.ground_truth_resolution,
        )

    # ----- Resolution -----

    def _handle_resolution(
        self, ticket: Ticket, action: HelpdeskAction
    ) -> StepResult:
        res_text = action.action_value.strip().lower()
        gt = (ticket.ground_truth_resolution or "").strip().lower()

        stopwords = {
            "the", "a", "an", "is", "it", "in", "on",
            "to", "for", "of", "and", "or",
        }
        gt_words = {
            w for w in gt.split()
            if w not in stopwords and len(w) > 2
        }
        res_words = {
            w for w in res_text.split()
            if w not in stopwords and len(w) > 2
        }

        if gt_words:
            overlap = (
                len(gt_words.intersection(res_words)) / len(gt_words)
            )
        else:
            overlap = 0.5

        res_score = min(1.0, overlap * 1.5)

        eff = grade_efficiency(
            self._state.steps_on_current_ticket,
            ticket.sla_steps,
            self._state.escalation_count,
        )

        final = round(res_score * 0.7 + eff * 0.3, 2)
        final = min(1.0, max(0.0, final))

        self._ticket_resolved = True

        return StepResult(
            task_id=ticket.ticket_id,
            reward=final,
            done=False,
            feedback=(
                f"Ticket resolved by {action.agent_role.value}.\n"
                f"  Resolution quality: {res_score:.2f}\n"
                f"  Efficiency: {eff:.2f}\n"
                f"  Final reward: {final:.2f}"
            ),
            correct_answer=ticket.ground_truth_resolution,
        )

    # ----- Diagnostic -----

    def _handle_diagnostic(
        self, ticket: Ticket, action: HelpdeskAction
    ) -> StepResult:
        rewards = {
            "diagnose": 0.1,
            "deep_diagnose": 0.15,
            "request_info": 0.05,
        }
        reward = rewards.get(action.action_type, 0.05)

        return StepResult(
            task_id=ticket.ticket_id,
            reward=reward,
            done=False,
            feedback=(
                f"{action.agent_role.value} performed "
                f"{action.action_type}: "
                f"{action.action_value[:100]}"
            ),
            correct_answer=None,
        )

    # ----- Advance -----

    def _advance_to_next_ticket(self, result: StepResult):
        self._current_ticket_idx += 1
        self._ticket_resolved = False
        self._triage_done = False

        if self._current_ticket_idx >= len(self._tickets):
            self._state.is_done = True
            self._state.current_ticket = None
            self._state.current_agent = None
            result.done = True
            result.feedback = (
                (result.feedback or "")
                + f"\n\nEpisode complete! "
                + f"Total reward: "
                + f"{self._state.total_reward:.2f}"
                + f" / {self._state.total_tickets}.00"
                + f"\nKB stats: {self._kb.stats()}"
            )
        else:
            nxt = self._tickets[self._current_ticket_idx]
            self._state.current_ticket = nxt
            self._state.current_agent = AgentRole.TRIAGE
            self._state.ticket_number = self._current_ticket_idx
            self._state.steps_on_current_ticket = 0
            result.done = False
            result.feedback = (
                (result.feedback or "")
                + f"\nAdvancing to ticket "
                + f"{self._current_ticket_idx + 1}"
                + f"/{self._state.total_tickets}: "
                + f"{nxt.subject}"
            )

    # ----- Accessors -----

    def state(self) -> HelpdeskEnvState:
        return self._state

    def kb(self) -> KnowledgeBase:
        return self._kb
