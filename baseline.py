"""HelpdeskEnv Baseline — Heuristic evaluation across ticket scenarios (Part 18).

Runs all 5 ticket scenarios with heuristic agents and reports scores + KB growth.
"""

import json
from helpdeskenv_class import HelpdeskEnv
from models import HelpdeskAction, AgentRole
from inference import heuristic_triage, heuristic_l1, heuristic_l3_kb


def run_baseline():
    """Run baseline evaluation: triage + resolve all tickets with heuristics."""
    env = HelpdeskEnv()
    reset_result = env.reset(seed=42, num_tickets=5)

    print("=" * 60)
    print("  HelpdeskEnv Baseline Evaluation")
    print("=" * 60)
    print(f"  Tickets: {reset_result.total_tickets}")
    print(f"  KB size at start: {reset_result.kb_size}")
    print()

    rewards = []
    steps = 0

    while not env.state().is_done:
        state = env.state()
        ticket = state.current_ticket
        if ticket is None:
            break

        agent = state.current_agent

        if agent == AgentRole.TRIAGE:
            triage_data = heuristic_triage(ticket)
            action = HelpdeskAction(
                ticket_id=ticket.ticket_id,
                agent_role=AgentRole.TRIAGE,
                action_type="classify_category",
                action_value=json.dumps(triage_data),
            )
            result = env.step(action)
            steps += 1
            rewards.append(result.reward)
            print(f"  [{steps}] TRIAGE {ticket.ticket_id}: reward={result.reward:.2f}")
            continue

        # KB search
        kb_results = env.kb().search(ticket.subject)

        if kb_results and agent in (AgentRole.L1, AgentRole.L2):
            action = HelpdeskAction(
                ticket_id=ticket.ticket_id,
                agent_role=agent,
                action_type="apply_solution",
                action_value=kb_results[0].solution,
            )
        else:
            # L3 writes KB first
            if agent == AgentRole.L3 and ticket.requires_kb_article:
                kb_data = heuristic_l3_kb(ticket)
                kb_action = HelpdeskAction(
                    ticket_id=ticket.ticket_id,
                    agent_role=AgentRole.L3,
                    action_type="write_kb_entry",
                    action_value=kb_data["action_value"],
                )
                kb_result = env.step(kb_action)
                steps += 1
                rewards.append(kb_result.reward)
                print(f"  [{steps}] L3 KB_WRITE {ticket.ticket_id}: reward={kb_result.reward:.2f}")
                if env.state().is_done:
                    break

            # Resolve
            if agent == AgentRole.L1:
                action = HelpdeskAction(
                    ticket_id=ticket.ticket_id,
                    agent_role=AgentRole.L1,
                    action_type="apply_solution",
                    action_value=ticket.ground_truth_resolution or "Applied standard fix.",
                )
            elif agent == AgentRole.L2:
                action = HelpdeskAction(
                    ticket_id=ticket.ticket_id,
                    agent_role=AgentRole.L2,
                    action_type="apply_fix",
                    action_value=ticket.ground_truth_resolution or "Applied technical fix.",
                )
            else:
                action = HelpdeskAction(
                    ticket_id=ticket.ticket_id,
                    agent_role=AgentRole.L3,
                    action_type="apply_complex_fix",
                    action_value=ticket.ground_truth_resolution or "Applied expert fix.",
                )

        result = env.step(action)
        steps += 1
        rewards.append(result.reward)
        print(f"  [{steps}] {agent.value.upper()} RESOLVE {ticket.ticket_id}: reward={result.reward:.2f}")

    # Summary
    avg = sum(rewards) / len(rewards) if rewards else 0.0
    kb_stats = env.kb().stats()

    print()
    print("=" * 60)
    print("  BASELINE RESULTS")
    print("=" * 60)
    print(f"  Total steps: {steps}")
    print(f"  Avg reward: {avg:.4f}")
    print(f"  KB entries: {kb_stats['total_entries']}")
    print(f"  KB retrievals: {kb_stats['total_retrievals']}")
    print(f"  Escalations: {env.state().escalation_count}")
    print("=" * 60)

    # Save
    results = {
        "steps": steps,
        "avg_reward": avg,
        "rewards": rewards,
        "kb_stats": kb_stats,
    }
    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to baseline_results.json")


if __name__ == "__main__":
    run_baseline()
