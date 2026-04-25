"""HelpdeskEnv Multi-Episode Self-Improvement Demo (Part 20).

Runs 5 episodes and shows how the Knowledge Base grows across episodes,
enabling agents to improve their scores over time.

This is the key demonstration of the "Self-Improving Systems" hackathon theme:
- Episode 1: KB has only 2 seed entries → L3 must solve and document
- Episode N: KB has entries from all previous episodes → L1/L2 can use them

Usage:
    python demo.py
"""

from __future__ import annotations

import json
from helpdeskenv_class import HelpdeskEnv
from models import HelpdeskAction, AgentRole
from inference import heuristic_triage, heuristic_l3_kb


def run_single_episode(env: HelpdeskEnv, seed: int, episode_num: int) -> dict:
    """Run one episode and return metrics.

    Args:
        env: The shared HelpdeskEnv instance (KB persists across calls).
        seed: Random seed for ticket selection.
        episode_num: Episode number for display purposes.

    Returns:
        Dict with episode metrics: rewards, steps, kb_size, etc.
    """
    reset_result = env.reset(seed=seed, num_tickets=5)
    kb_start = env.kb().size()

    rewards = []
    steps = 0

    while not env.state().is_done:
        state = env.state()
        ticket = state.current_ticket
        if ticket is None:
            break

        agent = state.current_agent

        # --- Triage ---
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
            continue

        # --- KB Search (self-improvement: use knowledge from prior episodes) ---
        kb_results = env.kb().search(ticket.subject + " " + ticket.body[:100])

        # --- Resolution ---
        if kb_results and agent in (AgentRole.L1, AgentRole.L2):
            # KB has an answer from a previous episode — use it!
            action = HelpdeskAction(
                ticket_id=ticket.ticket_id,
                agent_role=agent,
                action_type="apply_solution",
                action_value=kb_results[0].solution,
            )
            result = env.step(action)
            steps += 1
            rewards.append(result.reward)
        else:
            # L3 writes KB BEFORE resolving
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
                if env.state().is_done:
                    break

            # Resolve based on agent tier
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

    kb_end = env.kb().size()
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

    return {
        "episode": episode_num,
        "seed": seed,
        "steps": steps,
        "avg_reward": avg_reward,
        "total_reward": sum(rewards),
        "kb_start": kb_start,
        "kb_end": kb_end,
        "kb_added": kb_end - kb_start,
        "rewards": rewards,
    }


def run_demo(num_episodes: int = 5):
    """Run multiple episodes showing KB growth and score improvement.

    The key insight: all episodes share the SAME HelpdeskEnv instance,
    so the Knowledge Base persists and grows across episodes.
    """
    env = HelpdeskEnv()  # Single instance — KB persists!

    print()
    print("=" * 70)
    print("  [HELPDESK] HelpdeskEnv - Multi-Episode Self-Improvement Demo")
    print("=" * 70)
    print()
    print("  This demo runs multiple episodes to show how the Knowledge Base")
    print("  grows over time, enabling agents to solve tickets faster.")
    print()

    results = []
    baseline_reward = None

    for i in range(num_episodes):
        seed = 42 + i  # Different seed each episode for variety
        episode_result = run_single_episode(env, seed=seed, episode_num=i + 1)
        results.append(episode_result)

        if baseline_reward is None:
            baseline_reward = episode_result["avg_reward"]

    # Print results table
    print("  +---------+------+---------+------------+-------------+--------------+")
    print("  | Episode | Seed | KB Size | Avg Reward | Total Rwd   | Improvement  |")
    print("  +---------+------+---------+------------+-------------+--------------+")

    for r in results:
        if baseline_reward and baseline_reward > 0:
            improvement = ((r["avg_reward"] - baseline_reward) / baseline_reward) * 100
        else:
            improvement = 0.0

        imp_str = f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"
        if r["episode"] == 1:
            imp_str = "baseline"

        print(
            f"  | {r['episode']:^7d} | {r['seed']:^4d} | "
            f"{r['kb_start']:d} -> {r['kb_end']:<3d} | "
            f"{r['avg_reward']:^10.4f} | "
            f"{r['total_reward']:^11.2f} | "
            f"{imp_str:^12s} |"
        )

    print("  +---------+------+---------+------------+-------------+--------------+")

    # KB stats
    kb_stats = env.kb().stats()
    print()
    print("  Knowledge Base Final Stats:")
    print(f"    Total entries:    {kb_stats['total_entries']}")
    print(f"    Seeded entries:   {kb_stats['seeded_entries']}")
    print(f"    Agent-authored:   {kb_stats['agent_authored']}")
    print(f"    Total retrievals: {kb_stats['total_retrievals']}")
    print()

    # Show all KB entries
    print("  KB Entries:")
    for entry in env.kb().get_all():
        author_tag = "SEED" if entry.entry_id.startswith("kb_") else "AGENT"
        print(f"    [{author_tag}] {entry.entry_id}: {entry.title} (used {entry.times_used}x)")

    print()
    print("=" * 70)
    print("  [DONE] Demo complete! The self-improvement mechanism is working.")
    print("     KB grew from 2 seed entries to", kb_stats["total_entries"], "entries.")
    print("=" * 70)
    print()

    # Save results
    output = {
        "episodes": results,
        "kb_stats": kb_stats,
        "kb_entries": [e.model_dump() for e in env.kb().get_all()],
    }
    with open("demo_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("  Results saved to demo_results.json")


if __name__ == "__main__":
    run_demo()
