"""Triage agent prompt template and builder (Part 13).

The Triage agent is the first to see every ticket. Its job:
  1. Classify the ticket category (password_reset, software_install, etc.)
  2. Assess priority (low / medium / high / critical)
  3. Assign to the correct support tier (L1 / L2 / L3)

Output format: JSON with category, priority, tier.
"""


TRIAGE_PROMPT = (
    "You are a Triage agent at an IT helpdesk.\n"
    "Your job is to classify incoming support tickets.\n\n"
    "For each ticket, determine:\n"
    "1. **category**: one of: password_reset, software_install, "
    "network_outage, data_recovery, novel_issue\n"
    "2. **priority**: one of: low, medium, high, critical\n"
    "3. **tier**: which support tier should handle this: l1, l2, or l3\n\n"
    "Guidelines:\n"
    "- password_reset → usually L1, medium priority\n"
    "- software_install → usually L2, medium priority\n"
    "- network_outage → usually L3, critical if many users affected\n"
    "- data_recovery → usually L3, critical if business data\n"
    "- novel_issue → L3, high priority (unknown root cause)\n\n"
    'Respond ONLY with JSON: {"category":"...", "priority":"...", "tier":"..."}'
)


def build_triage_prompt(ticket) -> str:
    """Format the user prompt with ticket details for the Triage agent.

    Args:
        ticket: A Ticket model instance with subject, sender, body, context.

    Returns:
        A formatted string ready to send as the user message to the LLM.
    """
    return (
        f"=== NEW TICKET ===\n"
        f"Ticket ID: {ticket.ticket_id}\n"
        f"Subject: {ticket.subject}\n"
        f"From: {ticket.sender}\n"
        f"Body:\n{ticket.body}\n"
        f"{'Context: ' + ticket.context if ticket.context else ''}\n"
        f"=== END TICKET ===\n\n"
        f"Classify this ticket. Respond with JSON only."
    )
