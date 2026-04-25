"""Level 1 IT Support agent prompt template and builder (Part 14).

L1 agents handle simple, well-documented issues like password resets.
They have access to the Knowledge Base and can:
  - search_kb: Look up known solutions
  - apply_solution: Apply a KB-matched solution
  - respond_to_customer: Send a resolution message
  - escalate: Pass the ticket to L2 if beyond their scope
"""


L1_PROMPT = (
    "You are a Level 1 IT Support agent. You handle simple issues.\n"
    "You have access to a Knowledge Base of past solutions.\n\n"
    "Available actions:\n"
    "- search_kb: Search the Knowledge Base for a known solution\n"
    "- apply_solution: Apply a solution from the KB or standard procedure\n"
    "- respond_to_customer: Send a resolution response to the customer\n"
    "- escalate: Escalate to L2 if the issue is beyond your scope\n\n"
    "Strategy:\n"
    "1. First, search the KB for a matching solution\n"
    "2. If found, apply it directly\n"
    "3. If not found, escalate to L2\n\n"
    'Respond ONLY with JSON: {"action_type":"...", "action_value":"..."}'
)


def build_l1_prompt(ticket, kb_results=None) -> str:
    """Format the user prompt with ticket details and KB results for L1.

    Args:
        ticket: A Ticket model instance.
        kb_results: Optional list of KBEntry matches from a prior search.

    Returns:
        A formatted string ready to send as the user message to the LLM.
    """
    kb_section = ""
    if kb_results:
        kb_lines = []
        for i, entry in enumerate(kb_results, 1):
            kb_lines.append(
                f"  {i}. [{entry.title}]\n"
                f"     Problem: {entry.problem_description}\n"
                f"     Solution: {entry.solution}\n"
            )
        kb_section = (
            "\n=== KNOWLEDGE BASE RESULTS ===\n"
            + "\n".join(kb_lines)
            + "=== END KB RESULTS ===\n"
        )

    return (
        f"=== TICKET (ASSIGNED TO L1) ===\n"
        f"Ticket ID: {ticket.ticket_id}\n"
        f"Category: {ticket.category.value}\n"
        f"Subject: {ticket.subject}\n"
        f"From: {ticket.sender}\n"
        f"Body:\n{ticket.body}\n"
        f"{'Context: ' + ticket.context if ticket.context else ''}\n"
        f"=== END TICKET ==="
        f"{kb_section}\n\n"
        f"Resolve this ticket using KB if possible. Respond with JSON only."
    )
