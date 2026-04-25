"""Level 2 IT Support agent prompt template and builder (Part 14)."""


L2_PROMPT = (
    "You are a Level 2 IT Support agent. You handle medium-complexity issues.\n"
    "Available actions:\n"
    "- diagnose: Investigate the issue further\n"
    "- request_info: Request additional information from the customer\n"
    "- apply_fix: Apply a technical fix to resolve the ticket\n"
    "- escalate: Escalate to L3 if the issue is too complex\n"
    "- respond_to_customer: Send a resolution response to the customer\n\n"
    'Respond ONLY with JSON: {"action_type":"...", "action_value":"..."}'
)


def build_l2_prompt(ticket) -> str:
    """Format the user prompt with ticket details for the L2 agent."""
    return (
        f"=== TICKET (ESCALATED TO L2) ===\n"
        f"Ticket ID: {ticket.ticket_id}\n"
        f"Category: {ticket.category.value}\n"
        f"Subject: {ticket.subject}\n"
        f"From: {ticket.sender}\n"
        f"Body:\n{ticket.body}\n"
        f"{'Context: ' + ticket.context if ticket.context else ''}\n"
        f"=== END TICKET ===\n\n"
        f"Diagnose and resolve this ticket. Respond with JSON only."
    )
