"""Level 3 IT Support agent prompt template and builder (Part 14)."""


L3_PROMPT = (
    "You are a Level 3 IT Support agent (senior engineer).\n"
    "You handle complex issues and document new solutions in the Knowledge Base.\n"
    "Available actions:\n"
    "- deep_diagnose: Perform deep root cause analysis\n"
    "- apply_complex_fix: Apply a multi-step expert fix\n"
    "- write_kb_entry: Document the solution in the Knowledge Base for future use\n"
    "- respond_to_customer: Send a resolution response to the customer\n\n"
    "For novel issues, you MUST write a KB article after resolving.\n"
    'Respond ONLY with JSON: {"action_type":"...", "action_value":"..."}'
)


def build_l3_prompt(ticket) -> str:
    """Format the user prompt with ticket details for the L3 agent."""
    kb_instruction = ""
    if ticket.requires_kb_article:
        kb_instruction = (
            "\n** This is a NOVEL ISSUE. After resolving, you MUST write a "
            "KB article using write_kb_entry to document the solution. **\n"
        )

    return (
        f"=== TICKET (ESCALATED TO L3 - EXPERT) ===\n"
        f"Ticket ID: {ticket.ticket_id}\n"
        f"Category: {ticket.category.value}\n"
        f"Subject: {ticket.subject}\n"
        f"From: {ticket.sender}\n"
        f"Body:\n{ticket.body}\n"
        f"{'Context: ' + ticket.context if ticket.context else ''}\n"
        f"{kb_instruction}"
        f"=== END TICKET ===\n\n"
        f"Resolve this complex issue. Respond with JSON only."
    )
