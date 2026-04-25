"""Quick import test for HelpdeskEnv."""
import sys

def test_imports():
    print("Testing HelpdeskEnv imports...")

    try:
        print("  Importing models...")
        from models import (
            TicketCategory, AgentRole, TicketPriority, SupportTier,
            Ticket, HelpdeskAction, HelpdeskEnvState, HelpdeskResetResponse,
            StepResult, ErrorResponse,
        )
        print("  OK")

        print("  Importing helpdeskenv_class...")
        from helpdeskenv_class import HelpdeskEnv
        print("  OK")

        print("  Creating HelpdeskEnv instance...")
        env = HelpdeskEnv()
        print("  OK")

        print("  Testing reset...")
        result = env.reset(seed=42)
        print(f"  OK — ticket={result.ticket.ticket_id}, agent={result.current_agent.value}")

        print("\nAll imports passed!")
        return True

    except Exception as e:
        print(f"\nFailed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
