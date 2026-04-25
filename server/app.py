"""FastAPI server for the HelpdeskEnv multi-agent IT helpdesk.

Thin wrapper — ALL logic lives in HelpdeskEnv. The server only
serializes/deserializes and forwards to the environment.
"""

import os
import logging
from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from helpdeskenv_class import HelpdeskEnv
from models import (
    HelpdeskAction,
    HelpdeskEnvState,
    HelpdeskResetResponse,
    StepResult,
    ErrorResponse,
    AgentRole,
)

# Configure structured logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Environment variables
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-5-nano")

app = FastAPI(title="HelpdeskEnv", version="2.0.0")
_env = HelpdeskEnv()


class ResetRequest(BaseModel):
    seed: Optional[int] = None
    num_tickets: Optional[int] = 3

    class Config:
        json_schema_extra = {
            "example": {"seed": 42, "num_tickets": 3}
        }


# ============================================================================
# Required OpenEnv validator endpoints
# ============================================================================

@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint — must return 'healthy' for OpenEnv validator."""
    return {"status": "healthy"}


@app.get("/metadata")
async def metadata() -> Dict[str, Any]:
    """Environment metadata — required by OpenEnv validator."""
    return {
        "name": "HelpdeskEnv",
        "description": (
            "Multi-agent IT helpdesk simulation with triage, "
            "tiered support, SLA tracking, and knowledge base learning"
        ),
        "version": "2.0.0",
        "author": "HelpdeskEnv Maintainers",
    }


@app.get("/schema")
async def schema() -> Dict[str, Any]:
    """Action/observation/state schemas — required by OpenEnv validator."""
    return {
        "action": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "string"},
                "agent_role": {
                    "type": "string",
                    "enum": ["triage", "l1", "l2", "l3"],
                },
                "action_type": {
                    "type": "string",
                    "enum": [
                        "classify_category", "set_priority", "assign_tier",
                        "search_kb", "apply_solution", "respond_to_customer",
                        "escalate", "request_info", "diagnose",
                        "deep_diagnose", "apply_fix", "apply_complex_fix",
                        "write_kb_entry",
                    ],
                },
                "action_value": {"type": "string"},
            },
            "required": ["ticket_id", "agent_role", "action_type", "action_value"],
        },
        "observation": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "string"},
                "category": {"type": "string"},
                "subject": {"type": "string"},
                "sender": {"type": "string"},
                "body": {"type": "string"},
                "context": {"type": "string", "nullable": True},
                "current_agent": {"type": "string"},
            },
        },
        "state": {
            "type": "object",
            "properties": {
                "current_ticket": {"type": "object", "nullable": True},
                "current_agent": {"type": "string", "nullable": True},
                "ticket_number": {"type": "integer"},
                "total_tickets": {"type": "integer"},
                "total_reward": {"type": "number"},
                "steps_on_current_ticket": {"type": "integer"},
                "is_done": {"type": "boolean"},
                "kb_entries_added": {"type": "integer"},
                "escalation_count": {"type": "integer"},
            },
        },
    }


@app.get("/tasks")
async def tasks() -> List[Dict[str, Any]]:
    """List all tasks with grader info — required by OpenEnv validator."""
    return [
        {
            "id": "ticket_triage",
            "name": "Ticket Triage",
            "description": "Classify incoming ticket: category, priority, and support tier.",
            "difficulty": "easy",
            "grader": {
                "type": "python",
                "path": "graders.py",
                "function": "grade_triage",
            },
        },
        {
            "id": "ticket_resolution",
            "name": "Ticket Resolution",
            "description": "Resolve IT support tickets through diagnosis and solution application.",
            "difficulty": "medium",
            "grader": {
                "type": "python",
                "path": "graders.py",
                "function": "grade_reply",
            },
        },
        {
            "id": "kb_contribution",
            "name": "Knowledge Base Contribution",
            "description": "Document novel solutions for future retrieval — self-improvement mechanism.",
            "difficulty": "hard",
            "grader": {
                "type": "python",
                "path": "graders.py",
                "function": "grade_kb_contribution",
            },
        },
    ]


@app.post("/mcp")
async def mcp(body: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    """MCP JSON-RPC endpoint — required by OpenEnv validator."""
    return {
        "jsonrpc": "2.0",
        "id": body.get("id", 1),
        "result": {
            "name": "HelpdeskEnv",
            "description": (
                "Multi-agent IT helpdesk simulation with triage, "
                "tiered support, SLA tracking, and knowledge base learning"
            ),
        },
    }


# ============================================================================
# Homepage
# ============================================================================

@app.get("/web", response_class=HTMLResponse)
async def home():
    """Serve the homepage."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HelpdeskEnv</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            padding: 40px;
            max-width: 600px;
            text-align: center;
        }
        h1 { color: #333; margin-bottom: 10px; font-size: 2.5em; }
        .emoji { font-size: 3em; display: block; margin: 20px 0; }
        p { color: #666; line-height: 1.8; margin: 15px 0; }
        .tasks { background: #f5f5f5; border-radius: 8px; padding: 20px; margin: 25px 0; text-align: left; }
        .tasks h2 { color: #333; margin-bottom: 15px; font-size: 1.2em; }
        .task-item { padding: 10px 0; border-bottom: 1px solid #ddd; }
        .task-item:last-child { border-bottom: none; }
        .task-item strong { color: #667eea; }
        .links { margin-top: 30px; display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; }
        a { display: inline-block; padding: 10px 20px; background: #667eea; color: white; text-decoration: none; border-radius: 5px; transition: background 0.3s; }
        a:hover { background: #764ba2; }
    </style>
</head>
<body>
    <div class="container">
        <span class="emoji">🎫</span>
        <h1>HelpdeskEnv</h1>
        <p><strong>Multi-Agent IT Helpdesk — OpenEnv Environment v2.0</strong></p>
        <p>4 specialized agents (Triage, L1, L2, L3) collaborate to resolve IT tickets with a self-improving Knowledge Base.</p>
        <div class="tasks">
            <h2>Tasks</h2>
            <div class="task-item"><strong>Ticket Triage</strong> — Classify category, priority, and support tier</div>
            <div class="task-item"><strong>Ticket Resolution</strong> — Diagnose and resolve IT support tickets</div>
            <div class="task-item"><strong>KB Contribution</strong> — Document novel solutions for future retrieval</div>
        </div>
        <div class="tasks">
            <h2>Agents</h2>
            <div class="task-item"><strong>Triage</strong> — Routes tickets to the right team</div>
            <div class="task-item"><strong>L1 Support</strong> — Handles simple issues (password resets, etc.)</div>
            <div class="task-item"><strong>L2 Support</strong> — Handles medium-complexity issues</div>
            <div class="task-item"><strong>L3 Support</strong> — Handles complex issues + writes KB articles</div>
        </div>
        <div class="links">
            <a href="/docs">API Docs</a>
            <a href="https://github.com/HR5206/emailenv">GitHub</a>
        </div>
    </div>
</body>
</html>"""


# ============================================================================
# Core HelpdeskEnv endpoints (thin wrapper)
# ============================================================================

@app.post("/reset", response_model=HelpdeskResetResponse)
async def reset(body: Optional[ResetRequest] = Body(None)):
    """Reset the environment and get the first ticket."""
    try:
        seed = body.seed if body else None
        num_tickets = body.num_tickets if body and body.num_tickets else 3
        logger.info(f"[START] task=helpdesk env=helpdeskenv model={MODEL_NAME}")
        result = _env.reset(seed=seed, num_tickets=num_tickets)
        return result
    except Exception as e:
        logger.error(f"[RESET ERROR] {str(e)}")
        raise


@app.post("/step")
async def step(action: HelpdeskAction = Body(...)):
    """Take a step in the environment with a HelpdeskAction.

    The server does NOT interpret the action — it passes it directly
    to HelpdeskEnv.step() which handles all routing and grading.
    """
    try:
        action_str = f"agent={action.agent_role.value} type={action.action_type}"
        result = _env.step(action)

        if isinstance(result, ErrorResponse):
            logger.info(
                f"[STEP] step=error action={action_str} reward=0.00 "
                f"done=false error={result.error}"
            )
            return result

        logger.info(
            f"[STEP] step={result.task_id} action={action_str} "
            f"reward={result.reward:.2f} done={str(result.done).lower()} error=null"
        )
        return result
    except Exception as e:
        logger.info(
            f"[STEP] step=unknown action=unknown reward=0.00 done=false error={str(e)}"
        )
        raise


@app.get("/state", response_model=HelpdeskEnvState)
async def state() -> HelpdeskEnvState:
    """Return the current environment state."""
    return _env.state()


# ============================================================================
# Knowledge Base endpoints
# ============================================================================

@app.get("/kb")
async def get_kb():
    """Return the current Knowledge Base contents and stats."""
    return {
        "entries": [e.model_dump() for e in _env.kb().get_all()],
        "stats": _env.kb().stats(),
    }


@app.post("/kb/search")
async def search_kb(body: dict = Body(...)):
    """Search the Knowledge Base."""
    query = body.get("query", "")
    results = _env.kb().search(query)
    return {"results": [r.model_dump() for r in results]}


# ============================================================================
# Entry point
# ============================================================================

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()