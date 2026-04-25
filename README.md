---
title: HelpdeskEnv
emoji: 🎫
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
license: apache-2.0
tags:
  - openenv
  - multi-agent
  - helpdesk
  - self-improving
  - knowledge-base
  - IT-support
  - reinforcement-learning
---

# 🎫 HelpdeskEnv — Multi-Agent IT Helpdesk

A **multi-agent OpenEnv-compatible RL environment** where four specialized AI
agents (Triage, L1, L2, L3) collaborate to resolve IT support tickets. Features
SLA tracking, escalation chains, and a **self-improving Knowledge Base** that
grows across episodes.

---

## Motivation

IT helpdesks are high-value, real-world workflows with natural multi-agent
structure. Training RL agents in this setting teaches them to:

- **Triage** incoming issues by category, priority, and required expertise
- **Collaborate** across specialized support tiers with escalation decisions
- **Plan ahead** with SLA constraints and efficiency targets
- **Self-improve** by documenting novel solutions for future retrieval

This environment addresses all **four hackathon themes**:

| Theme | How HelpdeskEnv Implements It |
|-------|------|
| **Multi-Agent Interactions** | 4 agents (Triage → L1 → L2 → L3) with routing and escalation |
| **Long-Horizon Planning** | Ticket queues with SLA timers; agents must plan action sequences |
| **World Modeling** | Knowledge Base state + customer satisfaction + SLA clock |
| **Self-Improving Systems** | KB persists across episodes; agents learn from documented solutions |

---

## Architecture

```text
Ticket Queue ──→ [Triage Agent] ──→ classifies category / priority / tier
                      │
             ┌────────┼────────┐
             ▼        ▼        ▼
        [L1 Agent] [L2 Agent] [L3 Agent]
             │        │        │
             └────────┼────────┘
                      ▼
             [Knowledge Base]  ←── persists across episodes (self-improvement)
                      ▼
             [Resolution + Customer Response]
```

---

## Action & Observation Spaces

### Action Space

All actions use the `HelpdeskAction` model:

| Field | Type | Description |
|-------|------|-------------|
| `ticket_id` | `str` | Must match the current ticket |
| `agent_role` | `"triage" \| "l1" \| "l2" \| "l3"` | Which agent is acting |
| `action_type` | `str` | One of the action types below |
| `action_value` | `str` | The agent's payload (JSON for triage, text for others) |

**Available action types:**

| Action Type | Available To | Description |
|-------------|-------------|-------------|
| `classify_category` | Triage | JSON with category, priority, tier |
| `search_kb` | L1, L2 | Query the Knowledge Base |
| `apply_solution` | L1 | Apply a KB-matched solution |
| `apply_fix` | L2 | Apply a technical fix |
| `apply_complex_fix` | L3 | Apply expert-level multi-step fix |
| `diagnose` | L2 | Investigate the issue |
| `deep_diagnose` | L3 | Deep root cause analysis |
| `request_info` | L2, L3 | Request more info from customer |
| `respond_to_customer` | All | Send resolution response |
| `escalate` | L1, L2 | Pass to next tier |
| `write_kb_entry` | L3 | Document solution in KB |

### Observation Space

When a ticket is active, the agent sees:

| Field | Type | Description |
|-------|------|-------------|
| `ticket_id` | `str` | Unique ticket identifier |
| `category` | `str` | Ticket category |
| `subject` | `str` | Ticket subject line |
| `sender` | `str` | Customer email |
| `body` | `str` | Full ticket description |
| `context` | `str \| null` | Role-specific instructions |

### State Space

Environment state is tracked by `HelpdeskEnvState`:

| Field | Type | Description |
|-------|------|-------------|
| `current_ticket` | `Ticket \| null` | Active ticket |
| `current_agent` | `str \| null` | Active agent role |
| `ticket_number` | `int` | Current ticket index |
| `total_tickets` | `int` | Total tickets in episode |
| `total_reward` | `float` | Cumulative reward |
| `steps_on_current_ticket` | `int` | Steps taken on current ticket |
| `is_done` | `bool` | Whether episode is complete |
| `kb_entries_added` | `int` | KB articles written this session |
| `escalation_count` | `int` | Total escalations |

---

## Tasks

| ID | Name | Difficulty | Description |
|----|------|-----------|-------------|
| `ticket_triage` | Ticket Triage | 🟢 Easy | Classify category, priority, and support tier |
| `ticket_resolution` | Ticket Resolution | 🟡 Medium | Diagnose and resolve IT support tickets |
| `kb_contribution` | KB Contribution | 🔴 Hard | Document novel solutions for future retrieval |

### Ticket Scenarios

| Ticket | Category | Tier | Priority | SLA | KB Required |
|--------|----------|------|----------|-----|-------------|
| Password Reset | `password_reset` | L1 | Medium | 3 steps | No |
| Software Install | `software_install` | L2 | Medium | 4 steps | No |
| Network Outage | `network_outage` | L3 | Critical | 5 steps | Yes |
| Data Recovery | `data_recovery` | L3 | Critical | 4 steps | Yes |
| Novel Issue | `novel_issue` | L3 | High | 6 steps | Yes |

---

## Reward System

Each ticket's reward is a weighted combination:

```text
ticket_reward = weighted_average(
    resolution_correct   × 0.30    ← does the fix match ground truth?
    response_quality     × 0.20    ← politeness + relevance + length
    efficiency           × 0.20    ← fewer unnecessary escalations
    sla_compliance       × 0.15    ← resolved before deadline
    kb_contribution      × 0.15    ← did agent document the solution?
)
```

### Grader Details

| Grader | Weight | Criteria |
|--------|--------|----------|
| **Triage** | category × 0.4, priority × 0.3, tier × 0.3 | Exact match + distance-based partial credit for priority |
| **Efficiency** | SLA × 0.6, escalation × 0.4 | -0.25 per step over SLA; -0.2 per unnecessary escalation |
| **KB Contribution** | relevance × 0.35, length × 0.30, specificity × 0.35 | Keywords like "resolved", "root cause", "steps" boost score |
| **Reply Quality** | politeness × 0.40, length × 0.30, relevance × 0.30 | Reuses Round 1 reply grader |

---

## Self-Improvement Mechanism

The Knowledge Base is the key self-improvement mechanism:

1. **Episode 1**: KB starts with 2 seed entries (password reset, software install)
2. **During episodes**: L3 agents write KB articles for novel issues
3. **Future episodes**: L1/L2 agents search the KB and find solutions written by previous L3 agents
4. **Measurable**: Track KB size, retrieval hit rate, and score improvement over time

```text
Episode 1: KB=2  → L1 can solve password_reset from KB
Episode 2: KB=5  → L1/L2 can now solve previously-novel issues from KB
Episode 5: KB=8+ → Most tickets can be resolved at lower tiers using KB
```

---

## Setup & Usage

### Install

```bash
pip install -r requirements.txt
```

### Run the Server (Local)

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### HTTP API

```bash
# Health check
curl http://localhost:7860/health

# Reset environment
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"seed": 42, "num_tickets": 3}'

# Take a step (triage example)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"ticket_id":"ticket_001","agent_role":"triage","action_type":"classify_category","action_value":"{\"category\":\"password_reset\",\"priority\":\"medium\",\"tier\":\"l1\"}"}'

# Get current state
curl http://localhost:7860/state

# View Knowledge Base
curl http://localhost:7860/kb

# Search Knowledge Base
curl -X POST http://localhost:7860/kb/search \
  -H "Content-Type: application/json" \
  -d '{"query": "password reset locked"}'
```

### Run Inference Agent

```bash
# With LLM
export OPENAI_API_KEY=sk-...
export MODEL_NAME=gpt-5-nano
python inference.py

# Heuristic only (no API key needed)
python inference.py
```

On Windows PowerShell:

```powershell
$Env:OPENAI_API_KEY = "sk-..."
$Env:MODEL_NAME = "gpt-5-nano"
python inference.py
```

### Run Baseline Evaluation

```bash
python baseline.py
```

### Run Multi-Episode Self-Improvement Demo

```bash
python demo.py
```

### Docker / Hugging Face Spaces

```bash
docker build -t helpdeskenv .
docker run -p 7860:7860 helpdeskenv
```

---

## Baseline Scores

Heuristic baseline across all 5 ticket scenarios (seed=42):

| Metric | Value |
|--------|-------|
| Total steps | ~13 |
| Avg reward | ~0.55 |
| KB entries (start) | 2 |
| KB entries (end) | 5 |
| Escalations | 0 |

Multi-episode self-improvement (5 episodes):

| Episode | KB Size | Avg Reward | Improvement |
|---------|---------|------------|-------------|
| 1 | 2 → 5 | ~0.55 | baseline |
| 2 | 5 → 8 | ~0.58 | +5% |
| 3 | 8+ | ~0.60 | +9% |
| 5 | 10+ | ~0.62 | +13% |

---

## Project Structure

```text
emailenv/
├── models.py              # All Pydantic models (Round 1 + Round 2)
├── tasks.py               # Email scenarios + Ticket scenarios
├── graders.py             # All graders (spam, priority, reply, triage, efficiency, KB)
├── knowledge_base.py      # KBEntry model + KnowledgeBase class
├── emailenv_class.py      # Round 1 EmailEnv class (preserved)
├── helpdeskenv_class.py   # Round 2 HelpdeskEnv class (multi-agent)
├── agents/
│   ├── __init__.py        # Agent prompt exports
│   ├── triage.py          # Triage agent prompt + builder
│   ├── l1_agent.py        # L1 agent prompt + builder
│   ├── l2_agent.py        # L2 agent prompt + builder
│   └── l3_agent.py        # L3 agent prompt + builder
├── server/
│   ├── __init__.py
│   └── app.py             # FastAPI server (all OpenEnv endpoints)
├── inference.py           # Multi-agent inference loop + heuristics
├── baseline.py            # Baseline evaluation script
├── demo.py                # Multi-episode self-improvement demo
├── client.py              # HTTP client helpers
├── __init__.py            # Package exports
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata (v2.0.0)
├── requirements.txt       # Dependencies
├── Dockerfile             # Docker build for HF Spaces
├── server.py              # Shim for server.app:app
└── README.md              # This file
```

---

## OpenEnv API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check → `{"status": "healthy"}` |
| `/metadata` | GET | Environment name, version, description |
| `/schema` | GET | Action/observation/state JSON schemas |
| `/tasks` | GET | Task list with grader info |
| `/reset` | POST | Start new episode, get first ticket |
| `/step` | POST | Submit action, get reward + feedback |
| `/state` | GET | Current environment state |
| `/kb` | GET | Knowledge Base contents + stats |
| `/kb/search` | POST | Search KB by query |
| `/mcp` | POST | MCP JSON-RPC endpoint |
| `/web` | GET | HTML homepage |
| `/docs` | GET | Swagger API documentation |

---

## License

Apache-2.0
