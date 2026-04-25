"""Agent prompt configurations for the multi-agent helpdesk."""

from agents.triage import TRIAGE_PROMPT, build_triage_prompt
from agents.l1_agent import L1_PROMPT, build_l1_prompt
from agents.l2_agent import L2_PROMPT, build_l2_prompt
from agents.l3_agent import L3_PROMPT, build_l3_prompt

__all__ = [
    "TRIAGE_PROMPT", "build_triage_prompt",
    "L1_PROMPT", "build_l1_prompt",
    "L2_PROMPT", "build_l2_prompt",
    "L3_PROMPT", "build_l3_prompt",
]
