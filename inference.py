"""EmailEnv Inference Script

This script implements the standardized inference interface expected by the
benchmark harness. It:

- Reads configuration from environment variables (API endpoint, model name, etc.).
- Uses the OpenAI client for all LLM calls when credentials are provided.
- Interacts with the EmailEnv environment one step at a time.
- Logs progress to stdout using the required three-line format:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

The script is intentionally self‑contained so that the benchmark can simply run
`python inference.py` from the project root.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional

from openai import OpenAI

from emailenv import EmailEnv, Action


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# API / model configuration (must match the active inference setup).
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-5-nano")

# Primary API key for OpenAI / HF router style endpoints.
API_KEY: Optional[str] = (
    os.getenv("HF_TOKEN")
    or os.getenv("API_KEY")
    or os.getenv("OPENAI_API_KEY")
)

# Environment configuration.
TASK_NAME: str = os.getenv("EMAILENV_TASK", "spam_classification")
BENCHMARK: str = os.getenv("EMAILENV_BENCHMARK", "emailenv")

# Optional docker image name (not used directly here, but included to align
# with the generic OpenENV spec).
LOCAL_IMAGE_NAME: Optional[str] = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")

# Inference hyperparameters.
TEMPERATURE: float = float(os.getenv("EMAILENV_TEMPERATURE", "0.0"))
MAX_TOKENS: int = int(os.getenv("EMAILENV_MAX_TOKENS", "256"))
MAX_STEPS: int = int(os.getenv("EMAILENV_MAX_STEPS", "512"))

# Threshold in [0, 1] for deciding whether the episode is a success based on
# the normalized score.
SUCCESS_SCORE_THRESHOLD: float = float(os.getenv("EMAILENV_SUCCESS_THRESHOLD", "0.5"))


# ---------------------------------------------------------------------------
# Logging helpers (STDOUT contract)
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    """Log the beginning of an episode.

    Format: [START] task=<task_name> env=<benchmark> model=<model_name>
    """

    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Log a single environment step.

    Format: [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    """

    done_val = str(done).lower()
    error_val = error if error else "null"
    # Ensure there are no newlines within a line.
    action_clean = action.replace("\n", " ")
    error_clean = error_val.replace("\n", " ") if error_val is not None else "null"
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} "
        f"done={done_val} error={error_clean}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Log the end of an episode.

    Format: [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
    """

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} "
        f"rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


def _get_client() -> Optional[OpenAI]:
    """Create an OpenAI client if an API key is available.

    When no API key is provided, this returns ``None`` and the policy falls
    back to lightweight rule‑based behavior so the script can still run
    without network access. When a key *is* provided, all decisions are
    delegated to the model via the OpenAI Client, as required by the spec.
    """

    if not API_KEY:
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def _call_openai(client: OpenAI, system_prompt: str, user_prompt: str) -> str:
    """Call the chat completion API and return trimmed content.

    Any exception is surfaced to the caller, which can then decide how to
    recover. This keeps the core behavior simple for the benchmark harness.
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
    )
    choice = response.choices[0]
    content = choice.message.content or ""
    return content.strip()


def _parse_json(content: str) -> Dict:
    """Best‑effort JSON parsing that tolerates extra text around a JSON blob."""

    try:
        return json.loads(content)
    except Exception:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = content[start : end + 1]
            try:
                return json.loads(snippet)
            except Exception:
                return {}
        return {}


# ---------------------------------------------------------------------------
# Simple heuristic fallbacks (used only when no API key is configured)
# ---------------------------------------------------------------------------


def _heuristic_is_spam(subject: str, body: str) -> bool:
    text = (subject + " " + body).lower()
    spam_keywords = [
        "discount",
        "offer",
        "congratulations",
        "winner",
        "prize",
        "lottery",
        "free",
    ]
    return any(k in text for k in spam_keywords)


def _heuristic_priority(subject: str, body: str) -> str:
    text = (subject + " " + body).lower()
    if any(k in text for k in ["urgent", "critical", "unable", "failed", "error", "stuck", "cannot"]):
        return "high"
    if any(k in text for k in ["invoice", "reminder", "upgrade", "question", "request", "feature"]):
        return "medium"
    return "low"


def _heuristic_reply(subject: str, body: str) -> str:
    return (
        "Thank you for your email regarding '" + subject + "'. "
        "We have received your message and are looking into this. "
        "We will get back to you with an update as soon as possible."
    )


# ---------------------------------------------------------------------------
# Action builders for each EmailEnv task
# ---------------------------------------------------------------------------


def _build_spam_action(client: Optional[OpenAI], subject: str, body: str, sender: str) -> Action:
    if client is None:
        is_spam = _heuristic_is_spam(subject, body)
        return Action(type="classify_spam", is_spam=is_spam)

    system_prompt = (
        "You classify emails as spam or not spam. "
        "Respond only with JSON: {\"type\":\"classify_spam\",\"is_spam\":true|false}."
    )
    user_prompt = f"Subject: {subject}\nSender: {sender}\nBody: {body}"

    try:
        content = _call_openai(client, system_prompt, user_prompt)
        data = _parse_json(content)
        is_spam = bool(data.get("is_spam", False))
    except Exception:
        is_spam = _heuristic_is_spam(subject, body)

    return Action(type="classify_spam", is_spam=is_spam)


def _build_priority_action(client: Optional[OpenAI], subject: str, body: str, sender: str) -> Action:
    if client is None:
        priority_raw = _heuristic_priority(subject, body)
    else:
        system_prompt = (
            "You assign a priority level to customer emails. "
            "Respond only with JSON: {\"type\":\"set_priority\",\"priority\":\"low\"|\"medium\"|\"high\"}."
        )
        user_prompt = f"Subject: {subject}\nSender: {sender}\nBody: {body}"
        try:
            content = _call_openai(client, system_prompt, user_prompt)
            data = _parse_json(content)
            priority_raw = str(data.get("priority", "medium")).lower()
        except Exception:
            priority_raw = _heuristic_priority(subject, body)

    if priority_raw not in {"low", "medium", "high"}:
        priority_raw = "medium"

    return Action(type="set_priority", priority=priority_raw)


def _build_reply_action(client: Optional[OpenAI], subject: str, body: str, sender: str) -> Action:
    if client is None:
        reply_text = _heuristic_reply(subject, body)
        return Action(type="generate_reply", reply_text=reply_text)

    system_prompt = (
        "You write professional, concise, and helpful customer support replies. "
        "Respond only with JSON: {\"type\":\"generate_reply\",\"reply_text\":\"...\"}."
    )
    user_prompt = f"Subject: {subject}\nSender: {sender}\nBody: {body}"

    try:
        content = _call_openai(client, system_prompt, user_prompt)
        data = _parse_json(content)
        reply_text = str(data.get("reply_text", "")).strip()
        if not reply_text:
            reply_text = _heuristic_reply(subject, body)
    except Exception:
        reply_text = _heuristic_reply(subject, body)

    return Action(type="generate_reply", reply_text=reply_text)


def _format_action_for_log(action: Action) -> str:
    """Create a compact, one‑line string representation of an Action.

    This is used only for logging; it does not affect environment behavior.
    """

    if action.type == "classify_spam":
        return f"classify_spam(is_spam={str(bool(action.is_spam)).lower()})"
    if action.type == "set_priority":
        return f"set_priority(priority={action.priority})"
    if action.type == "generate_reply":
        # Truncate long replies in logs to keep lines readable.
        snippet = (action.reply_text or "").replace("\n", " ")
        if len(snippet) > 80:
            snippet = snippet[:77] + "..."
        return f"generate_reply(reply_text={snippet})"
    return "skip"


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def run_episode() -> None:
    """Run a single EmailEnv episode and emit the required logs."""

    client = _get_client()
    env = EmailEnv()

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        observation = env.reset(task=TASK_NAME)  # type: ignore[arg-type]
        done = False

        while not done and steps_taken < MAX_STEPS:
            email = observation.email

            if email is None:
                action = Action(type="skip")
            elif TASK_NAME == "spam_classification":
                action = _build_spam_action(client, email.subject, email.body, email.sender)
            elif TASK_NAME == "email_prioritization":
                action = _build_priority_action(client, email.subject, email.body, email.sender)
            elif TASK_NAME == "reply_generation":
                action = _build_reply_action(client, email.subject, email.body, email.sender)
            else:
                # Unknown task: safest is to no‑op.
                action = Action(type="skip")

            action_str = _format_action_for_log(action)

            try:
                observation, reward, done, _info = env.step(action)
                reward_value = float(getattr(reward, "value", 0.0) or 0.0)
                rewards.append(reward_value)
                steps_taken += 1
                log_step(step=steps_taken, action=action_str, reward=reward_value, done=done, error=None)
            except Exception as step_exc:  # pragma: no cover - defensive
                steps_taken += 1
                log_step(
                    step=steps_taken,
                    action=action_str,
                    reward=0.0,
                    done=True,
                    error=str(step_exc),
                )
                done = True

        # Compute normalized score in [0, 1]. EmailEnv ensures each per‑step
        # reward is already in [0, 1], so we simply average across steps.
        if rewards:
            score = sum(rewards) / float(len(rewards))
            if score < 0.0:
                score = 0.0
            if score > 1.0:
                score = 1.0
        else:
            score = 0.0

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:  # pragma: no cover - defensive
        # If reset or any outer logic fails, we still need to emit an [END]
        # line. No steps were taken, and success is False.
        steps_taken = max(steps_taken, 0)
        rewards.clear()
        score = 0.0
        success = False
        # Emit a synthetic step log to capture the error for debugging while
        # still respecting the contract that [STEP] lines follow env.step().
        # Here we skip that and rely on the [END] line; additional debug
        # information goes to stderr.
        print(f"[DEBUG] Episode failed: {exc}", file=sys.stderr, flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    run_episode()


if __name__ == "__main__":
    main()
