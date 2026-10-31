#!/usr/bin/env python3
"""sys12 MCP server — the System One / System Two switch as MCP tools.

Tools (all prefixed so they cannot collide with another server's tools):
  sys12_decide  → System One: choice / score / noul on Ollama Cloud
  sys12_ask     → System Two: chat completion on Ollama Cloud
  sys12_route   → the switch: NOUL gate on System One, escalate when deep
  sys12_browse  → browser navigation with System One picking each step

Every decision carries the escalate contract: if ``escalate`` is true for a
question, the label is a hint only and the caller decides. Decisions are cloud
only; there is no local model in the decision path. Run via run_mcp.sh (stdio).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

import sys12

INSTRUCTIONS = (
    "sys12 is a System One / System Two switch on Ollama Cloud. "
    "sys12_decide gives a fast structured judgment (choice, score, yes/no) from "
    "next-token probabilities in about a second. sys12_ask gives full reasoning and "
    "generation. sys12_route picks between them with a gate question. "
    "Always check the escalate flag: when it is true for a question, the returned "
    "label is a hint only and you decide instead. Redact MAC addresses, IP addresses, "
    "hostnames, and secrets before calling. A decision is not permission to change a system."
)

mcp = FastMCP("sys12", instructions=INSTRUCTIONS)

CFG = sys12.load_config()


def _safe(call, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run a sys12 call, returning a typed error rather than crashing the server.

    ``error_kind`` lets the caller tell a bad argument from an upstream failure:
    ``input`` never retries, ``upstream`` may, ``internal`` means a bug here.
    """
    try:
        return call(CFG, *args, **kwargs)
    except sys12.UpstreamError as exc:
        return {
            "error": str(exc),
            "error_kind": "upstream" if exc.retryable else "upstream_fatal",
            "status": exc.status,
        }
    except sys12.ShapeError as exc:
        return {"error": str(exc), "error_kind": "upstream_shape"}
    except sys12.Sys12Error as exc:
        return {"error": str(exc), "error_kind": "input"}
    except Exception as exc:  # a browser or SDK failure must not kill the server
        return {"error": f"{type(exc).__name__}: {exc}", "error_kind": "internal"}


@mcp.tool(
    annotations={
        "title": "System One decision",
        "readOnlyHint": True,
        "destructiveHint": False,
        "openWorldHint": True,
    }
)
def sys12_decide(
    state: str,
    questions: dict[str, Any],
    model: str | None = None,
    images: list[str] | None = None,
    notes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """System One: a fast structured judgment in one forward pass.

    Use when a quick closed judgment is needed: pick an option, place something
    on an ordered rubric, or answer yes/no. No text generation.

    Args:
        state: The situation, data, or task to judge. Facts only.
        questions: Mapping of question name to spec. Each spec needs
            "type" ("choice" | "score" | "noul"), optional "instructions", and
            "criteria": choice = {label: description}, score = [label, ...]
            low to high, noul = {"true": ..., "false": ...}. Up to 255 criteria.
        model: Override the decision model (default from config, cloud only).
        images: Base64 images (a data: prefix is optional) for a vision model.
        notes: One-line description per label, this call only. Nothing is stored.

    Returns:
        {"answers": {name: {...}}, "escalate": bool, "escalations": {...},
         "backend", "model", "latency_ms", "fallback_reason", "usage"}.
        Per answer: choice → choice, confidence, top; score → score, answer,
        legend, confidence; noul → noul, answer, confidence.
        Each answer has its own "escalate" and "escalate_reason". When escalate
        is true, ignore the label and decide yourself.
    """
    return _safe(sys12.decide, state, questions, model=model, images=images, notes=notes)


@mcp.tool(
    annotations={
        "title": "System Two chat",
        "readOnlyHint": True,
        "destructiveHint": False,
        "openWorldHint": True,
    }
)
def sys12_ask(
    prompt: str,
    model: str | None = None,
    system: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    """System Two: reasoning, long generation, and multi-step work.

    Use for explanations, code, plans, or any task that needs generated text.
    Slower than sys12_decide (seconds to minutes).

    Args:
        prompt: The task or question.
        model: Override the chat model (default from config).
        system: Optional system prompt.
        max_tokens: Optional output cap. Set this whenever the reply is short.
        temperature: Optional sampling temperature.

    Returns:
        {"model", "content", "reasoning", "usage"}.
    """
    return _safe(
        sys12.ask, prompt, model=model, system=system,
        max_tokens=max_tokens, temperature=temperature,
    )


@mcp.tool(
    annotations={
        "title": "System One/Two switch",
        "readOnlyHint": True,
        "destructiveHint": False,
        "openWorldHint": True,
    }
)
def sys12_route(
    task: str,
    state: str | None = None,
    questions: dict[str, Any] | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    """The switch: gate on System One, then answer or escalate to System Two.

    System One is asked a gate question — "can this NOT be settled by a quick
    structured judgment?" — and p(deep) at or above the threshold sends the task
    to System Two. Below it, System One answers the structured questions. A
    task with no questions goes straight to System Two, which costs one call less.

    Args:
        task: What must be decided or done.
        state: Extra facts (default: none).
        questions: Structured questions for System One when the gate says simple
            (same spec as sys12_decide).
        threshold: p(deep) escalation threshold 0..1 (default from config, 0.5).

    Returns:
        {"system": "one"|"two", "gate": {p_deep, escalate, threshold, uncertain},
         "result", "escalate"}. The threshold alone decides the route;
         gate.uncertain only reports that p(deep) sat inside 0.10-0.90, which is
         common and is not itself a reason to escalate. When system is "two",
         result is the chat reply. When system is "one", result is a sys12_decide
         answer — check its escalate.
    """
    return _safe(
        sys12.route, task, state=state, questions=questions, threshold=threshold
    )


_BROWSE_TRUTHY = {"1", "true", "yes", "on"}


def _browse_enabled() -> bool:
    """Read the browse opt-in. ``SYS12_ALLOW_BROWSE=0`` must mean off."""
    return os.environ.get("SYS12_ALLOW_BROWSE", "").strip().lower() in _BROWSE_TRUTHY


@mcp.tool(
    annotations={
        "title": "System One browser navigation",
        "readOnlyHint": False,
        "destructiveHint": False,
        "openWorldHint": True,
    }
)
def sys12_browse(
    url: str,
    goal: str,
    max_steps: int = 20,
    form: dict[str, str] | None = None,
    headless: bool = False,
    summarize: bool = False,
) -> dict[str, Any]:
    """Navigate a website with System One choosing each step.

    Playwright observes the page and executes actions; System One picks, per
    step, which element advances the goal or signals stop. About one to three
    seconds per step. One persistent browser is reused, so logins survive.

    Args:
        url: Starting URL.
        goal: What to find or accomplish. Drives element ranking and decisions.
        max_steps: Safety cap on observe-decide-act iterations (default 20).
        form: Optional {input label: value} for form fields System One selects.
        headless: Run the browser hidden (default: visible).
        summarize: Also run System Two over the final page to answer the goal.

    Returns:
        {"status": goal_reached|stuck|max_steps|needs_input, "stop_reason",
         "steps", "url", "title", "text", "answer"?}.
    """
    if not _browse_enabled():
        return {
            "error": (
                "sys12_browse is disabled. Set SYS12_ALLOW_BROWSE=1 in the MCP "
                "server environment and install the browser extra to enable it."
            )
        }
    return _safe(
        sys12.browse, url, goal, max_steps=max_steps, form=form,
        headless=headless, summarize=summarize,
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
