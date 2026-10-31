#!/usr/bin/env python3
"""sys12 MCP server — System One / System Two switch as OpenCode tools.

Exposes:
  decide(state, questions)  → System One (/v1/systemone): fast structured judgment
  ask(prompt, ...)          → System Two (/v1/chat/completions): reasoning + generation
  route(task, ...)          → the switch: gate on System One, escalate if deep
  status()                  → local oMLX loaded models + memory
  browse(url, goal, ...)    → System One-driven Playwright navigation

Models listed in config.cloud.models route to Ollama Cloud; everything else
route to the local oMLX server. Run via run_mcp.sh (stdio).
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

import sys12

mcp = FastMCP("sys12")
CFG = sys12.load_config()


def _safe(fn, *args, **kwargs):
    """Run a sys12 call, returning {"error": ...} instead of crashing the server."""
    try:
        return fn(*args, **kwargs)
    except sys12.Sys12Error as e:
        return {"error": str(e)}


_RESERVED = {"type", "instructions"}


def _normalize_questions(questions: dict[str, Any]) -> dict[str, Any]:
    """Tolerate common question-shape mistakes.

    - options/true-false placed on the question instead of in `criteria`
    - score given as a dict (values become the labels)
    - choice given as a list (labels become A, B, C, ...)
    """
    out: dict[str, Any] = {}
    for name, q in questions.items():
        if not isinstance(q, dict):
            out[name] = q
            continue
        q = dict(q)
        if "criteria" not in q:
            extra = {k: v for k, v in q.items() if k not in _RESERVED}
            if extra:
                q["criteria"] = extra  # works for noul {true, false} and choice {A: ...}
                for k in extra:
                    q.pop(k)
        elif isinstance(q["criteria"], dict) and q.get("type") == "score":
            q["criteria"] = list(q["criteria"].values())
        elif isinstance(q["criteria"], list) and q.get("type") == "choice":
            q["criteria"] = {chr(65 + i): c for i, c in enumerate(q["criteria"][:26])}
        out[name] = q
    return out


@mcp.tool
def decide(
    state: str,
    questions: dict[str, Any],
    model: str | None = None,
    images: list[str] | None = None,
) -> dict[str, Any]:
    """System One: fast structured decision (~1s, no text generation).

    Use when a quick judgment is needed: pick an option, score something,
    or answer a yes/no. Answers come from next-token probabilities.

    Args:
        state: Decision context — the situation, data, or task to judge.
        questions: Mapping of question name to spec. Spec has a required
            "type" ("choice" | "score" | "noul"), optional "instructions",
            and "criteria": choice = {label: description} (up to 26),
            score = [label, ...] (up to 26), noul = {"true": ..., "false": ...}.
        model: Override the System One model (default from config).
        images: Base64-encoded images (no data-URL prefix) for vision models.

    Returns:
        answers keyed by question name: choice → chosen label + probabilities,
        score → weighted score + legend, noul → p(true).
    """
    return _safe(sys12.decide, CFG, state, _normalize_questions(questions),
                 model=model, images=images)


@mcp.tool
def ask(
    prompt: str,
    model: str | None = None,
    system: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    """System Two: full reasoning and generation via chat completion.

    Use for long reasoning, multi-step work, explanations, code, or any
    task that needs a generated answer. Slower (seconds to minutes).

    Args:
        prompt: The task or question for the model.
        model: Override the System Two model (default from config).
        system: Optional system prompt.
        max_tokens: Optional output token cap.
        temperature: Optional sampling temperature.

    Returns:
        {"model", "content", "reasoning" (if any), "usage"}.
    """
    return _safe(sys12.ask, CFG, prompt, model=model, system=system,
                 max_tokens=max_tokens, temperature=temperature)


@mcp.tool
def route(
    task: str,
    state: str | None = None,
    questions: dict[str, Any] | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    """The System One/Two switch: automatic per-task routing.

    Asks System One a NOUL gate: "does this task need deep reasoning, long
    generation, or multi-step work?" If p(deep) >= threshold (default from
    config) the task escalates to System Two; otherwise the provided
    structured questions are answered by System One (if none were provided,
    only the gate verdict is returned).

    Args:
        task: What must be decided or done.
        state: Extra context (default: the task itself).
        questions: Structured questions to answer with System One when the
            gate says "simple" (same spec as decide()).
        threshold: p(deep) escalation threshold, 0..1 (default from config).

    Returns:
        {"system": "one"|"two", "gate": {p_deep, escalate, threshold}, "result"}.
    """
    return _safe(sys12.route, CFG, task, state=state,
                 questions=_normalize_questions(questions) if questions else None,
                 threshold=threshold)


@mcp.tool
def status() -> dict[str, Any]:
    """oMLX server status: which models are loaded, memory usage."""
    return _safe(sys12.status, CFG)


@mcp.tool
def browse(
    url: str,
    goal: str,
    max_steps: int = 20,
    form: dict[str, str] | None = None,
    headless: bool = False,
    summarize: bool = False,
) -> dict[str, Any]:
    """Navigate a website with System One as the fast per-step decision maker.

    A tight in-process loop: Playwright observes the page and executes actions,
    System One (~1 s, logits) picks which element advances the goal or signals
    stop. Much faster than a 27B agent generating each step. One persistent
    browser (system Chrome) is reused across calls, so cookies/login persist.

    Args:
        url: Starting URL.
        goal: What to find or accomplish (drives element ranking + decisions).
        max_steps: Safety cap on observe-decide-act iterations (default 20).
        form: Optional mapping of input label → value, for inputs the model picks.
        headless: Run Chrome hidden (default: visible window you can watch).
        summarize: Also run System Two over the final page text to produce an answer.

    Returns:
        {status: goal_reached|stuck|max_steps|needs_input, stop_reason, steps,
        url, title, text (truncated page content), answer? (if summarize)}.
    """
    import browser

    try:
        return browser.browse(CFG, url, goal, max_steps=max_steps, form=form,
                              headless=headless, summarize=summarize)
    except Exception as e:  # Playwright/oMLX failures must not kill the server
        return {"error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    mcp.run()
