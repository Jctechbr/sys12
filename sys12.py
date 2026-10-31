#!/usr/bin/env python3
"""sys12 — System One / System Two switch on Ollama Cloud.

System One  → POST {cloud}/v1/systemone        structured judgments: choice / score / noul
                                               next-token probabilities, no generation, ~1 s
System Two  → POST {cloud}/v1/chat/completions  full generation, reasoning, long context

Both sides are cloud only. There is no local model in the decision path.

Ollama Cloud serves /v1/systemone for decision models (``clef``). If that call
fails in a way worth retrying, ``decide`` falls back to one deterministic chat
completion that must reply with JSON, parsed into the same answer shape. That
fallback carries no probabilities, so its answers always escalate.

Every answer carries the escalate contract:
  choice / score  confidence below 0.55, or missing          → escalate
  noul            yes_probability strictly inside 0.10–0.90  → escalate
When escalate is true the caller decides and ignores the label.

CLI:
  sys12 decide --state "ctx" --q name:type:criteria[|criteria...] [--q ...] [--image PATH] [--json]
  sys12 ask "prompt" [--model M] [--max-tokens N] [--temperature T] [--json]
  sys12 route "task" [--state ctx] [--q ...] [--threshold P] [--json]
  sys12 status [--json]
  sys12 browse URL --goal "..." [--max-steps N] [--form JSON] [--headless] [--summarize] [--json]
"""

from __future__ import annotations

import argparse
import base64
import binascii
import getpass
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

DEFAULT_CONFIG: dict[str, Any] = {
    "cloud": {
        "base_url": "https://ollama.com",
        "decision_model": "clef",
        "chat_model": "deepseek-v4.1-flash:1017",
        "api_key_env": "OLLAMA_API_KEY",
        "keychain_service": "ollama-clef",
        "api_key_file": "~/.config/sys12/ollama.env",
    },
    "route_gate_threshold": 0.5,
    "decision_timeout": 60,
    "chat_timeout": 300,
    "omlx": {
        "base_url": "http://127.0.0.1:8000",
        "api_key_file": "~/Library/Application Support/omlx/.env",
    },
}

DECISION_PATH = "/v1/systemone"
CHAT_PATH = "/v1/chat/completions"

# Ollama Cloud accepts up to 255 criteria on a decision question.
MAX_CRITERIA = 255
# Confidence at or above this stands. Below it, the chat model decides.
CHOICE_CONFIDENCE_MIN = 0.55
# yes/no stands only outside this band. Inside it, the chat model decides.
YES_UNSURE_LOW = 0.10
YES_UNSURE_HIGH = 0.90
YES_THRESHOLD = 0.5

QUESTION_TYPES = {"choice", "score", "noul"}

GATE_INSTRUCTIONS = (
    "Judge the task in the state below. Answer true if it CANNOT be settled "
    "by a quick structured judgment (a few options, a short score, or yes/no) "
    "and needs deep reasoning, long-form text, or multi-step work instead. "
    "Answer false if a fast structured judgment suffices."
)
GATE_TRUE = "yes — deep reasoning / long generation / multi-step work required"
GATE_FALSE = "no — a quick structured judgment suffices"

_DEFAULT_INSTRUCTIONS = {
    "choice": "Pick the single option that best fits the situation.",
    "noul": "Judge, from the situation, whether this statement is true.",
    "score": "Rate the situation against this ordered rubric.",
}

_RESERVED_QUESTION_KEYS = {"type", "instructions", "criteria"}

# The transport signature tests inject: (url, body, token, timeout) -> payload.
Transport = Callable[[str, dict, str, float], dict]


# --------------------------------------------------------------------------- errors


class Sys12Error(RuntimeError):
    """A sys12 call could not be completed."""


class UpstreamError(Sys12Error):
    """The cloud answered with an error, or could not be reached."""

    def __init__(self, status: int | None, detail: str, retryable: bool):
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.retryable = retryable


class ShapeError(Sys12Error):
    """A 2xx response did not carry the answer fields we expect."""


# --------------------------------------------------------------------------- config


def load_config() -> dict[str, Any]:
    """Defaults, then config.json, then environment overrides."""
    cfg: dict[str, Any] = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
    if CONFIG_PATH.exists():
        try:
            loaded = json.loads(CONFIG_PATH.read_text())
        except json.JSONDecodeError as exc:
            print(f"warn: bad {CONFIG_PATH}: {exc}", file=sys.stderr)
        else:
            for key, value in loaded.items():
                if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                    cfg[key].update(value)
                else:
                    cfg[key] = value
    cloud = cfg["cloud"]
    for env_name, key in (
        ("SYS12_CLOUD_BASE_URL", "base_url"),
        ("SYS12_DECISION_MODEL", "decision_model"),
        ("SYS12_CHAT_MODEL", "chat_model"),
    ):
        if os.environ.get(env_name):
            cloud[key] = os.environ[env_name]
    for env_name, key in (
        ("SYS12_ROUTE_THRESHOLD", "route_gate_threshold"),
        ("SYS12_DECISION_TIMEOUT", "decision_timeout"),
        ("SYS12_CHAT_TIMEOUT", "chat_timeout"),
    ):
        raw = os.environ.get(env_name, "").strip()
        if raw:
            cfg[key] = float(raw)
    return cfg


def cloud_base(cfg: dict[str, Any]) -> str:
    return (cfg["cloud"].get("base_url") or "https://ollama.com").rstrip("/")


def decision_model(cfg: dict[str, Any]) -> str:
    return cfg["cloud"].get("decision_model") or "clef"


def chat_model(cfg: dict[str, Any]) -> str:
    return cfg["cloud"].get("chat_model") or "deepseek-v4.1-flash:1017"


def _key_from_file(path: str, name: str) -> str:
    expanded = os.path.expanduser(path or "")
    if not expanded or not Path(expanded).is_file():
        return ""
    try:
        lines = Path(expanded).read_text().splitlines()
    except OSError:
        return ""
    for line in lines:
        line = line.strip()
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def cloud_key(cfg: dict[str, Any] | None = None) -> str:
    """Environment first, then the macOS Keychain, then an optional key file.

    The Keychain item is the default on this Mac: service ``ollama-clef``,
    account the login user, secret the Cursor API Key from the Bitwarden item
    "Ollama (Corporativo)". A machine without that item can export
    ``OLLAMA_API_KEY`` or drop a key file at ``cloud.api_key_file``.
    """
    cfg = cfg or load_config()
    cloud = cfg["cloud"]
    env_name = cloud.get("api_key_env") or "OLLAMA_API_KEY"
    from_env = os.environ.get(env_name, "").strip()
    if from_env:
        return from_env

    service = cloud.get("keychain_service") or "ollama-clef"
    try:
        proc = subprocess.run(
            ["security", "find-generic-password", "-s", service,
             "-a", getpass.getuser(), "-w"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        proc = None
    if proc is not None and proc.returncode == 0:
        key = proc.stdout.strip()
        if key:
            return key

    from_file = _key_from_file(cloud.get("api_key_file") or "", env_name)
    if from_file:
        return from_file

    raise Sys12Error(
        f"No cloud API key. Set {env_name}, store the Keychain item "
        f"{service} (account {getpass.getuser()}), or write "
        f"{cloud.get('api_key_file')}."
    )


def omlx_key(cfg: dict[str, Any]) -> str:
    return _key_from_file((cfg.get("omlx") or {}).get("api_key_file") or "",
                          "OMLX_API_KEY")


# --------------------------------------------------------------------------- http


def _redact(text: str, secrets: list[str]) -> str:
    cleaned = text or ""
    for secret in secrets:
        if secret:
            cleaned = cleaned.replace(secret, "[redacted]")
    return cleaned[:400]


def post_json(url: str, body: dict, token: str, timeout: float) -> dict:
    """POST a JSON body. Raises UpstreamError, tagged retryable or not."""
    headers = {"content-type": "application/json", "accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        # 401/402/403 are credential or billing problems: retrying the same key
        # through the chat fallback would only fail twice. 400 is a bad request.
        retryable = exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
        raise UpstreamError(exc.code, f"HTTP {exc.code}: {detail}", retryable) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise UpstreamError(None, str(reason), True) from exc
    if not isinstance(payload, dict):
        raise UpstreamError(None, "response was not a JSON object", True)
    return payload


def get_json(url: str, token: str, timeout: float) -> dict:
    headers = {"accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise UpstreamError(exc.code, f"HTTP {exc.code}: {detail}", False) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise UpstreamError(None, str(reason), True) from exc
    if not isinstance(payload, dict):
        raise UpstreamError(None, "response was not a JSON object", True)
    return payload


def _call(
    cfg: dict[str, Any],
    path: str,
    body: dict[str, Any],
    timeout: float,
    transport: Transport | None = None,
) -> dict[str, Any]:
    token = cloud_key(cfg)
    url = cloud_base(cfg) + path
    runner = transport or post_json
    try:
        return runner(url, body, token, timeout)
    except UpstreamError as exc:
        raise UpstreamError(exc.status, _redact(exc.detail, [token]), exc.retryable) from None


# --------------------------------------------------------------------------- questions


def normalize_questions(questions: dict[str, Any]) -> dict[str, Any]:
    """Put a question dict into the shape /v1/systemone expects.

    Tolerates the usual mistakes: options placed beside ``type`` instead of in
    ``criteria``; a score given as a dict; a choice given as a list.
    """
    if not isinstance(questions, dict) or not questions:
        raise Sys12Error("questions must be a non-empty mapping of name to spec")
    out: dict[str, Any] = {}
    for name, spec in questions.items():
        if not str(name).strip():
            raise Sys12Error("question name cannot be blank")
        if isinstance(spec, str):
            spec = {"type": spec}
        if not isinstance(spec, dict):
            raise Sys12Error(f"question '{name}' must be a mapping")
        q = dict(spec)
        qtype = str(q.get("type") or "choice").strip().lower()
        qtype = "noul" if qtype in {"yes_no", "noul", "bool"} else qtype
        if qtype not in QUESTION_TYPES:
            raise Sys12Error(
                f"question '{name}' has type '{q.get('type')}'; "
                "use choice, score, or noul"
            )
        if "criteria" not in q:
            extra = {k: v for k, v in q.items() if k not in _RESERVED_QUESTION_KEYS}
            if extra:
                q["criteria"] = extra
                for key in extra:
                    q.pop(key)
        criteria = q.get("criteria")

        if qtype == "noul":
            if criteria is None:
                criteria = {}
            if isinstance(criteria, (list, tuple)):
                labels = [str(c) for c in criteria]
                criteria = {"true": labels[0] if labels else "true",
                            "false": labels[1] if len(labels) > 1 else "false"}
            if not isinstance(criteria, dict):
                raise Sys12Error(f"question '{name}': noul criteria must be a mapping")
            q["criteria"] = {"true": str(criteria.get("true", "true")),
                             "false": str(criteria.get("false", "false"))}
        elif qtype == "score":
            if isinstance(criteria, dict):
                criteria = [str(v) for v in criteria.values()]
            if not isinstance(criteria, (list, tuple)):
                raise Sys12Error(f"question '{name}': score needs an ordered list")
            labels = [str(c) for c in criteria]
            _check_count(name, labels)
            q["criteria"] = labels
        else:
            if isinstance(criteria, (list, tuple)):
                criteria = {chr(65 + i): str(c) for i, c in enumerate(criteria)}
            if not isinstance(criteria, dict):
                raise Sys12Error(
                    f"question '{name}': choice needs a mapping of label to description"
                )
            labels = [str(k) for k in criteria]
            _check_count(name, labels)
            q["criteria"] = {str(k): str(v) for k, v in criteria.items()}

        q["type"] = qtype
        if not str(q.get("instructions") or "").strip():
            q["instructions"] = _DEFAULT_INSTRUCTIONS[qtype]
        out[str(name)] = q
    return out


def _check_count(name: str, labels: list[str]) -> None:
    if len(labels) < 2:
        raise Sys12Error(f"question '{name}' needs at least two criteria")
    if len(labels) > MAX_CRITERIA:
        raise Sys12Error(f"question '{name}' accepts at most {MAX_CRITERIA} criteria")
    if len(set(labels)) != len(labels):
        raise Sys12Error(f"question '{name}' has duplicate criteria")


def apply_notes(questions: dict[str, Any], notes: dict[str, str] | None) -> dict[str, Any]:
    """Overlay one-line descriptions for this call only. Nothing is stored."""
    if not notes:
        return questions
    out: dict[str, Any] = {}
    for name, q in questions.items():
        q = dict(q)
        criteria = q.get("criteria")
        if q["type"] == "choice" and isinstance(criteria, dict):
            updated = dict(criteria)
            for label, text in notes.items():
                if label in updated and str(text).strip():
                    updated[label] = str(text).strip()
            q["criteria"] = updated
        out[name] = q
    return out


def parse_question_spec(spec: str) -> tuple[str, dict[str, Any]]:
    """Parse ``name:type:a|b|c`` into a question dict (CLI convenience)."""
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise Sys12Error(f"bad question spec '{spec}' (want name:type:criteria)")
    name, qtype, raw = parts
    criteria = [c.strip() for c in raw.split("|") if c.strip()]
    if not criteria:
        raise Sys12Error(f"empty criteria in '{spec}'")
    if qtype == "choice":
        return name, {"type": "choice",
                      "criteria": {chr(65 + i): c for i, c in enumerate(criteria[:MAX_CRITERIA])}}
    if qtype == "score":
        return name, {"type": "score", "criteria": criteria[:MAX_CRITERIA]}
    if qtype == "noul":
        if len(criteria) < 2:
            raise Sys12Error(f"noul needs two criteria: {spec}")
        return name, {"type": "noul",
                      "criteria": {"true": criteria[0], "false": criteria[1]}}
    raise Sys12Error(f"unknown question type '{qtype}' (choice|score|noul)")


# --------------------------------------------------------------------------- images


def _sniff_mime(raw: bytes) -> str:
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw.startswith(b"GIF8"):
        return "image/gif"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def to_data_uri(image: str) -> str:
    """Accept base64 with or without a ``data:`` prefix and normalize it."""
    text = str(image).strip()
    if not text:
        raise Sys12Error("image cannot be blank")
    if text.startswith("data:"):
        head, _, data = text.partition(",")
        mime = head[5:].split(";")[0] or "image/jpeg"
        try:
            base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise Sys12Error("image data is not valid base64") from exc
        return f"data:{mime};base64,{data}"
    try:
        raw = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise Sys12Error("image is not valid base64 (pass base64, not a path)") from exc
    return f"data:{_sniff_mime(raw)};base64,{text}"


def images_from_paths(paths: list[str]) -> list[str]:
    uris: list[str] = []
    for path in paths:
        candidate = Path(os.path.expanduser(path))
        if not candidate.is_file():
            raise Sys12Error(f"image not found: {path}")
        raw = candidate.read_bytes()
        uris.append(f"data:{_sniff_mime(raw)};base64,{base64.b64encode(raw).decode()}")
    return uris


# --------------------------------------------------------------------------- escalate


def _gate(qtype: str, answer: dict[str, Any]) -> tuple[bool, str | None]:
    """Apply the escalate contract to one normalized answer."""
    if qtype == "noul":
        probability = float(answer.get("noul") or 0.0)
        unsure = YES_UNSURE_LOW < probability < YES_UNSURE_HIGH
        reason = (
            f"yes_probability {probability:.4f} is inside 0.10–0.90; the chat model decides"
            if unsure
            else None
        )
        return unsure, reason
    confidence = answer.get("confidence")
    if confidence is None:
        return True, (
            "confidence missing (emulated answer has no probabilities); the chat model decides"
        )
    unsure = float(confidence) < CHOICE_CONFIDENCE_MIN
    reason = (
        f"confidence {float(confidence):.4f} is below {CHOICE_CONFIDENCE_MIN}; "
        "the chat model decides"
        if unsure
        else None
    )
    return unsure, reason


# --------------------------------------------------------------------------- decide


def build_body(
    model: str,
    state: str,
    questions: dict[str, Any],
    images: list[str] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"model": model, "state": state, "questions": questions}
    if images:
        body["images"] = [to_data_uri(image) for image in images]
    return body


def _normalize_answers(
    payload: dict[str, Any],
    questions: dict[str, Any],
    latency_ms: int,
    backend: str,
    model: str,
    fallback_reason: str | None,
) -> dict[str, Any]:
    raw = payload.get("answers")
    if not isinstance(raw, dict) or not raw:
        raise ShapeError("decision response had no answers")
    answers: dict[str, Any] = {}
    escalations: dict[str, str] = {}
    for name, question in questions.items():
        answer = raw.get(name)
        if not isinstance(answer, dict):
            raise ShapeError(f"decision response is missing an answer for '{name}'")
        qtype = question["type"]
        item: dict[str, Any]
        if qtype == "noul":
            probability = float(answer.get("noul"))
            item = {
                "type": "noul",
                "noul": round(probability, 4),
                "answer": "yes" if probability >= YES_THRESHOLD else "no",
                "confidence": None,
            }
        elif qtype == "choice":
            probabilities = answer.get("probabilities") or {}
            ranked = sorted(
                probabilities.items(), key=lambda item: item[1], reverse=True
            )[:3]
            item = {
                "type": "choice",
                "choice": answer.get("choice"),
                "confidence": answer.get("confidence"),
                "top": [
                    {"option": key, "probability": round(float(value), 4)}
                    for key, value in ranked
                ],
            }
        else:
            score = float(answer.get("score"))
            legend = question["criteria"]
            index = max(0, min(len(legend) - 1, int(round(score))))
            item = {
                "type": "score",
                "score": round(score, 4),
                "answer": legend[index],
                "legend": legend,
                "confidence": answer.get("confidence"),
            }
        unsure, reason = _gate(qtype, item)
        item["escalate"] = unsure
        item["escalate_reason"] = reason
        if unsure and reason:
            escalations[name] = reason
        answers[name] = item
    return {
        "backend": backend,
        "model": model,
        "latency_ms": latency_ms,
        "fallback_reason": fallback_reason,
        "answers": answers,
        "escalate": bool(escalations),
        "escalations": escalations,
        "usage": payload.get("usage"),
    }


def _decide_chat(
    cfg: dict[str, Any],
    state: str,
    questions: dict[str, Any],
    timeout: float,
    images: list[str] | None,
    transport: Transport | None,
    reason: str,
    fallback_model: str | None,
) -> dict[str, Any]:
    """Emulate System One on a chat model when /v1/systemone is unusable.

    One deterministic completion that must reply with JSON. The parsed answer
    has no confidence, so it escalates by construction.
    """
    lines: list[str] = []
    for name, question in questions.items():
        qtype = question["type"]
        criteria = question["criteria"]
        instructions = question.get("instructions") or ""
        if qtype == "choice":
            options = "; ".join(f"{k}: {v}" for k, v in criteria.items())
            lines.append(f"- {name} [choice — answer with one letter]: {options} ({instructions})")
        elif qtype == "noul":
            lines.append(
                f"- {name} [true/false]: true={criteria.get('true', '')} "
                f"false={criteria.get('false', '')} ({instructions})"
            )
        else:
            options = "; ".join(f"{i + 1}: {label}" for i, label in enumerate(criteria))
            lines.append(
                f"- {name} [score — answer with one number 1-{len(criteria)}]: "
                f"{options} ({instructions})"
            )
    prompt = (
        f"Situation:\n{state}\n\nDecide:\n" + "\n".join(lines)
        + "\n\nReply with ONLY a JSON object, one key per question, e.g. "
        '{"name": "A"} or {"name": "true"} or {"name": 2}. No other text.'
    )
    content: Any = prompt
    if images:
        content = [{"type": "text", "text": prompt}]
        for image in images:
            content.append({"type": "image_url", "image_url": {"url": to_data_uri(image)}})
    model = fallback_model or chat_model(cfg)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 256,
        "temperature": 0,
    }
    started = time.perf_counter()
    payload = _call(cfg, CHAT_PATH, body, timeout, transport)
    elapsed = int((time.perf_counter() - started) * 1000)
    text = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    emulated = _parse_emulated(questions, text)
    return _normalize_answers(
        {"answers": emulated, "usage": payload.get("usage")},
        questions,
        elapsed,
        "ollama-cloud-chat",
        payload.get("model") or model,
        reason,
    )


def _parse_emulated(questions: dict[str, Any], text: str) -> dict[str, Any]:
    """Parse a chat model's JSON reply into the /v1/systemone answer shape."""
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.S)
    if fence:
        cleaned = fence.group(1).strip()
    match = re.search(r"\{.*\}", cleaned, re.S)
    parsed: dict[str, Any] = {}
    if match:
        try:
            loaded = json.loads(match.group())
        except json.JSONDecodeError:
            loaded = {}
        if isinstance(loaded, dict):
            parsed = loaded
    answers: dict[str, Any] = {}
    for name, question in questions.items():
        raw = parsed.get(name)
        qtype = question["type"]
        criteria = question["criteria"]
        if qtype == "choice":
            letter = None
            keys = [str(k) for k in criteria]
            if isinstance(raw, str):
                value = raw.strip()
                if value in keys:
                    letter = value
                elif any(key.casefold() == value.casefold() for key in keys):
                    letter = next(
                        key for key in keys if key.casefold() == value.casefold()
                    )
                elif len(value) <= 2 and value[:1].upper() in keys:
                    # A chat model often answers "B", "B." or "(B)".
                    letter = value[:1].upper()
                else:
                    # Or it answers with the option text instead of the letter.
                    folded = value.casefold()
                    for key, description in criteria.items():
                        text = str(description).casefold()
                        if folded and text and (folded in text or text in folded):
                            letter = str(key)
                            break
            elif isinstance(raw, int) and 0 <= raw < len(keys):
                letter = keys[raw]
            answers[name] = {"type": "choice", "choice": letter,
                             "confidence": None, "probabilities": {}}
        elif qtype == "noul":
            value = str(raw).strip().lower() if raw is not None else ""
            trueish = raw is True or value.startswith(("t", "y", "1"))
            answers[name] = {"type": "noul", "noul": 1.0 if trueish else 0.0}
        else:
            digits = re.sub(r"[^0-9]", "", str(raw))
            number = int(digits) if digits else 0
            bounded = min(max(number, 1), len(criteria))
            denominator = max(len(criteria) - 1, 1)
            answers[name] = {
                "type": "score",
                "score": (bounded - 1) / denominator,
                "confidence": None,
            }
    return answers


def decide(
    cfg: dict[str, Any],
    state: str,
    questions: dict[str, Any],
    model: str | None = None,
    images: list[str] | None = None,
    timeout: float | None = None,
    notes: dict[str, str] | None = None,
    *,
    transport: Transport | None = None,
    fallback_model: str | None = None,
) -> dict[str, Any]:
    """System One: structured judgment on Ollama Cloud.

    Cloud /v1/systemone first. A retryable failure — or a 2xx that carries no
    answers — falls back to one deterministic chat completion. Credential and
    request errors do not fall back: retrying the same key only fails twice.
    """
    prepared = normalize_questions(questions)
    prepared = apply_notes(prepared, notes)
    state_text = (state or "").strip()
    if not state_text:
        state_text = "; ".join(
            q.get("instructions") or "" for q in prepared.values()
        ).strip(" ;") or "Decide from the question."
    model_name = model or decision_model(cfg)
    timeout_value = timeout or float(cfg.get("decision_timeout") or 60)
    body = build_body(model_name, state_text, prepared, images)

    started = time.perf_counter()
    try:
        payload = _call(cfg, DECISION_PATH, body, timeout_value, transport)
    except UpstreamError as exc:
        if not exc.retryable:
            raise Sys12Error(str(exc)) from None
        return _decide_chat(
            cfg, state_text, prepared, timeout_value, images, transport,
            reason=str(exc), fallback_model=fallback_model,
        )
    elapsed = int((time.perf_counter() - started) * 1000)
    try:
        return _normalize_answers(
            payload, prepared, elapsed, "ollama-cloud", model_name, None
        )
    except ShapeError as exc:
        return _decide_chat(
            cfg, state_text, prepared, timeout_value, images, transport,
            reason=f"{exc}; fell back to chat", fallback_model=fallback_model,
        )


# --------------------------------------------------------------------------- ask


def ask(
    cfg: dict[str, Any],
    prompt: str,
    model: str | None = None,
    system: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    timeout: float | None = None,
    *,
    transport: Transport | None = None,
) -> dict[str, Any]:
    """System Two: reasoning and generation on Ollama Cloud."""
    if not str(prompt or "").strip():
        raise Sys12Error("prompt is empty")
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    model_name = model or chat_model(cfg)
    body: dict[str, Any] = {"model": model_name, "messages": messages}
    if max_tokens is not None:
        body["max_tokens"] = int(max_tokens)
    if temperature is not None:
        body["temperature"] = float(temperature)
    payload = _call(
        cfg, CHAT_PATH, body, timeout or float(cfg.get("chat_timeout") or 300), transport
    )
    message = ((payload.get("choices") or [{}])[0].get("message") or {})
    return {
        "model": payload.get("model") or model_name,
        "content": message.get("content") or "",
        "reasoning": message.get("reasoning") or message.get("reasoning_content"),
        "usage": payload.get("usage"),
    }


# --------------------------------------------------------------------------- route


def route(
    cfg: dict[str, Any],
    task: str,
    state: str | None = None,
    questions: dict[str, Any] | None = None,
    threshold: float | None = None,
    timeout: float | None = None,
    *,
    transport: Transport | None = None,
) -> dict[str, Any]:
    """The switch: a NOUL gate on System One, then System One or System Two."""
    if not str(task or "").strip():
        raise Sys12Error("task is empty")
    gate_threshold = (
        float(threshold) if threshold is not None
        else float(cfg.get("route_gate_threshold") or 0.5)
    )
    context = f"\nContext: {state}" if state else ""
    if not questions:
        result = ask(cfg, f"{task}{context}", timeout=timeout, transport=transport)
        return {
            "system": "two",
            "gate": {
                "p_deep": None,
                "escalate": True,
                "threshold": None,
                "uncertain": False,
                "note": "no structured questions — free-form goes straight to System Two",
            },
            "result": result,
            "escalate": False,
        }

    prepared = normalize_questions(questions)
    gate_state = f"Task: {task}{context}"
    gate = decide(
        cfg,
        gate_state,
        {"gate": {"type": "noul", "instructions": GATE_INSTRUCTIONS,
                  "criteria": {"true": GATE_TRUE, "false": GATE_FALSE}}},
        timeout=timeout,
        transport=transport,
    )
    gate_answer = gate["answers"]["gate"]
    p_deep = float(gate_answer["noul"])
    deep = p_deep >= gate_threshold
    if deep:
        asked = json.dumps(prepared, ensure_ascii=False)
        result = ask(
            cfg,
            f"{task}{context}\n\nAnswer the following question(s) in text — "
            f"use the option text, not the letter: {asked}",
            timeout=timeout,
            transport=transport,
        )
        system = "two"
        escalate = False
    else:
        result = decide(
            cfg, gate_state, prepared, timeout=timeout, transport=transport
        )
        system = "one"
        escalate = bool(result.get("escalate"))
    return {
        "system": system,
        "gate": {
            "p_deep": round(p_deep, 4),
            "escalate": deep,
            "threshold": gate_threshold,
            "uncertain": bool(gate_answer.get("escalate")),
            "uncertain_reason": gate_answer.get("escalate_reason"),
        },
        "result": result,
        "escalate": escalate,
    }


# --------------------------------------------------------------------------- status


def status(cfg: dict[str, Any] | None = None, *, getter: Callable[..., dict] | None = None,
           timeout: float = 10.0) -> dict[str, Any]:
    """Report local oMLX state and the cloud routing. Reporting only.

    oMLX is never used as a decision backend; this exists because knowing which
    local models are resident is useful when choosing what to run in chat.
    """
    cfg = cfg or load_config()
    omlx = cfg.get("omlx") or {}
    base = (omlx.get("base_url") or "http://127.0.0.1:8000").rstrip("/")
    runner = getter or get_json
    report: dict[str, Any] = {
        "cloud": {
            "base_url": cloud_base(cfg),
            "decision_model": decision_model(cfg),
            "chat_model": chat_model(cfg),
            "decision_path": DECISION_PATH,
            "chat_path": CHAT_PATH,
        },
        "omlx": {"base_url": base, "reachable": False, "models": []},
    }
    try:
        payload = runner(f"{base}/v1/models/status", omlx_key(cfg), timeout)
    except (UpstreamError, Sys12Error) as exc:
        report["omlx"]["error"] = _redact(str(getattr(exc, "detail", exc)), [])
        return report
    models = [
        {
            "alias": item.get("model_alias"),
            "id": item.get("id"),
            "loaded": item.get("loaded"),
            "pinned": item.get("pinned"),
            "is_default": item.get("is_default"),
            "est_gb": round((item.get("estimated_size") or 0) / 1e9, 1),
        }
        for item in payload.get("models", [])
    ]
    report["omlx"].update({
        "reachable": True,
        "loaded_count": payload.get("loaded_count"),
        "model_count": payload.get("model_count"),
        "current_memory_gb": round((payload.get("current_model_memory") or 0) / 1e9, 1),
        "ceiling_gb": round((payload.get("final_ceiling") or 0) / 1e9, 1),
        "models": models,
    })
    return report


# --------------------------------------------------------------------------- browse


def browse(
    cfg: dict[str, Any],
    url: str,
    goal: str,
    max_steps: int = 20,
    form: dict[str, str] | None = None,
    headless: bool = False,
    summarize: bool = False,
) -> dict[str, Any]:
    """Navigate a site with System One picking each step. Needs Playwright."""
    import browser  # lazy: the server runs without Playwright installed

    return browser.browse(
        cfg, url, goal, max_steps=max_steps, form=form,
        headless=headless, summarize=summarize,
    )


# --------------------------------------------------------------------------- CLI


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sys12", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--decision-model", default=None, help="override System One model")
    parser.add_argument("--chat-model", default=None, help="override System Two model")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("decide", help="System One: structured judgment")
    p.add_argument("--state", required=True, help="decision context")
    p.add_argument("--q", action="append", default=[], metavar="NAME:TYPE:A|B|C",
                   help="question spec, repeatable (choice|score|noul)")
    p.add_argument("--note", action="append", default=[], metavar="LABEL=TEXT",
                   help="one-line description for a label, this call only")
    p.add_argument("--image", action="append", default=[], metavar="PATH",
                   help="image file for a vision decision model, repeatable")
    p.add_argument("--model", default=None)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("ask", help="System Two: chat completion")
    p.add_argument("prompt")
    p.add_argument("--model", default=None)
    p.add_argument("--system", default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("route", help="the switch: gate on System One, escalate to System Two")
    p.add_argument("task")
    p.add_argument("--state", default=None)
    p.add_argument("--q", action="append", default=[], metavar="NAME:TYPE:A|B|C")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("status", help="oMLX models, memory, and cloud routing")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("browse", help="System One-driven browser navigation")
    p.add_argument("url")
    p.add_argument("--goal", required=True)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--form", default=None, help="JSON object: input label -> value")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--json", action="store_true")

    args = parser.parse_args()
    cfg = load_config()
    if args.decision_model:
        cfg["cloud"]["decision_model"] = args.decision_model
    if args.chat_model:
        cfg["cloud"]["chat_model"] = args.chat_model

    try:
        _run(args, cfg, parser)
    except Sys12Error as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


def _notes_from_args(pairs: list[str]) -> dict[str, str] | None:
    notes: dict[str, str] = {}
    for pair in pairs:
        label, sep, text = pair.partition("=")
        if not sep:
            raise Sys12Error(f"bad --note '{pair}' (want LABEL=TEXT)")
        notes[label.strip()] = text.strip()
    return notes or None


def _print_decide(out: dict[str, Any]) -> None:
    for name, answer in out.get("answers", {}).items():
        confidence = answer.get("confidence")
        shown = f" (conf {confidence:.2f})" if isinstance(confidence, float) else ""
        if answer["type"] == "choice":
            print(f"{name}: {answer['choice']}{shown}")
        elif answer["type"] == "score":
            print(f"{name}: {answer['score']:.2f} -> {answer['answer']}{shown}")
        else:
            print(f"{name}: p(true)={answer['noul']:.3f} -> {answer['answer']}")
        if answer.get("escalate"):
            print(f"    escalate: {answer.get('escalate_reason')}")
    if out.get("fallback_reason"):
        print(f"    fallback: {out['fallback_reason'][:160]}")


def _run(args, cfg: dict[str, Any], parser: argparse.ArgumentParser) -> None:
    if args.cmd == "decide":
        questions = dict(parse_question_spec(spec) for spec in args.q)
        if not questions:
            parser.error("decide needs at least one --q")
        out = decide(
            cfg, args.state, questions, model=args.model,
            images=images_from_paths(args.image),
            notes=_notes_from_args(args.note),
        )
        if args.json:
            print(json.dumps(out, indent=2, ensure_ascii=False))
        else:
            _print_decide(out)
    elif args.cmd == "ask":
        out = ask(cfg, args.prompt, model=args.model, system=args.system,
                  max_tokens=args.max_tokens, temperature=args.temperature)
        if args.json:
            print(json.dumps(out, indent=2, ensure_ascii=False))
        else:
            if out.get("reasoning"):
                print(f"[reasoning] {out['reasoning'][:600]}\n")
            print(out.get("content") or "(empty)")
    elif args.cmd == "route":
        questions = dict(parse_question_spec(spec) for spec in args.q) or None
        out = route(cfg, args.task, state=args.state, questions=questions,
                    threshold=args.threshold)
        if args.json:
            print(json.dumps(out, indent=2, ensure_ascii=False))
        else:
            gate = out["gate"]
            p_deep = gate["p_deep"]
            shown = f"{p_deep:.3f}" if p_deep is not None else "  -  "
            print(f"gate: p(deep)={shown} -> system {out['system']}")
            if out["system"] == "one":
                _print_decide(out["result"])
            else:
                print((out["result"] or {}).get("content") or "(empty)")
            if out.get("escalate"):
                print("    escalate: System One was unsure; decide in the chat")
    elif args.cmd == "status":
        out = status(cfg)
        if args.json:
            print(json.dumps(out, indent=2, ensure_ascii=False))
        else:
            cloud = out["cloud"]
            print(f"cloud {cloud['base_url']}  one={cloud['decision_model']}  two={cloud['chat_model']}")
            omlx = out["omlx"]
            if not omlx["reachable"]:
                print(f"oMLX {omlx['base_url']} unreachable: {omlx.get('error')}")
            else:
                print(f"oMLX loaded {omlx['loaded_count']}/{omlx['model_count']}  "
                      f"memory {omlx['current_memory_gb']}GB / {omlx['ceiling_gb']}GB")
                for model in omlx["models"]:
                    mark = "●" if model["loaded"] else "○"
                    print(f"  {mark} {model['alias'] or model['id']}  ({model['est_gb']}GB est)")
    elif args.cmd == "browse":
        form = json.loads(args.form) if args.form else None
        out = browse(cfg, args.url, args.goal, max_steps=args.max_steps,
                     form=form, headless=args.headless, summarize=args.summarize)
        import browser

        browser.close()
        if args.json:
            print(json.dumps(out, indent=2, ensure_ascii=False))
        else:
            print(f"status: {out['status']}  url: {out['url']}")
            for step in out.get("steps", []):
                confidence = step.get("confidence")
                shown = f" (conf {confidence:.2f})" if isinstance(confidence, float) else ""
                print(f"  {step['step']}. {step.get('result') or step.get('action')}{shown}")
            if out.get("answer"):
                print(f"\nAnswer:\n{out['answer']}")


if __name__ == "__main__":
    main()
