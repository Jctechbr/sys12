#!/usr/bin/env python3
"""sys12 — System One / System Two switch for oMLX and Ollama Cloud.

System One  → POST /v1/systemone        (structured judgments: choice / score / noul;
                                            next-token logits, no generation, ~1s)
System Two  → POST /v1/chat/completions (full generation, reasoning, long context)

Models are routed per provider: IDs listed in `cloud.models` go to Ollama Cloud
with OLLAMA_API_KEY; everything else goes to the local oMLX server with
OMLX_API_KEY. The switch is endpoint-level, not a model swap.

Ollama Cloud has no /v1/systemone: for cloud decision models `decide` falls back
to one deterministic chat completion that must reply with JSON, parsed into the
same answer shape (no probabilities; noul becomes a hard 0/1).

CLI:
  sys12 decide --state "ctx" --q name:type:criteria[|criteria...] [--q ...] [--json]
  sys12 ask "prompt" [--model alias] [--max-tokens N] [--temperature T] [--json]
  sys12 route "task" [--state ctx] [--q ...] [--threshold P] [--json]
  sys12 status [--json]
  sys12 browse URL --goal "..." [--max-steps N] [--form JSON] [--headless] [--summarize] [--json]

`route` is the switch: a NOUL gate question asks System One whether the task
needs deep work. p(true) >= threshold escalates to System Two; otherwise the
provided structured questions are answered by System One. Free-form tasks (no
questions) go straight to System Two, since System One cannot generate text.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = {
    "base_url": "http://127.0.0.1:8000",
    "api_key_file": "~/Library/Application Support/omlx/.env",
    "system1": "clef-flash-oQ4e",
    "system2": "Qwen 3.8 27B Pi",
    "route_gate_threshold": 0.5,
    "systemone_timeout": 60,
    "ask_timeout": 300,
    "cloud": {"base_url": "https://ollama.com", "models": [],
              "api_key_env": "OLLAMA_API_KEY",
              "api_key_file": "~/.config/sys12/ollama.env"},
}

GATE_INSTRUCTIONS = (
    "Judge the task in the state below. Answer true if it CANNOT be settled "
    "by a quick structured judgment (a few options, a short score, or yes/no) "
    "and needs deep reasoning, long-form text, or multi-step work instead. "
    "Answer false if a fast structured judgment suffices."
)
GATE_TRUE = "yes — deep reasoning / long generation / multi-step work required"
GATE_FALSE = "no — a quick structured judgment suffices"


class Sys12Error(RuntimeError):
    """Raised on oMLX API errors (HTTP 4xx/5xx or connection failure)."""


def load_config() -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    here = Path(__file__).resolve().parent
    cfg_path = here / "config.json"
    if cfg_path.exists():
        try:
            cfg.update(json.loads(cfg_path.read_text()))
        except json.JSONDecodeError as e:
            print(f"warn: bad {cfg_path}: {e}", file=sys.stderr)
    # Env overrides.
    if os.environ.get("SYS12_BASE_URL"):
        cfg["base_url"] = os.environ["SYS12_BASE_URL"]
    if os.environ.get("SYS12_SYSTEM1"):
        cfg["system1"] = os.environ["SYS12_SYSTEM1"]
    if os.environ.get("SYS12_SYSTEM2"):
        cfg["system2"] = os.environ["SYS12_SYSTEM2"]
    return cfg


def api_key(cfg: dict[str, Any]) -> str | None:
    if os.environ.get("OMLX_API_KEY"):
        return os.environ["OMLX_API_KEY"]
    key_file = os.path.expanduser(cfg.get("api_key_file", ""))
    if key_file and Path(key_file).is_file():
        for line in Path(key_file).read_text().splitlines():
            line = line.strip()
            if line.startswith("OMLX_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _is_cloud(cfg: dict[str, Any], model: str | None) -> bool:
    cloud = cfg.get("cloud") or {}
    return bool(cloud.get("base_url")) and bool(model) and model in (cloud.get("models") or [])


def _key_from_file(path: str, name: str) -> str | None:
    path = os.path.expanduser(path)
    if not path or not Path(path).is_file():
        return None
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _resolve(cfg: dict[str, Any], model: str | None) -> tuple[str, str | None]:
    """(base_url, api_key) for a model — cloud models route to their provider."""
    if _is_cloud(cfg, model):
        cloud = cfg["cloud"]
        key_env = cloud.get("api_key_env") or "OLLAMA_API_KEY"
        key = os.environ.get(key_env)
        if not key:
            key = _key_from_file(cloud.get("api_key_file") or
                                 "~/.config/sys12/ollama.env", key_env)
        return cloud["base_url"], key
    return cfg["base_url"], api_key(cfg)


def _post(cfg: dict[str, Any], path: str, body: dict[str, Any], timeout: float,
          model: str | None = None) -> dict[str, Any]:
    base, key = _resolve(cfg, model)
    url = base.rstrip("/") + path
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:500]
        raise Sys12Error(f"error: {e.code} from {url}: {detail}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise Sys12Error(f"error: cannot reach {url}: {e}") from e


def _get(cfg: dict[str, Any], path: str, timeout: float = 15) -> dict[str, Any]:
    url = cfg["base_url"].rstrip("/") + path
    headers = {}
    key = api_key(cfg)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:500]
        raise Sys12Error(f"error: {e.code} from {url}: {detail}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise Sys12Error(f"error: cannot reach {url}: {e}") from e


def parse_question_spec(spec: str) -> tuple[str, dict[str, Any]]:
    """Parse `name:type:a|b|c` into (name, question dict).

    choice → criteria dict {A: a, B: b, ...}
    score  → criteria list [a, b, c]
    noul   → criteria {"true": a, "false": b}
    """
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise SystemExit(f"error: bad question spec '{spec}' (want name:type:criteria)")
    name, qtype, crit = parts
    crit = [c.strip() for c in crit.split("|") if c.strip()]
    if not crit:
        raise SystemExit(f"error: empty criteria in '{spec}'")
    if qtype == "choice":
        return name, {"type": "choice", "criteria": {chr(65 + i): c for i, c in enumerate(crit[:26])}}
    if qtype == "score":
        return name, {"type": "score", "criteria": crit[:26]}
    if qtype == "noul":
        if len(crit) < 2:
            raise SystemExit(f"error: noul needs two criteria: {spec}")
        return name, {"type": "noul", "criteria": {"true": crit[0], "false": crit[1]}}
    raise SystemExit(f"error: unknown question type '{qtype}' (choice|score|noul)")


def decide(cfg: dict[str, Any], state: str, questions: dict[str, Any],
           model: str | None = None, timeout: float | None = None,
           images: list[str] | None = None) -> dict[str, Any]:
    """System One: structured judgment via /v1/systemone (no generation).

    `images` (base64 PNG/JPEG, no data-URL prefix) is passed through for
    vision-capable decision models (e.g. clef:cloud) — the model judges the
    situation including what it sees.
    """
    model = model or cfg["system1"]
    questions = _with_instructions(questions)
    body: dict[str, Any] = {"model": model, "state": state, "questions": questions}
    if images:
        # oMLX requires data URIs; Ollama cloud accepts both — normalize.
        body["images"] = [i if str(i).startswith("data:")
                          else f"data:image/jpeg;base64,{i}" for i in images]
    try:
        return _post(cfg, "/v1/systemone", body,
                     timeout or cfg["systemone_timeout"], model=model)
    except Sys12Error as e:
        msg = str(e).lower()
        if _is_cloud(cfg, model) and ("decision model" in msg or "only supports" in msg
                                      or "404" in msg or "model not found" in msg):
            return _decide_chat(cfg, model, state, questions,
                                timeout or cfg["systemone_timeout"], images=images)
        raise


_DEFAULT_INSTRUCTIONS = {
    "choice": "Pick the single option that best fits the situation.",
    "noul": "Judge, from the situation, whether this statement is true.",
    "score": "Rate each option by how well it fits the situation.",
}


def _with_instructions(questions: dict[str, Any]) -> dict[str, Any]:
    """Providers (Ollama cloud) require per-question instructions."""
    out = {}
    for name, q in questions.items():
        if isinstance(q, dict) and not q.get("instructions"):
            q = dict(q)
            q["instructions"] = _DEFAULT_INSTRUCTIONS.get(q.get("type"), "Decide.")
        out[name] = q
    return out


def _decide_chat(cfg: dict[str, Any], model: str, state: str,
                 questions: dict[str, Any], timeout: float,
                 images: list[str] | None = None) -> dict[str, Any]:
    """Emulate System One for providers without /v1/systemone (e.g. Ollama cloud).

    One deterministic chat completion, then parse the JSON reply into the same
    answer shape the logits endpoint returns.
    """
    lines = []
    for name, q in questions.items():
        t = q.get("type")
        crit = q.get("criteria") or {}
        instr = q.get("instructions")
        tail = f" ({instr})" if instr else ""
        if t == "choice":
            opts = "; ".join(f"{k}: {v}" for k, v in crit.items())
            lines.append(f"- {name} [choice — answer with one letter]: {opts}{tail}")
        elif t == "noul":
            lines.append(f"- {name} [true/false — answer true or false]: "
                         f"true={crit.get('true', '')} false={crit.get('false', '')}{tail}")
        elif t == "score":
            opts = "; ".join(f"{i + 1}: {c}" for i, c in enumerate(crit))
            lines.append(f"- {name} [score — answer with one number 1-{len(crit)}]: {opts}{tail}")
    prompt = (f"Situation:\n{state}\n\nDecide:\n" + "\n".join(lines) +
              "\n\nReply with ONLY a JSON object, one key per question, e.g. "
              '{"name": "A"} or {"name": "true"} or {"name": 2}. No other text.')
    content: Any = prompt
    if images:
        content = [{"type": "text", "text": prompt}]
        for img in images:
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img}"}})
    out = _post(cfg, "/v1/chat/completions",
                {"model": model,
                 "messages": [{"role": "user", "content": content}],
                 "max_tokens": 256, "temperature": 0},
                timeout, model=model)
    text = ((out.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    answers = parse_decide_answers(questions, text)
    return {"model": out.get("model") or model, "answers": answers,
            "cloud": True, "raw": text[:300]}


def parse_decide_answers(questions: dict[str, Any], text: str) -> dict[str, Any]:
    """Parse a chat model's JSON reply into the /v1/systemone answer shape."""
    import re

    m = re.search(r"\{.*?\}", text, re.S)
    obj: dict[str, Any] = {}
    if m:
        try:
            obj = json.loads(m.group())
        except json.JSONDecodeError:
            obj = {}
    answers: dict[str, Any] = {}
    for name, q in questions.items():
        raw = obj.get(name)
        t = q.get("type")
        crit = q.get("criteria") or {}
        if t == "choice":
            letter = None
            if isinstance(raw, str):
                s = raw.strip()
                if s.upper()[:1] in crit:
                    letter = s.upper()[:1]
                else:
                    for k, v in crit.items():
                        if isinstance(v, str) and s.lower() in v.lower():
                            letter = k
                            break
            answers[name] = {"type": "choice", "choice": letter, "confidence": None,
                             "label": crit.get(letter) if letter else None}
        elif t == "noul":
            s = str(raw).strip().lower() if raw is not None else ""
            val = 1.0 if s.startswith("t") or raw is True else 0.0
            answers[name] = {"type": "noul", "noul": val}
        elif t == "score":
            try:
                n = int(re.sub(r"[^0-9]", "", str(raw)) or "0")
            except ValueError:
                n = 0
            denom = max(len(crit) - 1, 1)
            score = (min(max(n, 1), len(crit)) - 1) / denom if crit else 0.0
            answers[name] = {"type": "score", "score": round(score, 3),
                             "legend": crit, "confidence": None}
        else:
            answers[name] = {"type": t, "raw": raw}
    return answers


def ask(cfg: dict[str, Any], prompt: str, model: str | None = None,
        system: str | None = None, max_tokens: int | None = None,
        temperature: float | None = None, timeout: float | None = None) -> dict[str, Any]:
    """System Two (or any chat model): /v1/chat/completions."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body: dict[str, Any] = {"model": model or cfg["system2"], "messages": messages}
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if temperature is not None:
        body["temperature"] = temperature
    out = _post(cfg, "/v1/chat/completions", body, timeout or cfg["ask_timeout"],
                model=body["model"])
    msg = (out.get("choices") or [{}])[0].get("message") or {}
    return {
        "model": out.get("model"),
        "content": msg.get("content"),
        "reasoning": msg.get("reasoning") or msg.get("reasoning_content"),
        "usage": out.get("usage"),
    }


def route(cfg: dict[str, Any], task: str, state: str | None = None,
          questions: dict[str, Any] | None = None,
          threshold: float | None = None,
          timeout: float | None = None) -> dict[str, Any]:
    """The switch: System One gate → System One answer, or escalation to System Two."""
    t = threshold if threshold is not None else cfg["route_gate_threshold"]
    gate_state = f"Task: {task}" + (f"\nContext: {state}" if state else "")
    if not questions:
        # Free-form task: System One cannot generate text, so System Two
        # handles it directly — the gate would only add latency.
        prompt = task + (f"\n\nContext: {state}" if state else "")
        return {
            "system": "two",
            "gate": {"p_deep": None, "escalate": True, "threshold": None,
                     "note": "no structured questions — free-form goes straight to System Two"},
            "result": ask(cfg, prompt),
        }
    gate = decide(
        cfg,
        state=gate_state,
        questions={"gate": {
            "type": "noul",
            "instructions": GATE_INSTRUCTIONS,
            "criteria": {"true": GATE_TRUE, "false": GATE_FALSE},
        }},
        timeout=timeout,
    )
    p_deep = (gate["answers"].get("gate") or {}).get("noul")
    deep = p_deep is not None and p_deep >= t
    if deep:
        qtxt = json.dumps(questions)
        ctx = f"\n\nContext: {state}" if state else ""
        result = ask(cfg, f"{task}{ctx}\n\nAnswer the following question(s) in text — use the option text, not the letter: {qtxt}")
        system = "two"
    elif questions:
        result = decide(cfg, state=gate_state, questions=questions, timeout=timeout)
        system = "one"
    else:
        result = None
        system = "one"
    return {
        "system": system,
        "gate": {"p_deep": p_deep, "escalate": deep, "threshold": t},
        "result": result,
    }


def status(cfg: dict[str, Any]) -> dict[str, Any]:
    st = _get(cfg, "/v1/models/status")
    models = [
        {
            "alias": m.get("model_alias"),
            "id": m.get("id"),
            "loaded": m.get("loaded"),
            "pinned": m.get("pinned"),
            "is_default": m.get("is_default"),
            "est_gb": round((m.get("estimated_size") or 0) / 1e9, 1),
        }
        for m in st.get("models", [])
    ]
    return {
        "loaded_count": st.get("loaded_count"),
        "model_count": st.get("model_count"),
        "current_memory_gb": round((st.get("current_model_memory") or 0) / 1e9, 1),
        "ceiling_gb": round((st.get("final_ceiling") or 0) / 1e9, 1),
        "models": models,
    }


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(prog="sys12", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=None, help="oMLX base URL (default from config)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("decide", help="System One: structured judgment via /v1/systemone")
    p.add_argument("--state", required=True, help="decision context (string)")
    p.add_argument("--q", action="append", default=[], metavar="NAME:TYPE:A|B|C",
                   help="question spec, repeatable (types: choice|score|noul)")
    p.add_argument("--model", default=None, help="override System One model")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("ask", help="System Two: chat completion via /v1/chat/completions")
    p.add_argument("prompt")
    p.add_argument("--model", default=None, help="model (default: system2 from config)")
    p.add_argument("--system", default=None, help="system prompt")
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("route", help="the switch: gate on System One, escalate to System Two")
    p.add_argument("task")
    p.add_argument("--state", default=None, help="context (default: the task)")
    p.add_argument("--q", action="append", default=[], metavar="NAME:TYPE:A|B|C",
                   help="structured question to answer with System One if the gate says 'simple'")
    p.add_argument("--threshold", type=float, default=None,
                   help="p(deep) escalation threshold (default from config)")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("status", help="oMLX loaded models + memory")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("browse", help="System One-driven browser navigation (Playwright + system Chrome)")
    p.add_argument("url")
    p.add_argument("--goal", required=True, help="what to find/accomplish")
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--form", default=None, help='JSON object: input label -> value')
    p.add_argument("--headless", action="store_true", help="run Chrome hidden")
    p.add_argument("--summarize", action="store_true", help="System Two summarizes the final page")
    p.add_argument("--json", action="store_true")

    args = ap.parse_args()
    if args.base_url:
        cfg["base_url"] = args.base_url

    try:
        _run(args, cfg, ap)
    except Sys12Error as e:
        print(e, file=sys.stderr)
        sys.exit(1)


def _run(args, cfg: dict[str, Any], ap: argparse.ArgumentParser) -> None:
    if args.cmd == "decide":
        questions = dict(parse_question_spec(s) for s in args.q)
        if not questions:
            ap.error("decide needs at least one --q")
        out = decide(cfg, args.state, questions, model=args.model)
        if args.json:
            print(json.dumps(out, indent=2))
        else:
            for name, a in out.get("answers", {}).items():
                conf = a.get("confidence")
                c = f" (conf {conf:.2f})" if isinstance(conf, float) else ""
                if a["type"] == "choice":
                    print(f"{name}: {a['choice']}{c}")
                elif a["type"] == "score":
                    print(f"{name}: {a['score']:.2f}  legend={a['legend']}{c}")
                else:
                    print(f"{name}: p(true)={a['noul']:.3f}")
    elif args.cmd == "ask":
        out = ask(cfg, args.prompt, model=args.model, system=args.system,
                  max_tokens=args.max_tokens, temperature=args.temperature)
        if args.json:
            print(json.dumps(out, indent=2))
        else:
            if out.get("reasoning"):
                print(f"[reasoning] {out['reasoning'][:400]}")
            print(out.get("content") or "(empty)")
    elif args.cmd == "route":
        questions = dict(parse_question_spec(s) for s in args.q) or None
        out = route(cfg, args.task, state=args.state, questions=questions,
                    threshold=args.threshold)
        if args.json:
            print(json.dumps(out, indent=2))
        else:
            g = out["gate"]
            pd = f"{g['p_deep']:.3f}" if g["p_deep"] is not None else "  -  "
            print(f"gate: p(deep)={pd} → system {out['system']}")
            if out["result"] is not None and out["system"] == "one":
                for name, a in out["result"].get("answers", {}).items():
                    if a["type"] == "noul":
                        print(f"{name}: p(true)={a['noul']:.3f}")
                    elif a["type"] == "choice":
                        print(f"{name}: {a['choice']}")
                    else:
                        print(f"{name}: {a['score']:.2f}")
            elif out["result"] is not None:
                print(out["result"].get("content") or "(empty)")
    elif args.cmd == "status":
        out = status(cfg)
        if args.json:
            print(json.dumps(out, indent=2))
        else:
            print(f"loaded {out['loaded_count']}/{out['model_count']}  "
                  f"memory {out['current_memory_gb']}GB / {out['ceiling_gb']}GB")
            for m in out["models"]:
                mark = "●" if m["loaded"] else "○"
                print(f" {mark} {m['alias'] or m['id']}  ({m['est_gb']}GB est)"
                      + ("  [default]" if m["is_default"] else ""))
    elif args.cmd == "browse":
        import browser

        form = json.loads(args.form) if args.form else None
        out = browser.browse(cfg, args.url, args.goal, max_steps=args.max_steps,
                             form=form, headless=args.headless, summarize=args.summarize)
        browser.close()
        if args.json:
            print(json.dumps(out, indent=2, ensure_ascii=False))
        else:
            print(f"status: {out['status']}  url: {out['url']}")
            for s in out.get("steps", []):
                conf = s.get("confidence")
                ct = f" (conf {conf:.2f})" if isinstance(conf, float) else ""
                print(f"  {s['step']}. {s.get('result') or s.get('action')}{ct}")
            if out.get("answer"):
                print(f"\nAnswer:\n{out['answer']}")


if __name__ == "__main__":
    main()
