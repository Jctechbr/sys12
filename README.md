# sys12 — System One / System Two switch

Routes work between two "brains" — a **non-generative decision model** (System
One) and a **generative chat model** (System Two). Both sides run against local
[oMLX](https://github.com/omlx) or Ollama Cloud; routing is per model, so the
two systems can live on different providers.

| System | Endpoint | Default model (shipped config) | Use for |
|---|---|---|---|
| **One** | `POST /v1/systemone` | `clef:cloud` (Ollama Cloud) | Fast structured judgments — choice / score / noul. No text generation; answers come from next-token probabilities (local) or a deterministic JSON reply (cloud fallback). |
| **Two** | `POST /v1/chat/completions` | `deepseek-v4.1-flash:1017` (Ollama Cloud) | Deep reasoning, long generation, multi-step work. |

The models are independent: switch either side back to local oMLX by editing
`config.json` (e.g. System One `clef-flash-oQ4e`, System Two `Qwen 3.8 27B Pi`).
The switch is **endpoint-level**, not a model swap.

## Routing logic

`route(task, state?, questions?, threshold?)` is the switch:

1. **No structured questions** → System Two directly. The gate would only add
   latency, and System One cannot generate text.
2. **With questions** → System One answers a NOUL gate:
   *"can this NOT be settled by a quick structured judgment?"*
   - `p(deep) >= threshold` (default `0.5`) → **escalate to System Two**, which
     answers the questions in prose (option text, not letters).
   - otherwise → **System One answers the structured questions** directly.
3. Returns `{system: "one"|"two", gate: {p_deep, escalate, threshold}, result}`.

`decide` and `ask` call a single system directly when the caller already knows
which one it wants.

## Configuration

`config.json` next to this file (env overrides in parentheses):

| Key | Meaning | Default |
|---|---|---|
| `base_url` (`SYS12_BASE_URL`) | local oMLX base URL | `http://127.0.0.1:8000` |
| `api_key_file` | file containing `OMLX_API_KEY=...` | `~/Library/Application Support/omlx/.env` |
| `system1` (`SYS12_SYSTEM1`) | System One model | `clef:cloud` |
| `system2` (`SYS12_SYSTEM2`) | System Two model | `deepseek-v4.1-flash:1017` |
| `route_gate_threshold` | p(deep) escalation threshold | `0.5` |
| `systemone_timeout` | System One request timeout (s) | `60` |
| `ask_timeout` | System Two request timeout (s) | `300` |
| `cloud.base_url` | Ollama Cloud base URL | `https://ollama.com` |
| `cloud.models` | model IDs routed to the cloud | `["clef:cloud", "deepseek-v4.1-flash:1017"]` |
| `cloud.api_key_env` | env var with the cloud key | `OLLAMA_API_KEY` |
| `cloud.api_key_file` | fallback key file | `~/.config/sys12/ollama.env` |

**Model routing:** any model listed in `cloud.models` goes to `cloud.base_url`
with `OLLAMA_API_KEY` (environment first, then the key file). Any other model
goes to the local oMLX `base_url` with `OMLX_API_KEY`.

## CLI

```bash
./sys12.py status                                  # local oMLX loaded models + memory

./sys12.py decide --state "Disk at 91%, logs biggest" \
    --q "clean:noul:yes-now|later" \
    --q "severity:score:low|medium|high"           # System One

./sys12.py ask "Explain MTP speculative decoding"  # System Two

./sys12.py route "Design a backup strategy for 600GB" \
    --state "Postgres on one Mac" \
    --q "approach:choice:local-only|3-2-1|cloud"   # the switch

./sys12.py browse "https://example.com/pricing" \
    --goal "find the enterprise price" --summarize  # System One-driven browser
```

`--json` on any command for machine-readable output. `decide`/`ask` accept
`--model`; `route` accepts `--threshold`; `browse` accepts `--max-steps`,
`--form`, `--headless`, `--summarize`.

Question spec: `name:type:criteria[|criteria...]`
- `choice` → `A|B|C` options (letter labels)
- `score`  → `low|medium|high` labels (returns weighted score + legend)
- `noul`   → `yes|no` (returns p(true))

## MCP (OpenCode)

Registered in `~/.config/opencode/opencode.json` as `sys12` (stdio via
`run_mcp.sh`), exposing five tools:

| Tool | System | Purpose |
|---|---|---|
| `decide(state, questions, model?, images?)` | One | Structured judgment; `images` are base64 (no data-URL prefix) for vision decision models. |
| `ask(prompt, model?, system?, max_tokens?, temperature?)` | Two | Chat completion; returns `content`, `reasoning`, `usage`. |
| `route(task, state?, questions?, threshold?)` | switch | Gate on System One, escalate to System Two when deep. |
| `status()` | local | oMLX loaded models + memory. |
| `browse(url, goal, max_steps?, form?, headless?, summarize?)` | One + Playwright | Navigate a site with System One picking each step. |

Question specs for MCP tools (dict form; the server tolerates options placed
on the question instead of `criteria`):

```json
{
  "strategy":   { "type": "choice", "criteria": { "A": "canary", "B": "full" } },
  "severity":   { "type": "score",  "criteria": ["low", "medium", "high"] },
  "escalate":   { "type": "noul",   "criteria": { "true": "yes", "false": "no" } }
}
```

The MCP server also normalizes common mistakes: `score` criteria given as a
dict (values become labels) and `choice` criteria given as a list (labels
become `A`, `B`, `C`, …).

## Cloud fallback for System One

Ollama Cloud has no `/v1/systemone`. When `decide` hits a cloud decision model
and the endpoint/model is unavailable, sys12 falls back to **one deterministic
chat completion** (temperature `0`, max `256` tokens) that must reply with a
JSON object, then parses it into the same answer shape.

Consequences:
- `choice` returns the chosen letter/label but **no confidence**;
- `noul` is a hard `0`/`1`, not a probability;
- `score` is derived from the returned rank.

The local `/v1/systemone` path remains logits-based (real probabilities, ~1 s).

## Browser navigation

`browse` is a hybrid loop:

- **Deterministic Playwright** observes the page (interactive elements, text)
  and executes actions.
- **System One** picks, at each step, which element advances the goal — or one
  of two stop options ("goal is on this page" / "nothing here can advance it").
- Up to 24 ranked candidates + 2 stop options (System One's 26-choice limit).
- One persistent Chrome context is reused across calls, so cookies/login survive.
- Optional `summarize` runs System Two over the final page text.
- Returns `{status: goal_reached|stuck|max_steps|needs_input, stop_reason,
  steps, url, title, text, answer?}`.

## Autoplay (DOOM demo)

`autoplay.py` is an autonomous DOOM player built on the same switch — a fun
end-to-end stress test of the System One vision path:

- every step: canvas screenshot → System One decision (local `clef` model)
- every 5 steps: HUD health OCR (cheap) → death/respawn handling
- every 10 steps: automap level + coverage → stuck detection
- every 20 steps: full-screen OCR → menu/intermission handling

```bash
./.venv/bin/python autoplay.py [--steps N] [--minutes M]
```

Requires Playwright + system Chrome and the bundled OCR helper (`ocr.swift`,
built with `swiftc ocr.swift -o ocr`). Logs: `autoplay.log`, `autoplay.jsonl`.

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install fastmcp playwright
./.venv/bin/playwright install chromium   # system Chrome via channel is used by browse
```

Keys:
- `OMLX_API_KEY` — env or `~/Library/Application Support/omlx/.env`
  (`run_mcp.sh` loads it for the MCP server).
- `OLLAMA_API_KEY` — env or `~/.config/sys12/ollama.env`, used for cloud models.

Run the MCP server manually:

```bash
./run_mcp.sh        # stdio
```

## Files

| File | Purpose |
|---|---|
| `sys12.py` | Core client + CLI (decide / ask / route / status / browse). |
| `sys12_mcp.py` | FastMCP server (stdio) exposing the five tools. |
| `browser.py` | System One-driven Playwright navigation loop. |
| `autoplay.py` | Autonomous DOOM player demo. |
| `ocr.swift` / `ocr` | Vision helper (build with `swiftc ocr.swift -o ocr`). |
| `diag_menu.py` | Ad-hoc OCR/menu diagnostics. |
| `config.json` | Models, endpoints, cloud routing, timeouts. |
| `run_mcp.sh` | Launches the MCP server for OpenCode. |

## Limitations

- `status` reports the **local oMLX** server only; cloud models are not listed there.
- `clef:cloud` and local `clef-flash-oQ4e` are decision models — they only work
  through `/v1/systemone` (or the chat fallback), not as normal chat models.
- Cloud fallback loses probability information (see above).
- `browse` relies on system Chrome (`channel="chrome"`) and a persistent profile
  under `~/Library/Caches/sys12-chrome`.
