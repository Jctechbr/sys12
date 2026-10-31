# sys12 — System One / System Two switch

Routes work between two "brains" on **Ollama Cloud**: a non-generative decision
model (**System One**) and a generative chat model (**System Two**). There is no
local model in the decision path.

| System | Endpoint | Default model | Use for |
|---|---|---|---|
| **One** | `POST /v1/systemone` | `clef` | Fast closed judgments — choice, score, yes/no. Answers from next-token probabilities, ~1 s. |
| **Two** | `POST /v1/chat/completions` | `deepseek-v4.1-flash:1017` | Deep reasoning, long generation, multi-step work. |

`sys12_status` still reports the local oMLX server, because knowing which models
are resident is useful before a long local chat job. **It never routes a
decision there.**

## The escalate contract

A decision is a hint, not a verdict. Every answer says whether it stands:

| Kind | Escalate when |
|---|---|
| `choice`, `score` | confidence **below 0.55**, or missing |
| `noul` (yes/no) | `yes_probability` **strictly inside 0.10–0.90** |

Each answer carries `escalate` and `escalate_reason`; the call carries a
top-level `escalate` plus an `escalations` map. When `escalate` is true, the
caller decides and ignores the label. Clef never calls the chat model itself.

A yes/no answer still reports `answer` as yes at 0.5 or above, so a weak yes is
visible even when it escalates.

## Routing logic

`sys12_route` is the switch:

1. **No structured questions** → System Two directly. The gate would only add
   latency, and System One cannot generate text.
2. **With questions** → System One answers a NOUL gate:
   *"can this NOT be settled by a quick structured judgment?"* (`noul` is the
   wire name for yes/no.)
   - `p(deep) >= threshold` (default `0.5`) → escalate to System Two, which
     answers in prose, using option text rather than letters.
   - otherwise → System One answers the structured questions.
3. Returns `{system: "one"|"two", gate: {p_deep, escalate, threshold, uncertain}, result, escalate}`.

`sys12_decide` and `sys12_ask` call one system directly when the caller already
knows which it wants.

## Cloud fallback for System One

Ollama Cloud does serve `/v1/systemone` for decision models. If that call fails
in a way worth retrying (429, 5xx, timeout, connection reset, or a 2xx with no
`answers`), `sys12_decide` falls back to **one deterministic chat completion**
(`temperature 0`, `max_tokens 256`) that must reply with a JSON object, parsed
into the same answer shape.

Consequences:

- `choice` returns the chosen label but **no confidence**;
- `noul` is a hard `0`/`1`, not a probability;
- `score` is derived from the returned rank.

So every emulated answer escalates by construction. That is deliberate: the
fallback exists to turn a failure into a usable hint, not to look confident.

A **401, 402, 403, or 400 does not fall back.** Those are credential, billing,
or request errors, and retrying the same key through chat would only fail twice.

## Configuration

`config.json` next to this file. Every key has an environment override.

| Key | Meaning | Default | Env |
|---|---|---|---|
| `cloud.base_url` | Ollama Cloud base URL | `https://ollama.com` | `SYS12_CLOUD_BASE_URL` |
| `cloud.decision_model` | System One model | `clef` | `SYS12_DECISION_MODEL` |
| `cloud.chat_model` | System Two model | `deepseek-v4.1-flash:1017` | `SYS12_CHAT_MODEL` |
| `cloud.api_key_env` | env var holding the key | `OLLAMA_API_KEY` | |
| `cloud.keychain_service` | macOS Keychain item | `ollama-clef` | |
| `cloud.api_key_file` | last-resort key file | `~/.config/sys12/ollama.env` | |
| `route_gate_threshold` | p(deep) escalation threshold | `0.5` | `SYS12_ROUTE_THRESHOLD` |
| `decision_timeout` | System One timeout (s) | `60` | `SYS12_DECISION_TIMEOUT` |
| `chat_timeout` | System Two timeout (s) | `300` | `SYS12_CHAT_TIMEOUT` |
| `omlx.base_url` | local oMLX, for `status` only | `http://127.0.0.1:8000` | |
| `omlx.api_key_file` | oMLX key file | `~/Library/Application Support/omlx/.env` | |

### The API key

`cloud_key()` tries three sources, in order:

1. `OLLAMA_API_KEY` in the environment.
2. The macOS Keychain item **`ollama-clef`**, account = the login user. On this
   Mac the secret is the **Cursor API Key** field of the Bitwarden item
   "Ollama (Corporativo)". This is the default path and needs no setup here.
3. `~/.config/sys12/ollama.env`, containing `OLLAMA_API_KEY=...`.

To store the Keychain item on a new machine:

```bash
security add-generic-password -s ollama-clef -a "$(id -un)" -w '<key>' -U
```

If all three are absent, the call fails with a message naming all three. It does
not silently attempt an unauthenticated request.

## MCP

Registered in `~/.cursor/mcp.json` as **`sys12`** (stdio via `run_mcp.sh`).
Five tools, every name prefixed so nothing collides with another server:

| Tool | System | Purpose |
|---|---|---|
| `sys12_decide(state, questions, model?, images?, notes?)` | One | Closed judgment. `images` are base64, `data:` prefix optional. |
| `sys12_ask(prompt, model?, system?, max_tokens?, temperature?)` | Two | Chat completion; returns `content`, `reasoning`, `usage`. |
| `sys12_route(task, state?, questions?, threshold?)` | switch | Gate on One, escalate to Two when deep. |
| `sys12_status()` | local | oMLX models and memory, plus the cloud routing. |
| `sys12_browse(url, goal, max_steps?, form?, headless?, summarize?)` | One + Playwright | Navigate with One picking each step. Off unless `SYS12_ALLOW_BROWSE=1`. |

Question specs (dict form; the server tolerates options placed beside `type`):

```json
{
  "strategy": { "type": "choice", "criteria": { "A": "canary", "B": "full" } },
  "severity": { "type": "score",  "criteria": ["low", "medium", "high"] },
  "escalate": { "type": "noul",   "criteria": { "true": "yes", "false": "no" } }
}
```

Normalized automatically: `score` criteria given as a dict (values become the
ordered labels), `choice` criteria given as a list (labels become `A`, `B`, `C`),
`yes_no` as an alias for `noul`, and a missing `instructions` gets a default.

## CLI

```bash
./venv/bin/python sys12.py status                    # oMLX models + memory + routing

./venv/bin/python sys12.py decide --state "Disk at 91%, logs biggest" \
    --q "clean:noul:yes-now|later" \
    --q "severity:score:low|medium|high"

./venv/bin/python sys12.py decide --state "What color fills this image?" \
    --q "color:choice:red|blue|green" --image /tmp/red.png

./venv/bin/python sys12.py ask "Explain MTP speculative decoding"

./venv/bin/python sys12.py route "Design a backup strategy for 600GB" \
    --state "Postgres on one Mac" \
    --q "approach:choice:local-only|3-2-1|cloud"

./venv/bin/python sys12.py browse "https://example.com/pricing" \
    --goal "find the enterprise price" --summarize
```

`--json` on any command for machine-readable output. `--note LABEL=TEXT` passes
a one-line description for that call only. A question spec is
`name:type:criteria[|criteria...]`.

## Browser navigation

`sys12_browse` is a hybrid loop. Playwright observes the page (interactive
elements plus a text sample) and executes actions; System One picks, per step,
which element advances the goal, or one of two stop options. Up to 48 ranked
candidates per step. One persistent Chrome context is reused, so cookies and
logins survive. Optional `summarize` runs System Two over the final page.

Returns `{status: goal_reached|stuck|max_steps|needs_input, stop_reason, steps,
url, title, text, answer?}`.

It is disabled unless `SYS12_ALLOW_BROWSE=1` is in the server environment,
because it drives a real browser. Install the extra to use it:

```bash
./venv/bin/pip install playwright && ./venv/bin/playwright install chromium
```

## Autoplay (DOOM demo)

`autoplay.py` is an autonomous DOOM player built on the same switch — an
end-to-end stress test of the vision path: every step a canvas screenshot goes
to System One, every 5 steps a HUD health OCR, every 10 an automap coverage
check. It is a demo, not part of the server.

```bash
./venv/bin/python autoplay.py [--steps N] [--minutes M]
```

Needs Playwright, system Chrome, and the bundled OCR helper
(`swiftc ocr.swift -o ocr`). Logs: `autoplay.log`, `autoplay.jsonl`.

## Setup

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/python -m unittest discover -s tests    # 39 tests, no network
```

Key: nothing to do on this Mac (Keychain `ollama-clef`). See above for a new one.

Run the MCP server manually:

```bash
./run_mcp.sh        # stdio
```

## Files

| File | Purpose |
|---|---|
| `sys12.py` | Core client + CLI (decide / ask / route / status / browse). |
| `sys12_mcp.py` | FastMCP server (stdio) exposing the five prefixed tools. |
| `browser.py` | System One-driven Playwright navigation loop. |
| `autoplay.py` | Autonomous DOOM player demo. |
| `ocr.swift` | Vision helper (`swiftc ocr.swift -o ocr`). |
| `diag_menu.py` | Ad-hoc OCR/menu diagnostics. |
| `config.json` | Models, endpoints, timeouts, key sources. |
| `requirements.txt` | `fastmcp`; Playwright is an optional extra. |
| `run_mcp.sh` | Launches the MCP server for Cursor. |
| `tests/test_sys12.py` | 39 tests: escalate rule, fallback, redaction, shapes, routing, tool surface. |

## Limits

- `sys12_status` reports the **local oMLX** server only; cloud models are not
  listed there.
- `clef` is a decision model: it works through `/v1/systemone`, not as a chat
  model. A chat request to it returns 400 and does not fall back.
- The chat fallback loses probability information, so it always escalates.
- `sys12_browse` needs system Chrome and Playwright, and is off by default.
- Decisions are cloud only. There is deliberately no local decision path.
