#!/usr/bin/env python3
"""sys12 browser — System One-driven navigation loop.

System One (the cloud decision model) is non-generative: it classifies
(choice/score/noul), so it cannot emit actions or text. The browse loop is
therefore hybrid:

  - deterministic Playwright code  → observe (elements, page text) + execute
  - System One (sys12.decide)      → per-step choice: which element advances
                                     the goal, or stop (goal reached / stuck)

The loop is tight in-process (observe + decide + act ≈ 1–3 s/step), which is
much faster than a 27B agent generating each step. One persistent browser
context is reused across calls so cookies/login survive between browse() calls.
"""

from __future__ import annotations

import re
import string
from typing import Any

import sys12

SEL = "a, button, [role='button'], input:not([type='hidden']), select"

STOP_GOAL = "stop: the goal's answer is on this page"
STOP_STUCK = "stop: nothing here can advance the goal"
_SAMPLE_CHARS = 1200
_TEXT_CHARS = 6000

_STOPS = {STOP_GOAL, STOP_STUCK}

# Element letters A, B, C ... come from ASCII; the two stop options live past
# the candidate range so they can never collide with a real element letter
# (Y and Z collided as soon as the candidate cap passed 24).
_CANDIDATE_LETTERS = tuple(string.ascii_uppercase)
STOP_GOAL_LETTER = "~goal"
STOP_STUCK_LETTER = "~stuck"
# Derive the cap from the letter space so the two can never drift apart:
# a candidate the ranker keeps but _decide cannot letter is silently dropped.
MAX_CANDIDATES = len(_CANDIDATE_LETTERS)

_pw = None
_browser = None
_context = None
_page = None


def _teardown() -> None:
    """Best-effort cleanup of a dead/stale browser before relaunch."""
    global _pw, _browser, _context, _page
    try:
        if _browser is not None:
            _browser.close()
    except Exception:
        pass
    try:
        if _pw is not None:
            _pw.stop()
    except Exception:
        pass
    _pw = _browser = _context = _page = None


def _ensure_page(headless: bool = False):
    """Launch (once) and return the working page. System Chrome, persistent context.

    Re-launches automatically if the browser died (window closed / crash) so a
    stale page reference never causes a TargetClosedError.
    """
    global _pw, _browser, _context, _page
    if _page is not None and _browser is not None:
        try:
            if _browser.is_connected() and not _page.is_closed():
                return _page
        except Exception:
            pass
    _teardown()
    from playwright.sync_api import sync_playwright

    _pw = sync_playwright().start()
    _browser = _pw.chromium.launch(channel="chrome", headless=headless)
    _context = _browser.new_context(viewport={"width": 1366, "height": 900})
    _page = _context.new_page()
    return _page


def close() -> None:
    """Close the persistent browser (call when done, e.g. at CLI exit)."""
    global _pw, _browser, _context, _page
    try:
        if _browser is not None:
            _browser.close()
    except Exception:  # noqa: BLE001 - teardown must never raise
        pass
    finally:
        try:
            if _pw is not None:
                _pw.stop()
        except Exception:  # noqa: BLE001
            pass
        _pw = _browser = _context = _page = None


def _goal_keywords(goal: str) -> list[str]:
    words = re.findall(r"[a-z0-9à-ÿ]{3,}", goal.lower())
    stop = {"como", "obter", "conseguir", "fazer", "pedir", "pedido",
            "the", "how", "para", "qual", "onde", "que", "uma", "um", "de", "do", "da", "em", "no", "na", "com"}
    return [w for w in words if w not in stop]


def _rank(cands: list[dict], goal: str) -> list[dict]:
    """Deterministic pre-filter: keep the candidates most relevant to the goal.

    System One sees only the ranked head, so rank by keyword overlap (goal words
    in the element text) and keep the top MAX_CANDIDATES, tie-broken by DOM order.
    """
    kws = _goal_keywords(goal)
    for c in cands:
        c["_score"] = sum(1 for k in kws if k in c["label"].lower())
    cands.sort(key=lambda c: (-c["_score"], c["ref"]))
    return cands[:MAX_CANDIDATES]


# Semantic controls always count; div/span only when they show cursor:pointer
# (the main way SPAs make non-semantic elements clickable, e.g. React onClick).
BROAD = "a, button, [role='button'], input:not([type='hidden']), select, div, span"


def _observe(page, goal: str) -> dict:
    raw = page.evaluate(
        """(sel) => {
            // Scope to the topmost visible modal/dialog if one is open, so we act
            // on the dialog's controls rather than the (still-present) page behind it.
            let scope = document;
            const modals = Array.from(
                document.querySelectorAll('[role=dialog],[class*=modal],[class*=dialog],[class*=popup],[class*=drawer]'))
                .filter(e => e.offsetWidth || e.offsetHeight || e.getClientRects().length);
            if (modals.length) scope = modals[modals.length - 1];
            // Clear stale refs from the previous observe (they persist on the
            // same-page DOM, e.g. behind-modal elements, and would collide).
            document.querySelectorAll('[data-browse-ref]').forEach(e => e.removeAttribute('data-browse-ref'));
            const els = scope.querySelectorAll(sel);
            const out = []; const byKey = new Map(); let ref = 0;
            for (const el of els) {
                const tag = el.tagName.toLowerCase();
                const semantic = tag==='a'||tag==='button'||tag==='select'||tag==='input'||el.getAttribute('role')==='button';
                if (!semantic) {
                    if (tag!=='div' && tag!=='span') continue;
                    if (getComputedStyle(el).cursor !== 'pointer') continue;
                }
                const t = (el.innerText || el.value || el.getAttribute('aria-label') ||
                           el.getAttribute('placeholder') || el.getAttribute('name') || '').trim().replace(/\\s+/g,' ');
                if (!t) continue;
                // For inputs/selects, prefer an associated visible label (e.g. "CEP")
                // over the placeholder, so the model can tell fields apart.
                if (tag==='input' || tag==='select') {
                    let lbl = '';
                    if (el.id) { const l = document.querySelector('label[for="'+el.id+'"]'); if (l) lbl = (l.innerText||'').trim(); }
                    const wrap = el.closest('label'); if (!lbl && wrap) lbl = (wrap.innerText||'').trim();
                    if (lbl) t = lbl;
                }
                const v = !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length);
                if (!v) continue;
                if (!semantic && t.length > 60) continue;
                const key = tag + ':' + t;
                if (byKey.has(key)) {
                    // prefer the deepest (latest) element for identical text
                    const prevEl = byKey.get(key);
                    const r = prevEl.getAttribute('data-browse-ref');
                    prevEl.removeAttribute('data-browse-ref');
                    el.setAttribute('data-browse-ref', r);
                    byKey.set(key, el);
                    continue;
                }
                byKey.set(key, el);
                el.setAttribute('data-browse-ref', String(ref));
                out.push({ ref, tag, text: t.slice(0,80),
                           href: (el.getAttribute('href')||'').slice(0,80) });
                ref++;
            }
            return { scope: scope === document ? null : (scope.className||'').toString().slice(0,40), items: out };
        }""",
        BROAD,
    )
    scope = raw.get("scope")
    items = raw.get("items", [])
    seen: set[str] = set()
    cands: list[dict] = []
    for r in items:
        label = r["text"] or (f"({r['href']})" if r["href"] and r["tag"] == "a" else "")
        if not label:
            continue
        if r["tag"] == "a" and r["href"] and "http" in r["href"]:
            # Show the destination so System One can tell info pages from services/logins.
            from urllib.parse import urlsplit

            sp = urlsplit(r["href"])
            dest = (sp.netloc + sp.path)[:60]
            label = f"{label} → {dest}"
        key = f"{r['tag']}:{label}"
        if key in seen:
            continue
        seen.add(key)
        cands.append({"ref": r["ref"], "tag": r["tag"], "label": label,
                      "href": r["href"] if r["tag"] == "a" else None})
    cands = _rank(cands, goal)
    title = page.title()
    sample = re.sub(r"\s+", " ", page.inner_text("body"))[:_SAMPLE_CHARS]
    return {"title": title, "url": page.url, "sample": sample, "scope": scope,
            "candidates": cands, "n_raw": len(items)}


def _decide(cfg: dict[str, Any], goal: str, obs: dict) -> tuple[str, str, float | None]:
    """Ask System One which candidate to act on. Returns (letter, label, confidence)."""
    criteria: dict[str, str] = {}
    label_by_letter: dict[str, str] = {}
    for idx, c in enumerate(obs["candidates"]):
        if idx >= len(_CANDIDATE_LETTERS):
            break
        letter = _CANDIDATE_LETTERS[idx]
        criteria[letter] = c["label"]
        label_by_letter[letter] = c["label"]
    criteria[STOP_GOAL_LETTER] = STOP_GOAL
    criteria[STOP_STUCK_LETTER] = STOP_STUCK
    label_by_letter[STOP_GOAL_LETTER] = STOP_GOAL
    label_by_letter[STOP_STUCK_LETTER] = STOP_STUCK

    state = (
        f"Goal: {goal}\n"
        f"Page: {obs['title']} ({obs['url']})\n"
        f"Visible content sample:\n{obs['sample']}\n"
        f"Interactive elements on the page (lettered):"
    )
    out = sys12.decide(
        cfg, state,
        {"action": {
            "type": "choice",
            "instructions": (
                "Candidates are ordered by relevance (A = best keyword match to the goal). "
                "Pick the single element that best advances the goal; if several do, prefer "
                "the most relevant (earliest letter). For an informational goal (research, "
                "how-to, requirements) prefer an informational page on the current domain "
                "over external portals, login, account, or application pages. "
                "Choose 'stop: goal' only if the answer is already in the visible content; "
                "'stop: stuck' if no element can advance the goal."
            ),
            "criteria": criteria,
        }},
    )
    ans = out["answers"]["action"]
    letter = ans["choice"]
    # An escalated answer is only a hint; the letter is still the best guess,
    # so keep it and let the caller see the confidence.
    if letter is None:
        return None, "", ans.get("confidence")
    return str(letter), label_by_letter.get(letter, ""), ans.get("confidence")


def _act(page, context, cand: dict | None, label: str, form: dict[str, str] | None) -> str:
    """Execute the chosen candidate (located by its stable data-browse-ref)."""
    if cand is None:
        return f"stop: {label}"
    loc = page.locator(f'[data-browse-ref="{cand["ref"]}"]')
    if cand["tag"] == "a":
        href = cand.get("href")
        if href and (href.startswith("http") or href.startswith("/")):
            from urllib.parse import urljoin

            page.goto(urljoin(page.url, href), timeout=30000)
            page.wait_for_load_state("domcontentloaded", timeout=20000)
            page.wait_for_timeout(1500)  # let the SPA settle
            return f"navigated {href[:60]}"
    if cand["tag"] in ("a", "button", "div", "span"):
        # JS click: fires the handler directly, ignoring overlays/backdrops
        # (Playwright's .click() would time out waiting past a cookie backdrop).
        loc.evaluate("el => el.click()")
        page.wait_for_load_state("domcontentloaded", timeout=20000)
        page.wait_for_timeout(1200)  # let any modal/route render
        return f"clicked {cand['tag']} '{cand['label'][:50]}'"
    if cand["tag"] == "input":
        value = (form or {}).get(cand["label"])
        if value is None:
            return f"needs input: '{cand['label']}' (no value provided)"
        loc.fill(value)
        # Blur to fire any change/lookup handlers (e.g. a CEP field that
        # auto-fills the rest of the address), then let the async lookup land.
        try:
            loc.dispatch_event("blur")
        except Exception:
            pass
        page.wait_for_timeout(3500)
        return f"filled '{cand['label'][:50]}'"
    return f"skipped {cand['tag']} '{cand['label'][:50]}' (not actionable)"


def _state_key(page, obs: dict | None = None) -> str:
    """URL + a fingerprint of the visible candidates.

    The fingerprint distinguishes same-URL state changes (a dialog/modal opens
    or closes) from true cycles (identical URL *and* identical candidates).
    Without it, opening a modal on the same page is mistaken for a loop.
    """
    url = page.url.split("#", 1)[0].split("?", 1)[0]
    if url.endswith("/") and url.count("/") > 3:
        url = url.rstrip("/")
    if obs is None:
        return url
    import hashlib

    fp = hashlib.md5("".join(c["label"] for c in obs["candidates"]).encode()).hexdigest()[:8]
    return f"{url}#{fp}"


_CONSENT = ("aceitar", "accept", "ok", "got it", "entendi", "li e aceito",
            "fechar", "entendido", "allow", "permitir", "i agree", "agree")


def _dismiss_overlays(page, wait: bool = False) -> str:
    """Click a visible cookie/consent accept button if present.

    Consent backdrops intercept pointer events and block every other click, so
    they must go first. Uses a JS click (ignores overlays) for reliability.
    ``wait=True`` polls up to ~4 s for the dialog to appear (banners often
    animate in a few seconds after load); use it on the initial page only.
    """
    present = page.evaluate(
        """(wait) => {
            const check = () => {
                const d = document.querySelector(
                    '[role=dialog][aria-modal=true], [class*=cookie-banner-container], [class*=consent]');
                return !!(d && (d.offsetWidth || d.offsetHeight || d.getClientRects().length));
            };
            if (!wait) return check();
            return new Promise(res => {
                const start = Date.now();
                const t = setInterval(() => {
                    if (check()) { clearInterval(t); res(true); return; }
                    if (Date.now() - start > 4000) { clearInterval(t); res(false); }
                }, 250);
            });
        }""",
        wait,
    )
    if not present:
        return ""
    clicked = page.evaluate(
        """pats => {
            const els = document.querySelectorAll('a, button, [role=button]');
            for (const el of els) {
                const t = (el.innerText||'').trim().toLowerCase().replace(/\\s+/g,' ');
                if (!t || t.length > 30) continue;
                if (!(el.offsetWidth || el.offsetHeight || el.getClientRects().length)) continue;
                if (pats.some(p => t === p || t.startsWith(p))) { el.click(); return t; }
            }
            return null;
        }""",
        list(_CONSENT),
    )
    if clicked:
        page.wait_for_timeout(800)
        return f"dismissed overlay '{clicked}'"
    return ""


def browse(
    cfg: dict[str, Any],
    url: str,
    goal: str,
    max_steps: int = 20,
    form: dict[str, str] | None = None,
    headless: bool = False,
    summarize: bool = False,
) -> dict[str, Any]:
    """Navigate with System One as the per-step decision maker.

    Args:
        url: Starting URL.
        goal: What to find / accomplish (used for candidate ranking + decisions).
        max_steps: Safety cap on observe-decide-act iterations.
        form: Optional {input label: value} for filling inputs the model picks.
        headless: Run Chrome hidden (default: visible, so you can watch it).
        summarize: Also run System Two over the final page text to produce an answer.

    Returns:
        {status, stop_reason, steps, url, title, text, answer?}
    """
    if not str(url or "").strip():
        raise sys12.Sys12Error("url is empty")
    if not str(goal or "").strip():
        raise sys12.Sys12Error("goal is empty")
    max_steps = max(1, min(int(max_steps), 200))
    page = _ensure_page(headless=headless)
    page.goto(url, timeout=45000)
    page.wait_for_load_state("domcontentloaded", timeout=30000)
    cur = page  # may switch to a new tab after a click
    # Let the SPA hydrate its click handlers and let banners animate in before
    # we interact; clicking a not-yet-hydrated React/Angular element is a no-op.
    page.wait_for_timeout(2500)
    _dismiss_overlays(cur, wait=True)

    steps: list[dict[str, Any]] = []
    status = "max_steps"
    stop_reason = ""
    seen_states: set[str] = set()
    for step in range(1, max_steps + 1):
        note0 = _dismiss_overlays(cur)
        if note0:
            steps.append({"step": step, "action": note0})
        obs = _observe(cur, goal)
        state_key = _state_key(cur, obs)
        if step > 1 and state_key in seen_states:
            status = "stuck"
            stop_reason = "cycle: revisited a page, no progress"
            steps.append({"step": step, "action": stop_reason})
            break
        seen_states.add(state_key)
        letter, label, conf = _decide(cfg, goal, obs)
        if letter is None:
            status = "stuck"
            stop_reason = "no usable choice from the decision model"
            steps.append({"step": step, "action": stop_reason, "confidence": conf})
            break
        cand = None
        index = (
            _CANDIDATE_LETTERS.index(letter)
            if letter in _CANDIDATE_LETTERS else -1
        )
        if 0 <= index < len(obs["candidates"]):
            cand = obs["candidates"][index]
        if label in _STOPS:
            status = "goal_reached" if label == STOP_GOAL else "stuck"
            stop_reason = label
            steps.append({"step": step, "action": label, "confidence": conf})
            break
        if cand is None:
            status = "stuck"
            stop_reason = f"unresolvable choice '{str(label)[:50]}'"
            steps.append({"step": step, "action": stop_reason, "confidence": conf})
            break
        note = _act(cur, cur.context, cand, label, form)
        if len(cur.context.pages) > 1:
            cur = cur.context.pages[-1]
            cur.wait_for_load_state("domcontentloaded", timeout=20000)
        steps.append({"step": step, "picked": label[:60], "result": note,
                      "confidence": conf})
        if note.startswith("needs input"):
            status = "needs_input"
            stop_reason = note
            break

    text = re.sub(r"\s+", " ", cur.inner_text("body"))[:_TEXT_CHARS]
    out: dict[str, Any] = {
        "status": status,
        "stop_reason": stop_reason or None,
        "steps": steps,
        "url": cur.url,
        "title": cur.title(),
        "text": text,
    }
    if summarize and status != "needs_input":
        res = sys12.ask(cfg, f"Goal: {goal}\n\nFinal page content:\n{text}\n\n"
                             "Answer the goal concisely based only on this content.")
        out["answer"] = res.get("content")
    return out
