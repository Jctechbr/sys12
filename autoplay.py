#!/usr/bin/env python3
"""Autonomous DOOM player v2 — frequent state checks, map-coverage feedback.

Loop (in-process, no MCP round-trips):
  every step    : vision perception + action decision (System One model)
  every 5 steps : HUD health check (OCR, cheap) -> instant death/respawn
  every 10 steps: automap level + coverage -> stuck detection (search mode)
  every 20 steps: full-screen OCR -> menu / intermission handling

Run:  ./.venv/bin/python autoplay.py [--steps N] [--minutes M]
Logs: autoplay.log / autoplay.jsonl
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

import sys12

HERE = Path(__file__).resolve().parent
OCR_BIN = HERE / "ocr"
LOG = HERE / "autoplay.log"
JSONL = HERE / "autoplay.jsonl"
PROFILE = Path.home() / "Library/Caches/sys12-chrome"
SHOT = Path("/private/var/folders/mr/3fc4llxd4wj17hqrd_5xfqyc0000gn/T/opencode/shot.png")

CAP_JS = """() => { const c=document.querySelector('canvas');
  const o=document.createElement('canvas'); o.width=c.width; o.height=Math.floor(c.height*0.84);
  const x=o.getContext('2d'); x.drawImage(c,0,0); return o.toDataURL('image/jpeg',0.85); }"""

SHOT_JS = """() => { const c=document.querySelector('canvas');
  const o=document.createElement('canvas'); o.width=c.width*3; o.height=c.height*3;
  const x=o.getContext('2d'); x.imageSmoothingEnabled=false; x.drawImage(c,0,0,o.width,o.height);
  return o.toDataURL('image/png'); }"""

HUD_JS = """() => { const c=document.querySelector('canvas'); const sy=c.height-32;
  const cx=Math.floor(c.width*0.16), cw=Math.floor(c.width*0.30);
  const o=document.createElement('canvas'); o.width=cw*8; o.height=32*8;
  const x=o.getContext('2d'); x.imageSmoothingEnabled=true;
  x.drawImage(c, cx, sy, cw, 32, 0, 0, o.width, o.height); return o.toDataURL('image/png'); }"""

DIFF_JS = """() => { const c=document.querySelector('canvas');
  const o=document.createElement('canvas'); o.width=80; o.height=50;
  const x=o.getContext('2d'); x.drawImage(c,0,0,80,50);
  const d=x.getImageData(0,0,80,50).data; const arr=new Uint8Array(80*50);
  for (let i=0,j=0;i<d.length;i+=4,j++) arr[j]=(d[i]+d[i+1]+d[i+2])/3;
  const prev=window.__prev; window.__prev=arr;
  if (!prev) return 1.0;
  let n=0; for (let i=0;i<arr.length;i++) if (Math.abs(arr[i]-prev[i])>12) n++;
  return n/arr.length; }"""

BRIGHT_JS = """() => { const c=document.querySelector('canvas');
  const o=document.createElement('canvas'); o.width=c.width; o.height=c.height;
  const x=o.getContext('2d'); x.drawImage(c,0,0);
  const d=x.getImageData(0,0,o.width,o.height).data; let n=0;
  for (let i=0;i<d.length;i+=4) if (Math.max(d[i],d[i+1],d[i+2])>120) n++; return n; }"""

FP_JS = """() => { const c=document.querySelector('canvas');
  const cw = Math.floor(c.width*0.25), chh = Math.floor(c.height*0.30);
  const cx = Math.floor(c.width/2 - cw/2), cy = Math.floor(c.height*0.42 - chh/2);
  const o=document.createElement('canvas'); o.width=16; o.height=12;
  const x=o.getContext('2d'); x.drawImage(c, cx, cy, cw, chh, 0, 0, 16, 12);
  const d=x.getImageData(0,0,16,12).data; let s='';
  for (let i=0;i<d.length;i+=4) s += (Math.max(d[i],d[i+1],d[i+2])>120) ? '1' : '0';
  return s; }"""

KEYS = {"forward": "w", "back": "s", "turn_left": "ArrowLeft",
        "turn_right": "ArrowRight", "fire": "Control", "open": "Space",
        "strafe_left": "a", "strafe_right": "d"}


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def b64(durl: str) -> str:
    return durl.split(",", 1)[1]


def _ocr_file(page, js: str) -> str:
    SHOT.write_bytes(base64.b64decode(b64(page.evaluate(js))))
    try:
        out = subprocess.run([str(OCR_BIN), str(SHOT)],
                             capture_output=True, text=True, timeout=25).stdout
        return " ".join(out.split())
    except Exception as e:  # noqa: BLE001
        return f"OCR_ERROR {e}"


def ocr(page) -> str:
    return _ocr_file(page, SHOT_JS)


def hud_ocr(page) -> str:
    return _ocr_file(page, HUD_JS)


def parse_health(txt: str) -> int | None:
    m = re.search(r"(\d{1,3})\s*%", txt)
    return int(m.group(1)) if m else None


def probe(page, hold_ms: int = 420) -> float:
    page.evaluate(DIFF_JS)
    page.keyboard.down("w")
    page.wait_for_timeout(hold_ms)
    page.keyboard.up("w")
    page.wait_for_timeout(140)
    return float(page.evaluate(DIFF_JS))


def focus_game(page) -> bool:
    page.bring_to_front()
    page.wait_for_timeout(120)
    if probe(page) > 0.01:
        return True
    box = page.evaluate("""() => { const c=document.querySelector('canvas');
        const r=c.getBoundingClientRect(); return {x:r.x+r.width/2, y:r.y+r.height/2}; }""")
    page.mouse.click(box["x"], box["y"])
    page.wait_for_timeout(300)
    return probe(page) > 0.01


def norm(t: str) -> str:
    """Normalize OCR output: Cyrillic lookalikes -> Latin, digit confusions."""
    t = t.upper()
    for a, b in (("А", "A"), ("Е", "E"), ("О", "O"), ("Р", "P"), ("С", "C"),
                 ("У", "Y"), ("Х", "X"), ("К", "K"), ("М", "M"), ("Т", "T"),
                 ("Н", "H"), ("В", "B"), ("І", "I"), ("Є", "E"), ("ԁ", "D")):
        t = t.replace(a, b)
    t = re.sub(r"E[IL1][MN]", "E1M", t)
    t = re.sub(r"E[IL1][MN]", "E1M", t)
    return re.sub(r"\s+", " ", t)


MENU_WORDS = ("NEW GAME", "EPISODE", "SKILL", "OPTIONS", "MESSAGES",
              "SENSITIVITY", "KNEE", "SHORES", "INFERNO", "TOO YOUNG",
              "NIGHTMARE", "HURT ME", "READ THIS", "QUIT GAME", "LOAD GAME")


def menu_open(txt: str) -> bool:
    u = norm(txt)
    return any(w in u for w in MENU_WORDS)


def verify_in_game(page) -> bool:
    """Hard check: automap shows a level tag / name."""
    page.keyboard.press("Tab")
    page.wait_for_timeout(550)
    t = norm(ocr(page))
    page.keyboard.press("Tab")
    page.wait_for_timeout(400)
    if re.search(r"E[1-4]M\d", t) or "HANGAR" in t:
        return True
    return False


def advance_menus(page) -> bool:
    """Drive whatever menu is on screen until an actual game starts."""
    for i in range(8):
        if verify_in_game(page):
            return True
        if i in (3, 5):
            page.keyboard.press("1")           # skill hotkey fallback
        else:
            page.keyboard.press("Enter")       # main menu -> episode -> skill -> game
        page.wait_for_timeout(1200)
    return False


def start_game(page) -> bool:
    for attempt in range(3):
        if verify_in_game(page):
            log(f"IN GAME (verified, attempt {attempt})")
            return True
        log(f"start attempt {attempt}")
        for _ in range(3):                     # close whatever menu is open
            page.keyboard.press("Escape")
            page.wait_for_timeout(400)
        page.keyboard.press("Escape")          # open main menu
        page.wait_for_timeout(700)
        if advance_menus(page):
            log("IN GAME (menus advanced)")
            return True
        log("menu advance failed -> reloading page")
        reload_page(page)
    return False


def press(page, key: str, hold_ms: int = 420) -> None:
    if key in ("Control", "Space"):
        page.keyboard.down(key)
        page.wait_for_timeout(240)
        page.keyboard.up(key)
        page.wait_for_timeout(100)
    elif key == "w":
        page.keyboard.down("Shift")
        page.keyboard.down("w")
        page.wait_for_timeout(hold_ms + 80)
        page.keyboard.up("w")
        page.keyboard.up("Shift")
        page.wait_for_timeout(100)
    else:
        page.keyboard.down(key)
        page.wait_for_timeout(hold_ms)
        page.keyboard.up(key)
        page.wait_for_timeout(100)


def map_check(page, stats: dict) -> tuple[str, int, str]:
    """Automap: level name + explored-pixel count + location fingerprint."""
    page.keyboard.press("Tab")
    page.wait_for_timeout(550)
    level = ocr(page)
    bright = int(page.evaluate(BRIGHT_JS))
    bits = str(page.evaluate(FP_JS))
    page.keyboard.press("Tab")
    page.wait_for_timeout(450)
    return level, bright, bits


def reload_page(page) -> None:
    """Reload the game page and wait for the emulator bundle."""
    try:
        page.goto("https://dos.zone/doom-dec-1993", wait_until="domcontentloaded")
        page.wait_for_timeout(26000)
    except Exception as e:  # noqa: BLE001
        log(f"reload error: {e}")


def full_check(page, stats: dict) -> None:
    """Every ~20 steps: menu / intermission handling."""
    txt = ocr(page)
    u = norm(txt)
    log(f"screen OCR: {txt[:110]}")
    if "NEW GAME" in u and "QUIT" in u:
        log("menu detected -> closing")
        page.keyboard.press("Escape")
        page.wait_for_timeout(600)
    if "FINISHED" in u or ("KILLS" in u and "ITEMS" in u):
        log("intermission detected -> advancing")
        page.keyboard.press("Enter")
        page.wait_for_timeout(1200)
        page.keyboard.press("Enter")
        page.wait_for_timeout(1200)


def play(page, cfg, max_steps: int, deadline: float) -> None:
    hist: list[str] = []
    blocked = False
    since_turn = 0
    health = 100
    last_bright = 0
    loops = 0
    fp_hist: list[tuple[int, str]] = []
    stats = {"steps": 0, "level_ocr": "", "map_bright": 0, "deaths": 0,
             "errors": 0, "misses": 0}

    for step in range(1, max_steps + 1):
        if time.time() > deadline:
            log("deadline reached")
            break
        try:
            # --- perception (vision) ---
            img = b64(page.evaluate(CAP_JS))
            per = sys12.decide(cfg, "DOOM first-person screenshot. Recent: "
                               + (" -> ".join(hist[-3:]) or "none")
                               + (". A wall blocks forward movement." if blocked else ""),
                               {"ahead": {"type": "choice",
                                          "instructions": "What is in the center of the view, directly in front of the player?",
                                          "criteria": {"A": "open corridor or clear path",
                                                       "B": "a wall or obstacle very close",
                                                       "C": "a door", "D": "a monster or enemy",
                                                       "E": "unclear"}},
                                "enemy": {"type": "noul",
                                          "instructions": "Is a monster visible anywhere in this image?",
                                          "criteria": {"true": "a monster is visible",
                                                       "false": "no monster is visible"}}},
                               images=[img])
            ah = (per.get("answers", {}).get("ahead", {}) or {}).get("choice", "A")
            enemy_p = float((per.get("answers", {}).get("enemy", {}) or {}).get("noul") or 0)

            # --- action choice (only when there is a real choice) ---
            action = "forward"
            if ah == "D" or enemy_p > 0.7:
                opts = {"fire": "fire the gun", "strafe_left": "strafe left",
                        "strafe_right": "strafe right", "back": "back up"}
                crit = {chr(65 + i): v for i, v in enumerate(opts.values())}
                act = sys12.decide(cfg, f"DOOM. ahead={ah}, enemy_p={enemy_p:.2f}, blocked={blocked}. "
                                   f"Recent: {' -> '.join(hist[-3:]) or 'none'}.",
                                   {"act": {"type": "choice",
                                            "instructions": "Choose the single best next input.",
                                            "criteria": crit}})
                choice = (act.get("answers", {}).get("act", {}) or {}).get("choice", "A")
                idx = list(crit.keys()).index(choice) if choice in crit else 0
                action = list(opts.keys())[idx]

            # right-hand rule: always turn right at walls/doors, never flip
            if blocked or ah == "B":
                seq = ["Space", "ArrowRight", "w"]
                action = "wall_follow_R"
            elif ah == "C":
                seq = ["Space", "w"]
                action = "open"
            elif action == "forward":
                seq = ["Space", KEYS["forward"]]
            else:
                seq = [KEYS.get(action, "w")]

            page.evaluate(DIFF_JS)
            for k in seq:
                press(page, k)
            diff = float(page.evaluate(DIFF_JS))
            moved = diff > 0.01
            blocked = not moved
            since_turn = 0 if "turn" in action else since_turn + 1
            hist.append(action)
            if len(hist) > 6:
                hist.pop(0)
            stats["steps"] = step

            # --- health check every 5 steps (~10 s) ---
            if step % 5 == 0:
                htxt = hud_ocr(page)
                if "HEALTH" in htxt.upper():
                    h = parse_health(htxt)
                    if h is not None:
                        health = h
                    if h == 0:
                        stats["deaths"] += 1
                        log(f"DEATH detected (health 0) at step {step} -> respawning")
                        page.keyboard.press("Space")
                        page.wait_for_timeout(1500)
                        page.keyboard.press("Enter")
                        page.wait_for_timeout(1500)
                        blocked = False
                log(f"step {step}: ahead={ah} enemy={enemy_p:.2f} act={action} diff={diff:.2f} hp={health} loops={loops}")

            # --- map check every 10 steps (~20 s) ---
            if step % 10 == 0:
                level, bright, bits = map_check(page, stats)
                if not any(k in norm(level) for k in ("E1M", "E2M", "E3M", "E4M")):
                    log(f"automap not detected ({level[:60]}) -> full screen check")
                    full_check(page, stats)
                    if not focus_game(page):
                        stats["misses"] += 1
                    continue
                # loop detection: compare location fingerprint with history
                best = 999
                for (pstep, pbits) in fp_hist:
                    if step - pstep >= 6:
                        dist = sum(1 for a, b in zip(bits, pbits) if a != b)
                        best = min(best, dist)
                fp_hist.append((step, bits))
                if len(fp_hist) > 80:
                    fp_hist.pop(0)
                growth = bright - last_bright if last_bright else 0
                last_bright = bright
                if level:
                    stats["level_ocr"] = level[:70]
                stats["map_bright"] = bright
                revisit = best <= 24
                log(f"map: {level[:50]} | bright={bright} (+{growth}) fpdist={best} revisit={revisit}")
                if revisit:
                    loops += 1
                    log(f"LOOP detected ({loops}) -> use-sweep (Space at multiple angles)")
                    for _k in range(12):
                        page.keyboard.press("Space")
                        page.wait_for_timeout(90)
                        page.keyboard.down("ArrowRight")
                        page.wait_for_timeout(70)
                        page.keyboard.up("ArrowRight")
                        page.wait_for_timeout(40)
                    page.keyboard.down("Shift")
                    page.keyboard.down("w")
                    page.wait_for_timeout(1600)
                    page.keyboard.up("w")
                    page.keyboard.up("Shift")
                    page.wait_for_timeout(200)
                    blocked = False
                else:
                    loops = 0
                with JSONL.open("a") as f:
                    f.write(json.dumps({"t": time.time(), **stats, "fpdist": best}) + "\n")
                if not focus_game(page):
                    stats["misses"] += 1
                    log(f"unresponsive at map check (miss {stats['misses']}/3)")

            # --- full screen check every 20 steps (~40 s) ---
            if step % 20 == 0:
                full_check(page, stats)
                if not focus_game(page):
                    stats["misses"] += 1
                else:
                    stats["misses"] = 0

        except Exception as e:  # noqa: BLE001
            stats["errors"] += 1
            log(f"step {step} error: {type(e).__name__}: {e}")
            try:
                focus_game(page)
            except Exception:  # noqa: BLE001
                time.sleep(2)

    log(f"play loop ended: {json.dumps(stats)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--minutes", type=float, default=360)
    args = ap.parse_args()

    cfg = sys12.load_config()
    deadline = time.time() + args.minutes * 60
    log(f"=== autoplay v2 start: system1={cfg['system1']} steps={args.steps} minutes={args.minutes} ===")

    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE), channel="chrome", headless=False,
            viewport={"width": 1280, "height": 900},
            args=["--window-size=1280,920", "--no-first-run"],
        )
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        try:
            cdp = ctx.new_cdp_session(page)
            cdp.send("Emulation.setFocusEmulationEnabled", {"enabled": True})
            cdp.send("Page.setWebLifecycleState", {"state": "active"})
            log("focus emulation enabled (input works even if window is behind)")
        except Exception as e:  # noqa: BLE001
            log(f"focus emulation failed: {e}")
        page.goto("https://dos.zone/doom-dec-1993", wait_until="domcontentloaded")
        log("page loaded, waiting for emulator…")
        page.wait_for_timeout(26000)
        focus_game(page)
        if not start_game(page):
            log("ERROR: could not start the game; aborting instead of playing menus")
            ctx.close()
            return
        focus_game(page)
        if not verify_in_game(page):
            log("ERROR: not in a level after start; aborting")
            ctx.close()
            return
        play(page, cfg, args.steps, deadline)
        ctx.close()
    log("=== autoplay done ===")


if __name__ == "__main__":
    main()
