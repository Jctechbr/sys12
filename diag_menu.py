import base64
import subprocess
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
OCR = HERE / "ocr"
SHOT = Path(tempfile.gettempdir()) / "sys12-diag.png"
JS = """() => { const c=document.querySelector('canvas'); const o=document.createElement('canvas');
  o.width=c.width*3; o.height=c.height*3; const x=o.getContext('2d');
  x.imageSmoothingEnabled=false; x.drawImage(c,0,0,o.width,o.height); return o.toDataURL('image/png'); }"""


def ocr(page, tag):
    SHOT.write_bytes(base64.b64decode(page.evaluate(JS).split(",", 1)[1]))
    t = subprocess.run([str(OCR), str(SHOT)], capture_output=True, text=True,
                       timeout=25).stdout
    print(f"{tag}: {' '.join(t.split())[:150]}", flush=True)


with sync_playwright() as pw:
    ctx = pw.chromium.launch_persistent_context(user_data_dir=str(Path.home()/"Library/Caches/sys12-chrome"),
        channel="chrome", headless=False, viewport={"width":1280,"height":900},
        args=["--window-size=1280,920","--no-first-run"])
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    cdp = ctx.new_cdp_session(page)
    cdp.send("Emulation.setFocusEmulationEnabled", {"enabled": True})
    page.goto("https://dos.zone/doom-dec-1993", wait_until="domcontentloaded")
    page.wait_for_timeout(26000)
    page.bring_to_front()
    b = page.evaluate("() => { const c=document.querySelector('canvas'); const r=c.getBoundingClientRect(); return {x:r.x+r.width/2,y:r.y+r.height/2}; }")
    page.mouse.click(b["x"], b["y"]); page.wait_for_timeout(400)
    ocr(page, "loaded")
    page.keyboard.press("Escape"); page.wait_for_timeout(800); ocr(page, "after ESC")
    page.keyboard.press("n"); page.wait_for_timeout(900); ocr(page, "after n")
    page.keyboard.press("Enter"); page.wait_for_timeout(900); ocr(page, "after ENTER")
    page.keyboard.press("1"); page.wait_for_timeout(2500); ocr(page, "after 1")
    page.keyboard.press("Tab"); page.wait_for_timeout(700); ocr(page, "automap")
    page.keyboard.press("Tab"); page.wait_for_timeout(500)
    # movement test
    page.evaluate("() => { window.__p=null; }")
    page.keyboard.down('w'); page.wait_for_timeout(600); page.keyboard.up('w'); page.wait_for_timeout(200)
    ocr(page, "after W")
    ctx.close()
