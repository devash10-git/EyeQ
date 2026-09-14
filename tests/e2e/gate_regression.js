/* Regression test for the "dwell types nothing" bug.
 *
 * Root cause was a train/serve scale mismatch on eye openness: the movement
 * gate classified every open eye as a blink and refused every keystroke. These
 * checks pin both the fix and the watchdog that must contain any recurrence.
 *
 *   node tests/e2e/gate_regression.js      (server must be on :8321)
 */
const puppeteer = require("puppeteer-core");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  const browser = await puppeteer.launch({
    executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    headless: "new",
    args: ["--no-sandbox", "--use-fake-ui-for-media-stream"],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 800 });
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto("http://127.0.0.1:8321", { waitUntil: "networkidle2", timeout: 30000 });
  await sleep(800);

  const T = {};

  await page.evaluate(() => {
    const mt = document.getElementById("mouse-mode");
    mt.checked = false; mt.dispatchEvent(new Event("change"));
    document.getElementById("clear-btn").click();
  });

  // ---- 1. the openness mapping itself ----
  T.opennessMapping = await page.evaluate(() => {
    const t = Object.create(window.IrisTracker.prototype);
    t.earWindow = []; t.earBaseline = null; t.lastEar = null;
    const before = t.openness();               // unknown eye must read as OPEN
    for (let i = 0; i < 150; i++) {            // narrow-eyed user, occasional blink
      const raw = i % 25 === 0 ? 0.04 : 0.20 + Math.random() * 0.01;
      t.lastEar = raw; t._updateEarBaseline(raw);
    }
    const openNow = t.openness();
    t.lastEar = 0.03;
    const blinkNow = t.openness();
    return {
      unknownReadsOpen: before === 0.78,
      baseline: +t.earBaseline.toFixed(3),
      openIsFixationRange: openNow > 0.7,
      blinkIsBlinkRange: blinkNow < 0.3,
    };
  });

  // ---- 2. a "blink" verdict still blocks a keystroke ----
  await page.evaluate(() => {
    window.__gaze.setMovement({ available: true, name: "blink", usable: false });
    window.__gaze.commitKey("t");
  });
  T.gateStillBlocks = (await page.evaluate(() => window.__gaze.state().text)) === "";

  // ---- 3. watchdog: a gate that refuses everything disables itself ----
  await page.evaluate(() => {
    for (let i = 0; i < 6; i++) window.__gaze.commitKey("t");
  });
  const after = await page.evaluate(() => ({
    text: window.__gaze.state().text,
    disabled: window.__gaze.gateDisabled(),
  }));
  T.watchdogDisabledGate = after.disabled === true;
  T.typingResumesAfterWatchdog = after.text.length > 0;

  // ---- 4. a normal fixation types immediately ----
  await page.evaluate(() => {
    document.getElementById("clear-btn").click();
    window.__gaze.resetGate();
    window.__gaze.setMovement({ available: true, name: "fixation", usable: true });
    for (const ch of "hello") window.__gaze.commitKey(ch);
  });
  await sleep(200);
  T.fixationTypesNormally =
    (await page.evaluate(() => window.__gaze.state().text)) === "hello";

  T.pageErrors = errors.length === 0 ? true : errors.slice(0, 3);

  await browser.close();
  const failed = Object.entries(T).filter(([, v]) =>
    v === false || (v && typeof v === "object" && Object.values(v).includes(false)));
  console.log(JSON.stringify({ results: T, failed: failed.map(([k]) => k),
    verdict: failed.length ? "FAILURES" : "ALL PASS" }, null, 2));
  process.exit(failed.length ? 1 : 0);
})().catch((e) => { console.error("crashed:", e); process.exit(2); });
