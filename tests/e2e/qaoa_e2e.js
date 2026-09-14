/* Integration test for the QAOA typing-sequence pipeline.
 *
 * Types an ambiguous word through the real fusion path, then checks that the
 * word-boundary QAOA call proposes the joint-decoded word as a fix chip and
 * that picking it applies. Confident words must skip the call entirely.
 *
 *   node tests/e2e/qaoa_e2e.js       (server must be on :8321)
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
  const consoleErrors = [];
  page.on("pageerror", (e) => consoleErrors.push(String(e)));
  await page.goto("http://127.0.0.1:8321", { waitUntil: "networkidle2", timeout: 30000 });
  // The keyboard now ships the cluster layout by default; this suite exercises
  // the per-letter keyboard, which is still available behind the toggle.
  await page.waitForFunction(() => window.__gaze && window.__gaze.setLayout);
  await page.evaluate(() => window.__gaze.setLayout("letters"));

  await sleep(800);

  const T = {};

  T.assets = await page.evaluate(async () => {
    const src = await (await fetch("/static/app.js?" + Date.now())).text();
    return {
      hasRefine: /refineWord/.test(src),
      hasEvidence: /wordEvidence/.test(src),
      hookPresent: typeof window.__gaze.applyLetterEv === "function",
    };
  });

  await page.evaluate(() => {
    const mt = document.getElementById("mouse-mode");
    mt.checked = false; mt.dispatchEvent(new Event("change"));
    document.getElementById("clear-btn").click();
  });
  await sleep(300);

  // ---- 1. evidence accumulates per letter and pops on backspace ----
  await page.evaluate(() => {
    const g = window.__gaze;
    g.applyLetterEv("t", [["t", 0.55], ["r", 0.45]]);
    g.applyLetterEv("j", [["j", 0.52], ["h", 0.48]]);
  });
  T.evidenceGrows = (await page.evaluate(() => window.__gaze.getEvidence())).length === 2;
  await page.evaluate(() => window.__gaze.backspace());
  T.evidencePops = (await page.evaluate(() => window.__gaze.getEvidence())).length === 1;

  // ---- 2. the canonical rescue: type "tje" ambiguously, SPACE, chip = "the" ----
  await page.evaluate(() => document.getElementById("clear-btn").click());
  await sleep(250);
  await page.evaluate(() => {
    const g = window.__gaze;
    g.applyLetterEv("t", [["t", 0.55], ["r", 0.45]]);
    g.applyLetterEv("j", [["j", 0.52], ["h", 0.48]]);
    g.applyLetterEv("e", [["e", 0.60], ["w", 0.40]]);
    g.commitAction("space");
  });
  await sleep(1400);
  const afterSpace = await page.evaluate(() => ({
    text: window.__gaze.state().text,
    chipsMode: window.__gaze.state().chipsMode,
    fixChips: [...document.querySelectorAll(".chip.fix")].map((c) => c.dataset.fix),
    qaoa: window.__gaze.qaoaLast(),
  }));
  T.typedGreedy = afterSpace.text === "tje ";
  T.qaoaResponded = !!(afterSpace.qaoa && afterSpace.qaoa.available);
  T.qaoaProposedThe = afterSpace.qaoa && afterSpace.qaoa.word === "the" && afterSpace.qaoa.changed;
  T.chipOffered = afterSpace.chipsMode === "fix" && afterSpace.fixChips.includes("the");

  // ---- 3. picking the chip applies the QAOA word ----
  await page.evaluate(() => {
    const chip = [...document.querySelectorAll(".chip.fix")].find((c) => c.dataset.fix === "the");
    if (chip) chip.click();
  });
  await sleep(300);
  T.chipApplies = (await page.evaluate(() => window.__gaze.state().text)) === "the ";

  // ---- 4. confident words do NOT trigger a QAOA call ----
  await page.evaluate(() => document.getElementById("clear-btn").click());
  await sleep(250);
  await page.evaluate(() => {
    const g = window.__gaze;
    for (const ch of "hello") g.commitLetter(ch);
    g.commitAction("space");
  });
  await sleep(1000);
  const conf = await page.evaluate(() => ({
    text: window.__gaze.state().text,
    qaoa: window.__gaze.qaoaLast(),
  }));
  T.confidentSkipsQaoa = conf.text === "hello " && conf.qaoa === null;

  // ---- 5. normal flow regression: suggestions still live ----
  await page.evaluate(() => document.getElementById("clear-btn").click());
  await sleep(250);
  await page.evaluate(() => { for (const ch of "wor") window.__gaze.commitLetter(ch); });
  await sleep(700);
  const chips = await page.evaluate(() => window.__gaze.state().chips);
  T.suggestionsLive = chips.length > 0 && chips.every((w) => w && w.startsWith("wor"));

  T.pageErrors = consoleErrors.length === 0 ? true : consoleErrors.slice(0, 3);

  await browser.close();
  const failed = Object.entries(T).filter(([, v]) =>
    v === false || (v && typeof v === "object" && Object.values(v).includes(false)));
  console.log(JSON.stringify({ results: T, failed: failed.map(([k]) => k),
    verdict: failed.length ? "FAILURES" : "ALL PASS" }, null, 2));
  process.exit(failed.length ? 1 : 0);
})().catch((e) => { console.error("E2E crashed:", e); process.exit(2); });
