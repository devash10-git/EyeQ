/* Does dwelling on a key actually type that key, in the right order?
 *
 * The existing suites drive window.__gaze.commitKey(), which exercises the
 * posterior and the gate but never moves a cursor — so the dwell state machine,
 * the smoothing filter and the hit-test were untested end to end. This closes
 * that gap with real mouse events.
 *
 * Follows the pitfalls documented in .claude/skills/verify:
 *   - synthetic moves need `steps` plus settle moves, or the cursor rests short
 *   - a parked cursor keeps dwell-typing whatever is under it, so park off-key
 *
 *   node tests/e2e/dwell_typing.js       (server must be running)
 */
const puppeteer = require("puppeteer-core");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const PORT = process.env.PORT || "8321";

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
  await page.goto(`http://127.0.0.1:${PORT}/`, { waitUntil: "domcontentloaded" });
  // The keyboard now ships the cluster layout by default; this suite exercises
  // the per-letter keyboard, which is still available behind the toggle.
  await page.waitForFunction(() => window.__gaze && window.__gaze.setLayout);
  await page.evaluate(() => window.__gaze.setLayout("letters"));

  await sleep(1500);

  // Mouse mode: the pointer is exact, so any wrong letter is the dwell path's
  // doing and not gaze error.
  await page.evaluate(() => {
    const cb = document.getElementById("mouse-mode");
    cb.checked = true; cb.dispatchEvent(new Event("change"));
  });

  const geom = await page.evaluate(() => {
    const rects = {};
    for (const k of document.querySelectorAll(".key")) {
      const r = k.getBoundingClientRect();
      const id = k.dataset.letter || k.dataset.action;
      rects[id] = [r.left + r.width / 2, r.top + r.height / 2, r.width, r.height];
    }
    // a point that is neither key nor chip, so parking can never type
    let park = [2, 2];
    outer: for (let y = 4; y < innerHeight; y += 8)
      for (let x = 4; x < innerWidth; x += 40) {
        const e = document.elementFromPoint(x, y);
        if (e && !e.closest(".key") && !e.closest(".chip") && !e.closest("button")) { park = [x, y]; break outer; }
      }
    return { rects, park };
  });

  const textNow = () => page.evaluate(() => window.__gaze.state().text);

  async function park(ms = 500) {
    await page.mouse.move(geom.park[0], geom.park[1], { steps: 8 });
    await sleep(ms);
  }
  /** Move onto a key, settle, and wait for the text to grow. */
  async function dwell(id, offsetFrac = 0, timeout = 3500) {
    const [cx, cy, w] = geom.rects[id];
    const x = cx + w * offsetFrac;
    const before = await textNow();
    await page.mouse.move(x, cy, { steps: 12 });
    for (let i = 0; i < 4; i++) await page.mouse.move(x + (i % 2), cy, { steps: 1 });
    const t0 = Date.now();
    while (Date.now() - t0 < timeout) {
      await sleep(40);
      if ((await textNow()) !== before) return Date.now() - t0;
      await page.mouse.move(x + (Date.now() % 2), cy, { steps: 1 });   // keep frames coming
    }
    return "TIMEOUT";
  }
  async function typeWord(word, offsetFrac = 0) {
    await page.evaluate(() => document.getElementById("clear-btn").click());
    await park(500);
    const times = [];
    for (const ch of word) { times.push(await dwell(ch, offsetFrac)); await park(500); }
    return { typed: await textNow(), times };
  }

  const T = {};
  const cat = await typeWord("cat");
  T.typesCat = cat.typed === "cat" || cat;
  const hello = await typeWord("hello");           // double letter -> re-arm path
  T.typesDoubleLetter = hello.typed === "hello" || hello;
  const edges = await typeWord("qpzm");            // the four corner keys
  T.typesCornerKeys = edges.typed === "qpzm" || edges;
  const off = await typeWord("cat", 0.35);         // 35% toward the neighbour
  T.typesOffCentre = off.typed === "cat" || off;

  // a parked cursor must not type
  await page.evaluate(() => document.getElementById("clear-btn").click());
  await park(2500);
  T.parkedCursorTypesNothing = (await textNow()) === "";

  T.pageErrors = errors.length === 0 ? true : errors.slice(0, 3);

  await browser.close();
  const failed = Object.entries(T).filter(([, v]) => v !== true).map(([k]) => k);
  console.log(JSON.stringify({ results: T, failed, verdict: failed.length ? "FAIL" : "ALL PASS" }, null, 2));
  process.exit(failed.length ? 1 : 0);
})();
