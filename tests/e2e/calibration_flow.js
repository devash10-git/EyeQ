/* Drives the full 9-point calibration with a fake webcam that MediaPipe really
 * tracks, then checks that gaze — not the mouse — types.
 *
 * This is the one flow that could not be tested before: the Claude Code browser
 * pane blocks getUserMedia, so `gazeReady` stayed false and every calibration
 * click returned early. Chrome's fake-capture flags replace the camera with a
 * looping portrait, which the landmarker treats as a genuine face.
 *
 *   node tests/e2e/calibration_flow.js      (server must be running)
 *
 * The face is static, so the nine dots all produce near-identical features.
 * That is fine for testing the *flow* — samples recorded, model fitted, kernel
 * engaged, gaze drives the cursor — and tells you nothing about accuracy. Any
 * accuracy claim needs eval/capture.py with a real person.
 */
const puppeteer = require("puppeteer-core");
const path = require("path");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const PORT = process.env.PORT || "8321";
const FACE = path.join(__dirname, "..", "..", "tools", "fixtures", "face.y4m");

(async () => {
  const browser = await puppeteer.launch({
    executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    headless: "new",
    args: ["--no-sandbox",
           "--use-fake-ui-for-media-stream",          // auto-grant, no dialog
           "--use-fake-device-for-media-stream",
           `--use-file-for-fake-video-capture=${FACE}`],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 800 });
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto(`http://127.0.0.1:${PORT}/`, { waitUntil: "domcontentloaded" });

  const T = {};

  // ---- 1. the camera opens and the landmarker locks on -------------------
  let state = null;
  for (let i = 0; i < 120; i++) {                 // CDN + model load can take 30 s
    state = await page.evaluate(() => window.__gaze.state());
    if (state.faceState === "ok") break;
    await sleep(500);
  }
  T.cameraOpens = state.cameraState.startsWith("on") || state.cameraState;
  T.faceTracked = state.faceState === "ok" || state;
  if (state.faceState !== "ok") {
    console.log(JSON.stringify({ results: T, failed: ["faceTracked"], verdict: "FAIL", state }, null, 2));
    await browser.close(); process.exit(1);
  }

  // ---- 2. all nine spots calibrate ---------------------------------------
  const spots = await page.evaluate(() =>
    [...document.querySelectorAll("button.calib")].map(b => b.dataset.corner));
  T.nineSpotsPresent = spots.length === 9 || spots;

  for (const id of spots) {
    await page.click(`#calib-${id}`);
    for (let c = 0; c < 5; c++) { await page.click("#calib-dot"); await sleep(90); }
    await sleep(120);
  }
  const after = await page.evaluate(() => ({
    calibrated: window.__gaze.state().calibrated,
    samples: window.__gaze.getTracker().samples.length,
    fitted: window.__gaze.getTracker().fitted,
    krr: !!window.__gaze.getTracker().krr,
    model: window.__gaze.getTracker().model,
    groups: window.__gaze.getTracker()._krrGroups().length,
  }));
  T.allNineRegistered = after.calibrated === 9 || after;
  T.samplesRecorded = after.samples >= 45 || after.samples;
  T.modelFitted = after.fitted === true || after;
  /* Assert the property, not the current default.
   *
   * This previously required model === "krr" and so failed the moment the
   * default was reverted to the linear ridge — a test tracking a config value
   * rather than a behaviour. What matters is that whichever model is selected
   * produces a usable fit: the linear solution is always present, and the
   * kernel either engages or declines, both of which are correct. (A static
   * fixture face gives all nine dots near-identical features, so declining is
   * the expected outcome here.) */
  T.modelModeValid = ["ridge", "krr"].includes(after.model) || after.model;
  T.linearAlwaysFitted = after.fitted === true || after;
  T.kernelOnlyWhenSelected = (after.model === "ridge" ? after.krr === false : true)
    || { model: after.model, krr: after.krr };

  // ---- 3. gaze drives the cursor, with the mouse parked ------------------
  await page.mouse.move(3, 3);
  const before = await page.evaluate(() => window.__gaze.state().predictionCount);
  await sleep(1500);
  const now = await page.evaluate(() => window.__gaze.state());
  T.gazeFramesFlowing = now.predictionCount > before || { before, after: now.predictionCount };
  T.notInMouseMode = now.mouseMode === false || now;

  T.pageErrors = errors.length === 0 ? true : errors.slice(0, 3);

  await browser.close();
  const failed = Object.entries(T).filter(([, v]) => v !== true).map(([k]) => k);
  console.log(JSON.stringify({ results: T, failed, verdict: failed.length ? "FAIL" : "ALL PASS" }, null, 2));
  process.exit(failed.length ? 1 : 0);
})();
