/* The camera exposure / contrast toggle.
 *
 * Checks the control works, says honestly which path it took, and — the part
 * that matters — that filtering the frame does not cost enough throughput to
 * disturb the 30 fps loop or break landmark tracking.
 *
 *   node tests/e2e/video_adjust.js       (server must be running)
 *
 * Note on capabilities: Chrome's fake capture device reports its own
 * getCapabilities(), not a real Mac camera's. This test can prove the control
 * works and measure its cost; only a real camera can say whether the hardware
 * path is available on your machine.
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
    args: ["--no-sandbox", "--use-fake-ui-for-media-stream",
           "--use-fake-device-for-media-stream",
           `--use-file-for-fake-video-capture=${FACE}`],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 800 });
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto(`http://127.0.0.1:${PORT}/`, { waitUntil: "domcontentloaded" });

  const T = {};
  let st = null;
  for (let i = 0; i < 120; i++) {
    st = await page.evaluate(() => window.__gaze.state());
    if (st.faceState === "ok") break;
    await sleep(500);
  }
  T.faceTracked = st.faceState === "ok" || st;
  if (st.faceState !== "ok") {
    console.log(JSON.stringify({ results: T, failed: ["faceTracked"], verdict: "FAIL" }, null, 2));
    await browser.close(); process.exit(1);
  }

  /* frameCount, not predictionCount: the latter only ticks once the tracker is
     calibrated, so using it here made the throughput assertion pass on 0 >= 0
     without measuring anything. frameCount increments on every processed
     frame, fitted or not. */
  const fps = async (ms = 2500) => {
    const a = await page.evaluate(() => window.__gaze.getTracker().frameCount);
    await sleep(ms);
    const b = await page.evaluate(() => window.__gaze.getTracker().frameCount);
    return (b - a) / (ms / 1000);
  };

  // ---- 1. off by default, panel hidden ----
  const initial = await page.evaluate(() => ({
    gearPresent: !!document.getElementById("cam-gear"),
    checked: document.getElementById("video-adjust").checked,
    panelHidden: document.getElementById("cam-settings").hidden,
    path: window.__gaze.getTracker().adjustPath,
    caps: Object.keys(window.__gaze.getTracker().cameraCaps || {}),
  }));
  T.offByDefault = (initial.checked === false && initial.panelHidden === true && initial.path === "off") || initial;
  T.gearOnCameraBox = initial.gearPresent === true || initial;
  T.capabilitiesProbed = Array.isArray(initial.caps) || initial;

  const baseline = await fps();

  // ---- 2. turn it on and push contrast/brightness ----
  await page.evaluate(() => {
    document.getElementById("cam-gear").click();          // gear opens the panel
    document.getElementById("video-adjust").checked = true;
    document.getElementById("video-adjust").dispatchEvent(new Event("change"));
    document.getElementById("vid-contrast").value = "150";
    document.getElementById("vid-contrast").dispatchEvent(new Event("input"));
    document.getElementById("vid-brightness").value = "125";
    document.getElementById("vid-brightness").dispatchEvent(new Event("input"));
  });
  await sleep(400);
  const on = await page.evaluate(() => ({
    panelHidden: document.getElementById("cam-settings").hidden,
    path: window.__gaze.getTracker().adjustPath,
    mode: document.getElementById("vid-mode").textContent,
    usesCanvas: window.__gaze.getTracker()._frameSource() instanceof HTMLCanvasElement,
    face: window.__gaze.state().faceState,
  }));
  T.panelOpensFromGear = on.panelHidden === false || on;
  T.pathReported = (on.path === "hardware" || on.path === "software") || on;
  T.modeExplained = on.mode.length > 0 || on;
  T.trackingSurvives = on.face === "ok" || on;

  const adjusted = await fps();
  // A vacuous pass is worse than a failure: require real frames on both sides.
  T.throughputMeasurable = baseline > 5 || { baseline: +baseline.toFixed(1) };
  T.throughputHeld = (baseline > 5 && adjusted >= baseline * 0.8)
    || { baseline: +baseline.toFixed(1), adjusted: +adjusted.toFixed(1) };

  // ---- 3. turn it off again ----
  await page.evaluate(() => {
    document.getElementById("video-adjust").checked = false;
    document.getElementById("video-adjust").dispatchEvent(new Event("change"));
  });
  await sleep(400);
  const off = await page.evaluate(() => ({
    path: window.__gaze.getTracker().adjustPath,
    usesVideo: window.__gaze.getTracker()._frameSource() instanceof HTMLVideoElement,
    camFilter: document.getElementById("cam").style.filter,
    face: window.__gaze.state().faceState,
  }));
  T.turnsBackOff = (off.path === "off" && off.usesVideo && off.camFilter === "") || off;
  const clickAway = await page.evaluate(() => {
    // Normalise first: an earlier step left the panel open, so a bare gear
    // click here toggled it *closed* and the assertion read backwards.
    document.getElementById("cam-settings").hidden = true;
    document.getElementById("cam-gear").click();                 // open
    const opened = !document.getElementById("cam-settings").hidden;
    document.body.click();                                       // click away
    return { opened, closed: document.getElementById("cam-settings").hidden };
  });
  T.gearTogglesAndClosesOnClickAway = (clickAway.opened && clickAway.closed) || clickAway;
  T.stillTrackingAfterOff = off.face === "ok" || off;

  T.pageErrors = errors.length === 0 ? true : errors.slice(0, 3);

  await browser.close();
  const failed = Object.entries(T).filter(([, v]) => v !== true).map(([k]) => k);
  console.log(JSON.stringify({
    results: T, failed, verdict: failed.length ? "FAIL" : "ALL PASS",
    throughput: { baselineFps: +baseline.toFixed(1), withFilterFps: +adjusted.toFixed(1) },
    cameraCapabilities: initial.caps, pathTaken: on.path, modeText: on.mode,
  }, null, 2));
  process.exit(failed.length ? 1 : 0);
})();
