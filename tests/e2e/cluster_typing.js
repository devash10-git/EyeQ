/* Does the cluster keyboard actually turn gazed regions into words?
 *
 * The cluster layout exists for a measured reason: on the recorded sessions the
 * gaze error is 1.0-1.7x the width of a single QWERTY key, so the fused
 * posterior almost never clears THETA and dwelling produces choice chips rather
 * than text. A cluster is ~3x wider, which puts the error at 0.30-0.50x the
 * target. At the same THETA=0.80 that raises commits from 10.4%/2.9%/0.3% of
 * fixations to 52%/36%/5%, at 93-96% selection accuracy.
 *
 * What that buys is ambiguity: "hello" and "fellow" are both 51663, so the word
 * is resolved at the space by decoder.py. This suite checks the whole chain --
 * region dwell -> digits -> /api/decode -> words -> consistency -- rather than
 * any one link.
 *
 * Run: node tests/e2e/cluster_typing.js       (server must be running)
 */

const puppeteer = require("puppeteer-core");

const PORT = process.env.PORT || "8321";
const T = {};
const failed = [];

function check(name, cond, detail) {
  T[name] = cond ? true : (detail || false);
  if (!cond) failed.push(name);
}

/* Commit a cluster by its centre. Sigma is forced small so the geometry is not
 * what is under test here -- the chain is. Whether a real fixation clears
 * THETA is a question about screen size and is measured separately. */
async function gaze(page, sel) {
  const box = await page.evaluate((s) => {
    const el = document.querySelector(s);
    const r = el.getBoundingClientRect();
    return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
  }, sel);
  await page.evaluate((x, y) => window.__gaze.smartCommitAt(x, y), box.x, box.y);
  await new Promise((r) => setTimeout(r, 120));
}

(async () => {
  const browser = await puppeteer.launch({
    executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    headless: "new",
    args: ["--no-sandbox", "--use-fake-ui-for-media-stream"],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e).slice(0, 200)));

  await page.goto(`http://127.0.0.1:${PORT}/`, { waitUntil: "networkidle2" });
  await page.waitForFunction(() => window.__gaze && window.__gaze.setLayout);
  await page.evaluate(() => {
    window.__gaze.setLayout("clusters");
    window.__gaze.setSigma(20);
  });

  // ---- the layout is the one that ships ----
  const layout = await page.evaluate(() => window.__gaze.state().layout);
  check("defaultLayoutIsClusters", layout === "clusters", layout);

  const shape = await page.evaluate(() => {
    const ks = [...document.querySelectorAll("[data-cluster]")];
    const letters = [...document.querySelectorAll("[data-letter]")];
    const vis = (el) => el.getBoundingClientRect().width > 0;
    return {
      clusters: ks.length,
      clustersVisible: ks.every(vis),
      lettersHidden: letters.every((el) => !vis(el)),
      digits: ks.map((k) => k.dataset.cluster).join(""),
    };
  });
  check("eightClusterKeys", shape.clusters === 8, shape.clusters);
  check("clusterKeysVisible", shape.clustersVisible);
  check("letterKeysHiddenByDefault", shape.lettersHidden);
  check("clusterDigitsMatchClustersPy", shape.digits === "12345679", shape.digits);

  // ---- gazing regions builds a code, not letters ----
  for (const d of ["5", "1", "6", "6", "3"]) await gaze(page, `[data-cluster="${d}"]`);
  const built = await page.evaluate(() => window.__gaze.state());
  check("regionsBuildACode", built.clusterCode === "51663", built.clusterCode);
  check("noTextBeforeSpace", built.text.trim() === "", built.text);

  // ---- candidates for the partial code are offered ----
  const cands = await page.evaluate(async () => {
    const r = await fetch("/api/cluster_suggest?code=51663&n=4");
    return (await r.json()).words.map((w) => w.word);
  });
  check("candidatesOfferedForCode", cands.includes("hello"), cands.join(","));

  // ---- space resolves the code into a word ----
  await gaze(page, '[data-action="space"]');
  await new Promise((r) => setTimeout(r, 500));
  let st = await page.evaluate(() => window.__gaze.state());
  check("spaceDecodesTheWord", st.text.trim() === "hello", st.text);
  check("codeClearedAfterSpace", st.clusterCode === "", st.clusterCode);

  // ---- a second and third word, with sentence context ----
  for (const d of ["2", "5", "1"]) await gaze(page, `[data-cluster="${d}"]`);
  await gaze(page, '[data-action="space"]');
  await new Promise((r) => setTimeout(r, 400));
  for (const d of ["9", "1", "4", "2"]) await gaze(page, `[data-cluster="${d}"]`);
  await gaze(page, '[data-action="space"]');
  await new Promise((r) => setTimeout(r, 600));
  st = await page.evaluate(() => window.__gaze.state());
  check("decodesAWholeSentence", st.text.trim() === "hello the best", st.text);
  check("codeWordsRecorded",
        JSON.stringify(st.codeWords) === JSON.stringify(["51663", "251", "9142"]),
        JSON.stringify(st.codeWords));

  // ---- the consistency guarantee actually holds ----
  check("decodeReportsConsistent", st.lastDecode && st.lastDecode.consistent === true,
        st.lastDecode && st.lastDecode.method);
  const reencodes = await page.evaluate(async (text) => {
    const r = await fetch("/api/decode", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clusters: "51663 251 9142" }),
    });
    const d = await r.json();
    return d.text.trim() === text.trim();
  }, st.text);
  check("outputReEncodesToGazedClusters", reencodes);

  // ---- backspace peels a digit, then steps back over a word ----
  for (const d of ["5", "1"]) await gaze(page, `[data-cluster="${d}"]`);
  await gaze(page, '[data-action="backspace"]');
  st = await page.evaluate(() => window.__gaze.state());
  check("backspacePeelsADigit", st.clusterCode === "5", st.clusterCode);
  await gaze(page, '[data-action="backspace"]');
  await gaze(page, '[data-action="backspace"]');
  await new Promise((r) => setTimeout(r, 500));
  st = await page.evaluate(() => window.__gaze.state());
  check("backspaceStepsBackOverAWord", st.codeWords.length === 2,
        JSON.stringify(st.codeWords));

  // ---- the toggle still reaches the letter keyboard ----
  await page.evaluate(() => window.__gaze.setLayout("letters"));
  const back = await page.evaluate(() => {
    const vis = (el) => el.getBoundingClientRect().width > 0;
    return {
      layout: window.__gaze.state().layout,
      lettersVisible: [...document.querySelectorAll("[data-letter]")].every(vis),
      clustersHidden: [...document.querySelectorAll("[data-cluster]")].every((el) => !vis(el)),
    };
  });
  check("toggleRestoresLetterKeys",
        back.layout === "letters" && back.lettersVisible && back.clustersHidden,
        JSON.stringify(back));

  check("pageErrors", errors.length === 0, errors.slice(0, 3));

  await browser.close();
  console.log(JSON.stringify(T, null, 2));
  if (failed.length) {
    console.log("\nFAILED: " + failed.join(", "));
    process.exit(1);
  }
  console.log("\nALL PASS");
})();
