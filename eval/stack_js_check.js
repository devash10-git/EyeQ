/* Do the canthus features and the kernel ridge stack, in the browser code?
 *
 * Runs the real iris-tracker.js fit/predict over the recorded sessions in a
 * 2x2: deployed-8 vs canthus-12, linear ridge vs kernel ridge. Held-out
 * targets, the split the protocol reserved.
 *
 *   node eval/stack_js_check.js
 *
 * Scope: samples are injected as recorded feature vectors, so this exercises
 * the model path, not _features() landmark arithmetic. That formula is shared
 * verbatim with eval/capture.html, which produced these recordings.
 */
const fs = require("fs"), path = require("path"), vm = require("vm");

function loadTracker(search) {
  const ctx = { window: { innerWidth: 1470, innerHeight: 867 },
    localStorage: { getItem: () => null, setItem() {}, removeItem() {} },
    location: { search }, URLSearchParams,
    performance: { now: () => Date.now() }, requestAnimationFrame: () => {},
    document: { createElement: () => ({ getContext: () => null }) }, console };
  ctx.globalThis = ctx; vm.createContext(ctx);
  vm.runInContext(fs.readFileSync(path.join(__dirname, "..", "static", "iris-tracker.js"), "utf8"), ctx);
  return ctx;
}

const dir = path.join(__dirname, "sessions");
const files = fs.readdirSync(dir).filter(f => f.endsWith(".json")).sort();
const MODES = [
  ["deployed 8  + ridge", "?features=deployed&model=ridge"],
  ["deployed 8  + kernel", "?features=deployed&model=krr"],
  ["canthus 12  + ridge", "?features=canthus&model=ridge"],
  ["canthus 12  + kernel", "?features=canthus&model=krr"],
];

const rows = {};
let anyFail = false;
for (const file of files) {
  const d = JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
  const S = d.screen, sm = d.samples || [];
  const dropped = d.dropped_frames || 0;
  if (new Set(sm.map(s => s.target_id)).size < 21 || dropped / (dropped + sm.length) > 0.2) continue;

  const fit = sm.filter(s => s.split === "fit"), hold = sm.filter(s => s.split === "holdout");
  for (const [label, search] of MODES) {
    const ctx = loadTracker(search);
    ctx.window.innerWidth = S.w_px; ctx.window.innerHeight = S.h_px;
    const t = new ctx.window.IrisTracker({ video: null, overlay: null });
    const wide = search.includes("canthus");
    const vec = s => wide ? s.f.concat(s.canthus || [0, 0, 0, 0]) : s.f;
    t.samples = fit.map(s => ({ f: vec(s), x: s.target_px[0] / S.w_px, y: s.target_px[1] / S.h_px, w: 1 }));
    t._fit();
    let se = 0;
    for (const s of hold) {
      const f = vec(s);
      const k = t.krr ? t._krrPredict(f) : null;
      const px = (k ? k[0] : t._dot(t.wx, f)) * S.w_px;
      const py = (k ? k[1] : t._dot(t.wy, f)) * S.h_px;
      se += (px - s.target_px[0]) ** 2 + (py - s.target_px[1]) ** 2;
    }
    const rmse = Math.sqrt(se / hold.length);
    (rows[label] = rows[label] || []).push({ rmse, deg: rmse / d.px_per_degree, kernel: !!t.krr });
  }
}

const base = rows["deployed 8  + ridge"];
const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
const basePx = mean(base.map(r => r.rmse));
console.log(`\n2x2 on ${base.length} usable session(s), held-out targets\n`);
console.log(`  ${"configuration".padEnd(22)}${"px".padStart(8)}${"deg".padStart(8)}${"vs deployed".padStart(14)}   kernel engaged`);
for (const [label, rs] of Object.entries(rows)) {
  const px = mean(rs.map(r => r.rmse)), deg = mean(rs.map(r => r.deg));
  const gain = (1 - px / basePx) * 100;
  const tag = label === "deployed 8  + ridge" ? "baseline" : `${gain >= 0 ? "+" : ""}${gain.toFixed(1)}%`;
  console.log(`  ${label.padEnd(22)}${px.toFixed(1).padStart(8)}${deg.toFixed(2).padStart(8)}${tag.padStart(14)}   ${rs.map(r => r.kernel ? "yes" : "no").join(", ")}`);
}
const best = Object.entries(rows).map(([l, rs]) => [l, mean(rs.map(r => r.rmse))]).sort((a, b) => a[1] - b[1])[0];
console.log(`\n  best: ${best[0]} at ${best[1].toFixed(1)} px`);
const canthusRidge = mean(rows["canthus 12  + ridge"].map(r => r.rmse));
const canthusKrr = mean(rows["canthus 12  + kernel"].map(r => r.rmse));
console.log(`  does the kernel still add anything on top of canthus? ${((1 - canthusKrr / canthusRidge) * 100).toFixed(1)}%`);
if (mean(rows["canthus 12  + ridge"].map(r => r.rmse)) >= basePx) { console.log("  !! canthus did not beat the deployed baseline"); anyFail = true; }
process.exit(anyFail ? 1 : 0);
