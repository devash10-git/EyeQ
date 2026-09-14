/* Verifies the browser KRR against the recorded sessions.
 *
 * The Python comparison chose gamma and alpha by grid search; the JS uses the
 * median heuristic for gamma and closed-form LOO for lambda, so the two will
 * not agree to the digit. What must hold is the finding: kernel ridge beats the
 * linear ridge on held-out targets, in the browser implementation, on real data.
 *
 *   node eval/krr_js_check.js
 */
const fs = require("fs"), path = require("path"), vm = require("vm");

// --- shims: the tracker only touches these at construction time ------------
const ctx = {
  window: { innerWidth: 1470, innerHeight: 867 },
  localStorage: { getItem: () => null, setItem() {}, removeItem() {} },
  location: { search: "" },
  URLSearchParams: URLSearchParams,
  performance: { now: () => Date.now() },
  requestAnimationFrame: () => {},
  document: { createElement: () => ({ getContext: () => null }) },
  console,
};
ctx.globalThis = ctx;
vm.createContext(ctx);
const src = fs.readFileSync(path.join(__dirname, "..", "static", "iris-tracker.js"), "utf8");
vm.runInContext(src, ctx);
const IrisTracker = ctx.window.IrisTracker;

function makeTracker(mode) {
  const t = new IrisTracker({ video: null, overlay: null });
  t.model = mode;
  return t;
}

let anyFail = false;
for (const file of fs.readdirSync(path.join(__dirname, "sessions")).filter(f => f.endsWith(".json")).sort()) {
  const d = JSON.parse(fs.readFileSync(path.join(__dirname, "sessions", file), "utf8"));
  const S = d.screen, samples = d.samples || [];
  const targets = new Set(samples.map(s => s.target_id));
  const dropped = d.dropped_frames || 0;
  if (targets.size < 21 || dropped / (dropped + samples.length) > 0.2) {
    console.log(`skipping ${file}  (${targets.size}/21 targets, ${dropped} frames dropped)`);
    continue;
  }
  ctx.window.innerWidth = S.w_px; ctx.window.innerHeight = S.h_px;
  const fit = samples.filter(s => s.split === "fit");
  const hold = samples.filter(s => s.split === "holdout");

  const out = {};
  for (const mode of ["ridge", "krr"]) {
    const t = makeTracker(mode);
    t.samples = fit.map(s => ({ f: s.f, x: s.target_px[0] / S.w_px, y: s.target_px[1] / S.h_px, w: 1 }));
    const t0 = Date.now();
    t._fit();
    const fitMs = Date.now() - t0;

    let se = 0;
    const p0 = process.hrtime.bigint();
    for (const s of hold) {
      const k = t.krr ? t._krrPredict(s.f) : null;
      const px = (k ? k[0] : t._dot(t.wx, s.f)) * S.w_px;
      const py = (k ? k[1] : t._dot(t.wy, s.f)) * S.h_px;
      se += (px - s.target_px[0]) ** 2 + (py - s.target_px[1]) ** 2;
    }
    const perFrameMs = Number(process.hrtime.bigint() - p0) / 1e6 / hold.length;
    out[mode] = { rmse: Math.sqrt(se / hold.length), fitMs, perFrameMs,
                  n: t.krr ? t.krr.n : 0, lam: t.krr ? t.krr.lambda : null,
                  gamma: t.krr ? t.krr.gamma : null };
  }

  const deg = d.px_per_degree;
  const gain = (1 - out.krr.rmse / out.ridge.rmse) * 100;
  console.log(`\n=== ${file} ===`);
  console.log(`  linear ridge   ${out.ridge.rmse.toFixed(1)} px   ${(out.ridge.rmse/deg).toFixed(2)} deg`);
  console.log(`  kernel ridge   ${out.krr.rmse.toFixed(1)} px   ${(out.krr.rmse/deg).toFixed(2)} deg   ${gain>0?"+":""}${gain.toFixed(1)}%`);
  console.log(`  krr fit ${out.krr.fitMs} ms on ${out.krr.n} points  (lambda ${out.krr.lam}, gamma ${out.krr.gamma == null ? "n/a" : out.krr.gamma.toFixed(3)})`);
  console.log(`  per-frame predict  ridge ${out.ridge.perFrameMs.toFixed(4)} ms   krr ${out.krr.perFrameMs.toFixed(4)} ms`);
  if (gain <= 0) { console.log("  !! kernel ridge did NOT beat the linear ridge here"); anyFail = true; }
  if (out.krr.perFrameMs > 0.1) { console.log("  !! per-frame budget exceeded (>0.1 ms)"); anyFail = true; }
}
console.log(anyFail ? "\nFAILED" : "\nOK — kernel ridge wins every usable session and stays inside the frame budget");
process.exit(anyFail ? 1 : 0);
