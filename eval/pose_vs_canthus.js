/* The literature's head-pose treatment vs this project's canthus frame.
 *
 * Both run through the real iris-tracker.js fit/predict, on the same recorded
 * sessions, held-out targets, reported in degrees of visual angle — so this is
 * a like-for-like comparison rather than a percentage from a different problem.
 *
 *   node eval/pose_vs_canthus.js
 */
const fs = require("fs"), path = require("path"), vm = require("vm");

function load(search) {
  const ctx = { window: { innerWidth: 1470, innerHeight: 867 },
    localStorage: { getItem: () => null, setItem() {}, removeItem() {} },
    location: { search }, URLSearchParams, performance: { now: () => Date.now() },
    requestAnimationFrame: () => {}, document: { createElement: () => ({ getContext: () => null }) }, console };
  ctx.globalThis = ctx; vm.createContext(ctx);
  vm.runInContext(fs.readFileSync(path.join(__dirname, "..", "static", "iris-tracker.js"), "utf8"), ctx);
  return ctx;
}
// Feature vectors built exactly as static/iris-tracker.js::_features would.
const VEC = {
  deployed: s => s.f,
  canthus: s => s.f.concat(s.canthus || [0, 0, 0, 0]),
  pose: s => {
    const h = (s.f[1] + s.f[2]) / 2, v = (s.f[3] + s.f[4]) / 2;
    const yaw = s.yaw || 0, pitch = s.pitch || 0;
    return s.f.concat([yaw, pitch, h * yaw, v * pitch, h * pitch, v * yaw]);
  },
};
const MODES = [
  ["deployed 8      (baseline)", "deployed", "?features=deployed&model=ridge"],
  ["+ head pose 14  (literature)", "pose", "?features=pose&model=ridge"],
  ["+ canthus 12    (this project)", "canthus", "?features=canthus&model=ridge"],
];

const dir = path.join(__dirname, "sessions");
const rows = {}, headMove = [];
for (const file of fs.readdirSync(dir).filter(f => f.endsWith(".json")).sort()) {
  const d = JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
  const S = d.screen, sm = d.samples || [], dropped = d.dropped_frames || 0;
  if (new Set(sm.map(s => s.target_id)).size < 21 || dropped / (dropped + sm.length) > 0.2) continue;

  const deg = x => x * 180 / Math.PI;
  headMove.push({
    session: file.slice(8, 24),
    yawDeg: deg(Math.max(...sm.map(s => s.yaw)) - Math.min(...sm.map(s => s.yaw))),
    pitchDeg: deg(Math.max(...sm.map(s => s.pitch)) - Math.min(...sm.map(s => s.pitch))),
  });

  const fit = sm.filter(s => s.split === "fit"), hold = sm.filter(s => s.split === "holdout");
  for (const [label, key, search] of MODES) {
    const ctx = load(search);
    ctx.window.innerWidth = S.w_px; ctx.window.innerHeight = S.h_px;
    const t = new ctx.window.IrisTracker({ video: null, overlay: null });
    t.samples = fit.map(s => ({ f: VEC[key](s), x: s.target_px[0] / S.w_px, y: s.target_px[1] / S.h_px, w: 1 }));

    const t0 = process.hrtime.bigint(); t._fit();
    const fitMs = Number(process.hrtime.bigint() - t0) / 1e6;

    let se = 0;
    const p0 = process.hrtime.bigint();
    for (const s of hold) {
      const f = VEC[key](s);
      const px = t._dot(t.wx, f) * S.w_px, py = t._dot(t.wy, f) * S.h_px;
      se += (px - s.target_px[0]) ** 2 + (py - s.target_px[1]) ** 2;
    }
    const perFrameUs = Number(process.hrtime.bigint() - p0) / 1000 / hold.length;
    const rmse = Math.sqrt(se / hold.length);
    (rows[label] = rows[label] || []).push({ rmse, deg: rmse / d.px_per_degree,
      keys: rmse / d.key_pitch_px, fitMs, perFrameUs, dims: VEC[key](hold[0]).length });
  }
}

const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
console.log("\nHead movement actually present in these recordings");
for (const h of headMove) console.log(`  ${h.session}   yaw ${h.yawDeg.toFixed(1)}deg   pitch ${h.pitchDeg.toFixed(1)}deg`);
console.log("\nHeld-out accuracy, same sessions, same code path\n");
console.log(`  ${"feature set".padEnd(32)}${"dims".padStart(5)}${"px".padStart(9)}${"deg".padStart(8)}${"key pitches".padStart(13)}${"vs base".padStart(10)}`);
const base = mean(rows[MODES[0][0]].map(r => r.rmse));
for (const [label] of MODES) {
  const r = rows[label];
  const px = mean(r.map(x => x.rmse));
  const tag = label.startsWith("deployed") ? "baseline" : `${((1 - px / base) * 100).toFixed(1)}%`;
  console.log(`  ${label.padEnd(32)}${r[0].dims.toString().padStart(5)}${px.toFixed(1).padStart(9)}` +
              `${mean(r.map(x => x.deg)).toFixed(2).padStart(8)}${mean(r.map(x => x.keys)).toFixed(2).padStart(13)}${tag.padStart(10)}`);
}
console.log("\nReal-time cost");
console.log(`  ${"feature set".padEnd(32)}${"calibration fit".padStart(17)}${"per frame".padStart(13)}`);
for (const [label] of MODES) {
  const r = rows[label];
  console.log(`  ${label.padEnd(32)}${(mean(r.map(x => x.fitMs)).toFixed(1) + " ms").padStart(17)}` +
              `${(mean(r.map(x => x.perFrameUs)).toFixed(1) + " us").padStart(13)}`);
}
