/* Are the canthus features actually invariant to head roll?
 *
 * The claim behind projecting into the eye frame is a structural one: rotating
 * the whole face in the image must leave the features unchanged, because the
 * axis they are measured against rotates with it. That is testable exactly, so
 * it is tested exactly rather than argued.
 *
 *   node eval/eye_frame_check.js
 */
const fs = require("fs"), path = require("path"), vm = require("vm");

const ctx = { window: { innerWidth: 1470, innerHeight: 867 },
  localStorage: { getItem: () => null, setItem() {}, removeItem() {} },
  location: { search: "" }, URLSearchParams, performance: { now: () => Date.now() },
  requestAnimationFrame: () => {}, document: { createElement: () => ({ getContext: () => null }) }, console };
ctx.globalThis = ctx; vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(__dirname, "..", "static", "iris-tracker.js"), "utf8"), ctx);
const t = new ctx.window.IrisTracker({ video: null, overlay: null });

/** A face looking slightly off-centre, so the offsets are non-zero. */
function face(irisDX = 0.010, irisDY = 0.004) {
  const l = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5 }));
  l[33]  = { x: 0.360, y: 0.450 };  l[133] = { x: 0.440, y: 0.450 };   // left outer/inner
  l[159] = { x: 0.400, y: 0.432 };  l[145] = { x: 0.400, y: 0.468 };   // left lids
  l[468] = { x: 0.400 + irisDX, y: 0.450 + irisDY };                   // left iris
  l[362] = { x: 0.560, y: 0.450 };  l[263] = { x: 0.640, y: 0.450 };   // right inner/outer
  l[386] = { x: 0.600, y: 0.432 };  l[374] = { x: 0.600, y: 0.468 };
  l[473] = { x: 0.600 + irisDX, y: 0.450 + irisDY };
  l[1]   = { x: 0.500, y: 0.550 };
  return l;
}
const rotate = (l, deg) => {
  const th = deg * Math.PI / 180, c = Math.cos(th), s = Math.sin(th), cx = 0.5, cy = 0.5;
  return l.map(p => ({ x: cx + (p.x - cx) * c - (p.y - cy) * s,
                       y: cy + (p.x - cx) * s + (p.y - cy) * c }));
};
// The formulation this replaced, for contrast.
const imageAxes = (l) => {
  const f = (I, O, IR) => {
    const w = Math.hypot(l[I].x - l[O].x, l[I].y - l[O].y) + 1e-6;
    return [(l[IR].x - (l[I].x + l[O].x) / 2) / w, (l[IR].y - (l[I].y + l[O].y) / 2) / w];
  };
  return [...f(133, 33, 468), ...f(362, 263, 473)];
};

const base = t._features(face());
const baseCanthus = base.slice(8);
const baseImage = imageAxes(face());
let failed = [];
console.log(`\n  feature vector: ${base.length} dims, canthus block = last 4\n`);
console.log(`  ${"head roll".padStart(10)}${"eye-frame drift".padStart(20)}${"image-axes drift".padStart(20)}`);
for (const deg of [0, 2, 5, 10, 15, 25, 40]) {
  const l = rotate(face(), deg);
  const nowC = t._features(l).slice(8);
  const nowI = imageAxes(l);
  const dC = Math.max(...nowC.map((v, i) => Math.abs(v - baseCanthus[i])));
  const dI = Math.max(...nowI.map((v, i) => Math.abs(v - baseImage[i])));
  console.log(`  ${(deg + " deg").padStart(10)}${dC.toExponential(2).padStart(20)}${dI.toFixed(5).padStart(20)}`);
  if (dC > 1e-9) failed.push(`${deg} deg drifted ${dC.toExponential(2)}`);
}
// and the features must still respond to the thing they are supposed to measure
const moved = t._features(face(0.020, 0.004)).slice(8);
const responds = Math.max(...moved.map((v, i) => Math.abs(v - baseCanthus[i]))) > 1e-3;
console.log(`\n  responds to actual iris movement: ${responds ? "yes" : "NO — feature is dead"}`);
if (!responds) failed.push("insensitive to iris movement");
console.log(failed.length ? `\n  FAILED: ${failed.join("; ")}` : "\n  PASS — invariant to roll, still sensitive to gaze");
process.exit(failed.length ? 1 : 0);
