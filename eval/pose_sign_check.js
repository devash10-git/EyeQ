/* Head-angle signs must match how a person would describe the movement.
 *
 * The capture page prompted "turn right" and required a left turn, and "chin
 * up" required chin down, because MediaPipe's raw extraction is opposite to the
 * user-centred convention on both axes. This pins the fix: synthetic rotation
 * matrices for movements with unambiguous names, checked against the sign the
 * prompt implies.
 *
 *   node eval/pose_sign_check.js
 */
const fs = require("fs"), path = require("path"), vm = require("vm");

/** Column-major 4x4, the layout MediaPipe hands back. */
function mat(R) {
  const m = new Array(16).fill(0); m[15] = 1;
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) m[c * 4 + r] = R[r][c];
  return m;
}
const rad = (d) => d * Math.PI / 180;
const Rx = (f) => [[1,0,0],[0,Math.cos(f),-Math.sin(f)],[0,Math.sin(f),Math.cos(f)]];
const Ry = (t) => [[Math.cos(t),0,Math.sin(t)],[0,1,0],[-Math.sin(t),0,Math.cos(t)]];
const Rz = (p) => [[Math.cos(p),-Math.sin(p),0],[Math.sin(p),Math.cos(p),0],[0,0,1]];

// The extraction under test, lifted from eval/capture.html so the test tracks
// the shipped code rather than a copy of it.
const html = fs.readFileSync(path.join(__dirname, "capture.html"), "utf8");
const src = html.match(/function poseFromMatrix\(m\)[\s\S]*?\n}/)[0];
const poseFromMatrix = new Function(src + "; return poseFromMatrix;")();

/* A camera sees you as another person does. Turning to YOUR right rotates your
   nose toward -X, which is a negative rotation about +Y; raising your chin
   rotates +Z toward +Y, negative about +X; tilting your right ear down moves
   the crown toward -X, which is positive about +Z. */
const CASES = [
  ["turn to my RIGHT",   mat(Ry(rad(-15))), "yaw",   +1],
  ["turn to my LEFT",    mat(Ry(rad(+15))), "yaw",   -1],
  ["chin UP",            mat(Rx(rad(-12))), "pitch", +1],
  ["chin DOWN",          mat(Rx(rad(+12))), "pitch", -1],
  ["tilt RIGHT ear down", mat(Rz(rad(+15))), "roll",  +1],
  ["tilt LEFT ear down",  mat(Rz(rad(-15))), "roll",  -1],
];

let failed = [];
console.log("\n  movement the user is asked to make -> sign the prompt implies\n");
console.log(`  ${"movement".padEnd(22)}${"axis".padEnd(8)}${"measured".padStart(11)}${"expected".padStart(11)}`);
for (const [name, m, axis, want] of CASES) {
  const p = poseFromMatrix(m);
  const deg = p[axis] * 180 / Math.PI;
  const ok = Math.sign(deg) === want && Math.abs(deg) > 5;
  console.log(`  ${name.padEnd(22)}${axis.padEnd(8)}${deg.toFixed(1).padStart(10)}°` +
              `${(want > 0 ? "positive" : "negative").padStart(11)}   ${ok ? "" : "  <-- WRONG"}`);
  if (!ok) failed.push(`${name}: got ${deg.toFixed(1)}°`);
}
// the two axes we did not touch must not have moved
const pure = poseFromMatrix(mat(Ry(rad(-15))));
const bleed = Math.max(Math.abs(pure.pitch), Math.abs(pure.roll)) * 180 / Math.PI;
console.log(`\n  a pure yaw leaks ${bleed.toFixed(3)}° into pitch/roll`);
if (bleed > 0.01) failed.push(`axis bleed ${bleed.toFixed(3)}°`);

console.log(failed.length ? `\n  FAILED: ${failed.join("; ")}` : "\n  PASS — every prompt direction matches the sign it implies");
process.exit(failed.length ? 1 : 0);
