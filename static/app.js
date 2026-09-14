/* Eye-gaze keyboard: webcam iris tracking (IrisTracker) or mouse drives a
 * dwell-to-select cursor over an individual-letter QWERTY keyboard. Dwelling a
 * letter types it directly; SPACE / ⌫ handle word boundary and delete. A
 * suggestion strip autocompletes the current word from the vocabulary. */

const DWELL_MS = 693;        // fixation time to commit a key
const REARM_MS = 450;        // pause before the same key can fire again (double letters)
const SMOOTHING = 0.35;      // legacy EMA factor (mouse mode / fallback only)

/* Adaptive pointer filtering (One-Euro, Casiez/Roussel/Vogel CHI 2012).
 *
 * A fixed EMA has one setting for two opposite jobs. At 0.35 it is far too
 * loose to hide sensor noise while the eye rests on a key — which is the
 * visible jitter — and tightening it would add lag every time the gaze
 * crosses the keyboard. One-Euro resolves that by adapting its cutoff to
 * speed: near-still input is filtered hard, fast input passes almost
 * untouched, so the pointer sits still without feeling sluggish.
 * Constants chosen by sweeping both objectives against the deployed EMA at
 * 45 px sensor noise (tools/tune_euro.js). The first attempt used beta = 0.010
 * and made tremor *worse* — at this noise level the frame-to-frame speed is
 * ~1300 px/s from noise alone, which opened the cutoff to ~13 Hz and stopped
 * the filter from filtering. Measured at the values below: resting tremor
 * 20.9 -> 12.5 px RMS (40% less), settling after a saccade 200 -> 175 ms. */
const EURO_MIN_CUTOFF = 0.30;   // Hz — the resting cutoff; lower = steadier
const EURO_BETA = 0.0006;       // how quickly the cutoff opens with speed
const EURO_DCUTOFF = 1.0;       // cutoff for the speed estimate itself

/* Dwell stickiness. Jitter across a key boundary used to reset dwellStart on
 * every flicker, so a dwell near an edge either never completed or completed
 * on the neighbour. The target is now held until the pointer is genuinely
 * outside it — past a margin, for several consecutive frames. */
const DWELL_EXIT_MARGIN = 20;   // px beyond the key edge before it counts as leaving
const DWELL_EXIT_FRAMES = 3;    // consecutive outside frames before the target switches
const CALIB_CLICKS = 5;      // clicks per calibration corner

// ---- uncertainty-propagating commit ----
const SIGMA_FLOOR_PX = 14;   // never trust gaze below this std-dev
const MOUSE_SIGMA_PX = 6;    // pointer is near-exact: posterior ≈ hit test
/* Tuned by tune.py against an end-to-end typing simulation over gaze noise.
 * THETA trades accuracy against how often the system asks instead of guessing;
 * measured character accuracy / prompt rate at sigma=40px:
 *   0.00 -> 85.9% /  0%      0.80 -> 98.8% / 22%
 *   0.70 -> 96.1% / 17%      0.90 -> 99.6% / 30%      0.99 -> 100% / 60%
 * LM_BETA=0.70 measured best (85.9% vs 85.2% at 0.55); above ~0.85 the
 * language prior starts overriding correct gaze evidence and accuracy falls. */
/* THETA is reached far less often than the tuning above suggests, and the
 * reason is geometry rather than the constant. Measured on the three recorded
 * sessions, the gaze error is 1.0-1.7x the width of a QWERTY key (4.7 deg
 * against 4.2 deg, 3.4 against 3.4, 5.1 against 3.1), so a single key is at or
 * below the tracker's resolution. Only 10.4%, 2.9% and 0.3% of fixations clear
 * 0.80, and dwelling on a key therefore shows choice chips rather than typing.
 *
 * Lowering the bar does not fix it -- it was tried and measured. Targeting a
 * 25% prompt rate raises typing to 43-63% but drops commit accuracy to 20-42%
 * and raises wrong commits from 3.1 to 37 per 100; sweeping the target from
 * 0.25 to 0.90 leaves accuracy flat at 20-47% throughout, while the commits
 * that do clear 0.80 are 70-100% correct. The threshold is refusing for a good
 * reason, and the chips are the designed path when it does.
 *
 * The fix that works is a larger target: on the same sessions a 3x3 cluster
 * layout puts the error at 0.30-0.50x the target width. See clusters.py. */
const THETA = 0.80;          // commit only above this fused confidence
const LM_BETA = 0.70;        // language-prior exponent (gaze stays in charge)
const LM_FLOOR = 1e-4;       // minimum prior: the LM may advise, never veto
const GATE_REFUSAL_LIMIT = 5; // consecutive refusals before the watchdog disables the gate
const QGAZE_BETA = 0.5;      // quantum gaze-stack exponent (advisory, not a veto)
const QGAZE_FLOOR = 1e-3;    // a letter outside its top-5 is unlikely, not banned

const keyboard = document.getElementById("keyboard");
const letterKeysToggle = document.getElementById("letter-keys");
if (letterKeysToggle) {
  letterKeysToggle.checked = (localStorage.getItem("kb-layout") || "clusters") === "letters";
  letterKeysToggle.addEventListener("change", () => {
    layoutMode = letterKeysToggle.checked ? "letters" : "clusters";
    applyLayout();
  });
}
const keys = [...document.querySelectorAll(".key")];
const kbClusters = document.getElementById("kb-clusters");
const kbLetters = document.getElementById("kb-letters");

/* ---------------- cluster layout ----------------
 * Each key is a whole QWERTY region rather than one letter. The reason is
 * measured, not aesthetic: gaze error on the recorded sessions is 1.0-1.7x the
 * width of a single letter key, so the fused posterior almost never clears
 * THETA and dwelling produces choice chips instead of text. A cluster is ~3x
 * wider, putting the error at 0.30-0.50x the target, which is what makes a
 * one-dwell commit possible at all.
 *
 * The cost is ambiguity: "hello" and "fellow" both gaze as 51663. The word is
 * therefore resolved at the space, by decoder.py, whose consistency oracle
 * guarantees whatever comes back re-encodes to exactly the clusters gazed. */
let layoutMode = localStorage.getItem("kb-layout") || "clusters";
let clusterCode = "";        // digits of the word currently being gazed
let codeWords = [];          // completed word codes, for sentence-level decoding
let clusterCands = [];       // live candidates for clusterCode
let clusterToken = 0;        // guards out-of-order candidate fetches
let lastDecode = null;       // last /api/decode response (diagnostics/tests)

function activeKeys() {
  const pane = layoutMode === "clusters" ? kbClusters : kbLetters;
  return keys.filter(k => !pane || pane.contains(k) || k.dataset.action);
}

function applyLayout() {
  const clusters = layoutMode === "clusters";
  if (kbClusters) kbClusters.hidden = !clusters;
  if (kbLetters) kbLetters.hidden = clusters;
  localStorage.setItem("kb-layout", layoutMode);
  clusterCode = ""; codeWords = []; clusterCands = [];
  renderLabel();
}
const cursor = document.getElementById("cursor");
const ring = document.getElementById("dwell-ring");
const labelEl = document.getElementById("label");
const chipsEl = document.getElementById("suggestions");
const statusEl = document.getElementById("status");
const gazeX = document.getElementById("gaze-x");
const gazeY = document.getElementById("gaze-y");
const calibDot = document.getElementById("calib-dot");
const mouseToggle = document.getElementById("mouse-mode");
const banner = document.getElementById("banner");
const faceHint = document.getElementById("face-hint");
const cameraBox = document.getElementById("camera-box");
const rawDot = document.getElementById("raw-dot");

let text = "";               // typed letters incl. spaces, e.g. "hello wor"
let chipEls = [];            // live suggestion chips (dwell/click targets)
let lastChipSig = "";        // avoids re-rendering chips mid-dwell when unchanged
let suggestToken = 0;        // drops out-of-order autocomplete responses
let chipsMode = "words";     // "words" | "choice" (uncertain letter) | "fix" (correction)
let letterPrior = null;      // { prefix, prior } — P(next letter | prefix), "$" = end-of-word
let movementState = null;    // latest /api/movement verdict {available,name,usable}
let gateRefusals = 0;        // consecutive keystrokes refused by the movement gate
let gateDisabled = false;    // watchdog: the gate blocked everything, so it is off
let movementBusy = false;    // one in-flight movement request at a time
let wordEvidence = [];       // per letter of the current word: [[letter,p],...] top-2
let qaoaLast = null;         // last /api/qaoa_word response (diagnostics/tests)
let qgazeState = null;       // latest /api/gaze_letter posterior (quantum stack)
let qgazeBusy = false;       // one in-flight gaze-letter request at a time
let qgazeActive = false;     // server says the stack may influence typing
let priorToken = 0;
let sigmaOverride = null;    // test hook: force a gaze std-dev in px
let lastFused = null;        // last fused posterior (diagnostics/tests)
let point = null;            // smoothed cursor position
let dwellKey = null;         // key currently dwelled on
let dwellStart = 0;
let rearmUntil = 0;
let mouseMode = false;
let gazeReady = false;
let autoFallback = false;    // mouse mode was engaged by us, not the user
let cameraState = "starting";
let faceState = "none";
let lastError = "";
let predictionCount = 0;
let tracker = null;
const calibratedCorners = new Set();

function setStatus(msg) { statusEl.textContent = msg; }

function showBanner(msg, info) {
  banner.textContent = msg;
  banner.className = info ? "info" : "";
  banner.style.display = "block";
}
function hideBanner() { banner.style.display = "none"; }

function esc(s) {
  return s.replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function renderLabel() {
  refreshLetterPrior(); // keep P(next letter | prefix) hot for the next commit
  if (layoutMode === "clusters") {
    const done = text.trim();
    if (!done && !clusterCode) {
      labelEl.innerHTML = `<span class="placeholder">Type something\u2026</span>`;
      return;
    }
    const best = clusterCands.find(c => c.exact) || clusterCands[0];
    labelEl.innerHTML =
      (done ? `<span class="tok-word">${esc(done)}</span>` : "") +
      (clusterCode ? `<span class="pending">${esc(clusterCode)}</span>` : "") +
      (clusterCode && best ? `<span class="pending-word">${esc(best.word)}</span>` : "");
    return;
  }
  if (!text) {
    labelEl.innerHTML = `<span class="placeholder">Type something\u2026</span>`;
    renderChips([]);
    return;
  }
  // show the exact typed letters; the word being typed carries the caret
  const words = text.split(" ");
  const typingLast = !text.endsWith(" ");
  labelEl.innerHTML = words.map((w, i) => {
    if (!w) return "";
    const current = typingLast && i === words.length - 1;
    return `<span class="tok-word${current ? " current" : ""}">${esc(w)}</span>`;
  }).join("");
  refreshSuggest();
}

// ---------------- word suggestions (vocabulary autocomplete) ----------------
// Letters are exact, so this is a plain prefix lookup: complete the word being
// typed from the 16k frequency vocabulary. Dwelling a suggestion finishes it.

function currentWord() {
  const parts = text.split(" ");
  return parts[parts.length - 1];
}

async function refreshSuggest() {
  const prefix = text.endsWith(" ") ? "" : currentWord();
  const token = ++suggestToken;
  if (!prefix) { renderChips([]); return; }
  try {
    const res = await fetch("/api/suggest?prefix=" + encodeURIComponent(prefix) + "&n=6");
    const data = await res.json();
    if (token !== suggestToken) return; // a newer keystroke superseded this
    renderChips(data.words || []);
  } catch (_) {
    if (token === suggestToken) renderChips([]);
  }
}

function renderChips(words) {
  if (chipsMode !== "words") return; // a choice/fix prompt owns the strip
  const sig = words.join("|");
  if (sig === lastChipSig) return; // don't rebuild chips mid-dwell if unchanged
  lastChipSig = sig;
  chipsEl.innerHTML = "";
  chipEls = words.map(w => {
    const chip = document.createElement("div");
    chip.className = "chip";
    chip.dataset.word = w;
    chip.textContent = w;
    chip.addEventListener("click", () => commit(chip));
    chipsEl.appendChild(chip);
    return chip;
  });
}

/* Accept a suggestion: replace the partial word with the full word + a space,
 * so completing a word costs a single dwell. */
function chooseWord(word) {
  const parts = text.split(" ");
  parts[parts.length - 1] = word;
  text = parts.join(" ") + " ";
  wordEvidence = [];   // chip pick supersedes any per-letter gaze evidence
  renderLabel();
}

// ---------------- uncertainty-propagating commit ----------------
// The gaze estimate is a Gaussian, not a point. Integrate it over every key
// to get P(key | gaze), fuse with P(letter | prefix) from the vocabulary,
// and commit only when the fused posterior clears THETA — otherwise ask,
// with big dwellable letter chips. Committed text is never a guess.

function erf(z) { // Abramowitz–Stegun 7.1.26, |error| < 1.5e-7
  const s = z < 0 ? -1 : 1; z = Math.abs(z);
  const t = 1 / (1 + 0.3275911 * z);
  const poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  return s * (1 - poly * Math.exp(-z * z));
}

function normCdf(x, mu, sigma) {
  return 0.5 * (1 + erf((x - mu) / (sigma * Math.SQRT2)));
}

async function refreshLetterPrior() {
  const prefix = text.endsWith(" ") || !text ? "" : currentWord();
  const token = ++priorToken;
  try {
    const res = await fetch("/api/letter_prior?prefix=" + encodeURIComponent(prefix));
    const data = await res.json();
    if (token !== priorToken) return;
    letterPrior = { prefix, prior: data.prior || {} };
  } catch (_) { /* keep the previous prior */ }
}

function gazeSigmaPx() {
  if (sigmaOverride != null) return sigmaOverride;
  if (!mouseMode && tracker?.lastSigma) {
    return {
      sx: Math.max(SIGMA_FLOOR_PX, tracker.lastSigma.sx * window.innerWidth),
      sy: Math.max(SIGMA_FLOOR_PX, tracker.lastSigma.sy * window.innerHeight),
    };
  }
  return { sx: MOUSE_SIGMA_PX, sy: MOUSE_SIGMA_PX };
}

/* P(key | gaze): integral of the gaze Gaussian over each key rectangle
 * (exact for an axis-aligned Gaussian: product of two 1-D CDF differences). */
function keyPosterior(mu, sig) {
  const out = [];
  let total = 0;
  for (const k of activeKeys()) {
    const r = k.getBoundingClientRect();
    const px = (normCdf(r.right, mu.x, sig.sx) - normCdf(r.left, mu.x, sig.sx)) *
               (normCdf(r.bottom, mu.y, sig.sy) - normCdf(r.top, mu.y, sig.sy));
    if (px > 1e-6) {
      out.push({ el: k, letter: k.dataset.letter || null, action: k.dataset.action || null, pGaze: px });
      total += px;
    }
  }
  if (total <= 1e-9) return [];
  out.forEach(c => c.pGaze /= total);
  return out.sort((a, b) => b.pGaze - a.pGaze);
}

function smartCommit(domKey, at) {
  const cands = keyPosterior(at || point, gazeSigmaPx());
  if (!cands.length) { commit(domKey); return; } // off-keyboard: legacy path

  // backspace is an editing action — decided by gaze alone
  if (cands[0].action === "backspace") { commit(cands[0].el); return; }

  /* Cluster layout: the decision is which *region* was gazed, so the language
   * evidence has to be lifted to regions too. P(cluster) is the sum of
   * P(letter) over the letters that cluster contains, which is exact rather
   * than a re-tuned approximation. */
  if (layoutMode === "clusters" && cands.some(c => c.el.dataset.cluster)) {
    const prefix = clusterCode;
    const cp = clusterPrior();
    const scored = cands
      .filter(c => c.el.dataset.cluster || c.action === "space")
      .map(c => {
        const d = c.el.dataset.cluster;
        const lm = d ? Math.max(cp[d] ?? 0, LM_FLOOR) : Math.max(cp["$"] ?? 0, LM_FLOOR);
        return { ...c, digit: d, score: c.pGaze * Math.pow(lm, LM_BETA) };
      });
    if (!scored.length) { commit(domKey); return; }
    const zc = scored.reduce((a, c) => a + c.score, 0) || 1;
    scored.forEach(c => c.p = c.score / zc);
    scored.sort((a, b) => b.p - a.p);
    lastFused = scored.map(c => ({ key: c.digit || "space", p: +c.p.toFixed(4) }));
    const best = scored[0];
    if (best.p >= THETA) {
      best.el.classList.add("flash");
      setTimeout(() => best.el.classList.remove("flash"), 200);
      if (best.digit) applyCluster(best.digit); else applySpace();
    } else {
      renderChoiceChips(scored.slice(0, 3));
    }
    return;
  }

  // Quantum-kernel movement gate: refuse to type from a window the classifier
  // says was a saccade or blink, where the iris position is not trustworthy.
  // Uses the most recent polled verdict (never blocks), and only intervenes
  // when the classifier is actually available and confident it was unusable.
  /* Watchdog. A gate whose job is to refuse keystrokes can, if its input ever
   * drifts outside the distribution it was trained on, refuse *every* keystroke
   * and leave the keyboard silently untypable — which is exactly what a raw-vs-
   * normalised eye-openness mismatch once did here. Typing is the product; the
   * gate is an accuracy refinement, so the gate yields. After this many
   * consecutive refusals it disables itself for the session and says so. */
  if (!gateDisabled && movementState && movementState.available
      && movementState.usable === false) {
    gateRefusals++;
    setStatus("ignored — " + movementState.name + " detected, hold your gaze steady");
    rearmUntil = performance.now() + REARM_MS;
    if (gateRefusals >= GATE_REFUSAL_LIMIT) {
      gateDisabled = true;
      console.warn("[gaze] movement gate refused " + GATE_REFUSAL_LIMIT +
                   " keystrokes in a row — disabling it so typing can continue");
      setStatus("eye-state filter disabled (it was blocking every key)");
    }
    return;
  }
  if (movementState && movementState.usable !== false) gateRefusals = 0;

  const prefix = text.endsWith(" ") || !text ? "" : currentWord();
  const prior = (letterPrior && letterPrior.prefix === prefix) ? letterPrior.prior : null;

  const fused = cands
    .filter(c => c.letter || c.action === "space")
    .map(c => {
      const raw = prior ? (c.letter ? (prior[c.letter] ?? 0) : (prior["$"] ?? 0)) : 1;
      // The prior may be exactly 0 for a letter the vocabulary never follows
      // this prefix with. Left unfloored that is an absolute veto: no amount of
      // gaze certainty could ever type it, so names, acronyms and any
      // out-of-vocabulary word become impossible. LM_FLOOR keeps the language
      // model advisory — decisive on ambiguous gaze, overridable by a confident
      // one. Calibrated so a centred fixation still wins against a zero prior.
      let score = c.pGaze * Math.pow(Math.max(raw, LM_FLOOR), LM_BETA);
      // Quantum gaze channel: the stack's letter posterior enters as a third
      // factor, exponent-damped like the language prior so it advises rather
      // than overrides. Only active when the server says the stack is fitted
      // for this user's geometry.
      if (qgazeActive && qgazeState && qgazeState.available && c.letter) {
        const hit = (qgazeState.posterior || []).find(e => e[0] === c.letter);
        score *= Math.pow(Math.max(hit ? hit[1] : 0, QGAZE_FLOOR), QGAZE_BETA);
      }
      return { ...c, score };
    });
  if (!fused.length) { commit(domKey); return; }
  const z = fused.reduce((s, c) => s + c.score, 0);
  fused.forEach(c => c.p = c.score / z);

  /* Quantum reliability tempering. The QNN classifies what the eye was doing and
   * the Bayesian network turns that into P(key correct | state); raising the
   * posterior to that power flattens it when the fixation looks suspect, so a
   * marginal frame falls below THETA and asks instead of committing. The binary
   * movement gate above already refuses the clearest saccades and blinks — this
   * catches the partial ones it lets through. Measured on corrupted fixations:
   * wrong commits per 100 fall 0.47 -> 0.33 at sigma=0.02 and 4.47 -> 3.47 at
   * sigma=0.03, for about 2 pp more prompts. */
  const rel = (!gateDisabled && movementState && movementState.available &&
               typeof movementState.reliability === "number")
    ? Math.max(0.05, Math.min(1, movementState.reliability)) : 1;
  if (rel < 0.999) {
    fused.forEach(c => c.p = Math.pow(c.p, rel));
    const zr = fused.reduce((s, c) => s + c.p, 0) || 1;
    fused.forEach(c => c.p /= zr);
  }
  fused.sort((a, b) => b.p - a.p);
  lastFused = fused.map(c => ({ key: c.letter || "space", p: +c.p.toFixed(4) }));

  const top = fused[0];
  if (top.p >= THETA) {
    top.el.classList.add("flash");
    setTimeout(() => top.el.classList.remove("flash"), 200);
    if (top.letter) {
      // record the letters-only posterior for this position so the whole word
      // can be jointly re-decoded (QAOA) at the word boundary
      const letters = fused.filter(c => c.letter).slice(0, 2);
      const z2 = letters.reduce((s, c) => s + c.p, 0) || 1;
      applyLetter(top.letter, letters.map(c => [c.letter, +(c.p / z2).toFixed(4)]));
    } else {
      applySpace();
    }
    return;
  }
  renderChoiceChips(fused.slice(0, 3)); // not sure -> ask, never guess
}

const CLUSTER_MEMBERS = {
  "1": "qwe", "2": "rtyu", "3": "iop", "4": "asd",
  "5": "fgh", "6": "jkl", "7": "zxcv", "9": "bnm",
};

/* P(next cluster | prefix), lifted exactly from P(next letter | prefix). */
function clusterPrior() {
  const prior = (letterPrior && letterPrior.prefixIsCode) ? letterPrior.prior : null;
  const out = {};
  for (const [d, members] of Object.entries(CLUSTER_MEMBERS)) {
    let sum = 0;
    for (const ch of members) sum += (prior ? (prior[ch] ?? 0) : 1 / 26);
    out[d] = sum;
  }
  out["$"] = prior ? (prior["$"] ?? 0) : 1 / 26;
  return out;
}

function applyCluster(digit) {
  clusterCode += digit;
  refreshClusterCands();
  renderLabel();
}

/* Resolve the gazed clusters into words. The whole sentence is sent, not just
 * the last word, so the decoder's bigram context applies; its consistency
 * oracle guarantees the result re-encodes to exactly these clusters. */
async function decodeClusters() {
  const seq = codeWords.join(" ");
  if (!seq) { text = ""; renderLabel(); return; }
  try {
    const res = await fetch("/api/decode", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clusters: seq }),
    });
    const data = await res.json();
    text = (data.text || "") + " ";
    lastDecode = data;
  } catch (_) {
    text = seq + " ";          // offline: show the codes rather than losing them
  }
  renderLabel();
}

async function refreshClusterCands() {
  if (!clusterCode) { clusterCands = []; renderChips([]); return; }
  const token = ++clusterToken;
  try {
    const res = await fetch("/api/cluster_suggest?n=6&code=" + encodeURIComponent(clusterCode));
    const data = await res.json();
    if (token !== clusterToken) return;
    clusterCands = data.words || [];
  } catch (_) { clusterCands = []; }
  renderChips(clusterCands.map(c => c.word));
  renderLabel();
}

function applyLetter(ch, evidence) {
  chipsMode = "words";
  text += ch;
  // default: a deliberate pick (chip choice / physical key) is near-certain
  wordEvidence.push(evidence && evidence.length ? evidence : [[ch, 0.99]]);
  renderLabel();
}

function applyBackspace() {
  chipsMode = "words";
  if (layoutMode === "clusters") {
    // peel the partial code first, then step back over completed words
    if (clusterCode) {
      clusterCode = clusterCode.slice(0, -1);
      refreshClusterCands();
      renderLabel();
    } else if (codeWords.length) {
      clusterCode = codeWords.pop().slice(0, -1);
      refreshClusterCands();
      decodeClusters();
    }
    return;
  }
  if (!text) return;
  if (!text.endsWith(" ")) wordEvidence.pop();
  else wordEvidence = [];   // stepping back across a boundary: evidence is gone
  text = text.slice(0, -1);
  renderLabel();
}

function applySpace() {
  chipsMode = "words";
  if (layoutMode === "clusters") {
    if (!clusterCode) return;          // nothing gazed yet: space is a no-op
    codeWords.push(clusterCode);
    clusterCode = "";
    clusterCands = [];
    decodeClusters();
    return;
  }
  if (text && !text.endsWith(" ")) {
    const finished = currentWord();
    const evidence = wordEvidence.slice();
    wordEvidence = [];
    text += " ";
    renderLabel();
    refineWord(finished, evidence);
  }
}

function replaceLastWord(word) {
  const trailing = text.endsWith(" ");
  const parts = text.trim().split(" ");
  parts[parts.length - 1] = word;
  text = parts.join(" ") + (trailing ? " " : "");
  chipsMode = "words";
  wordEvidence = [];   // the fix chip resolves the word; evidence is spent
  renderLabel();
}

/* Word-boundary refinement, fully asynchronous (typing never waits on it):
 *  1. QAOA joint re-decode of the letter posteriors accumulated while the
 *     word was typed (quantum/typing_qaoa.py) — catches gaze slips where a
 *     *sequence* of borderline letters is individually plausible but jointly
 *     unlikely ("tje" -> "the").
 *  2. Gaze-aware noisy-channel correction for out-of-vocabulary words.
 * Results merge into one fix-chip prompt; the user picks, nothing is silently
 * rewritten. Either service failing simply drops its contribution. */
async function refineWord(word, evidence) {
  if (!word) return;
  const ambiguous = evidence.length === word.length &&
    evidence.some(pos => pos.length > 1 && pos[0][1] < 0.9);
  const corrP = fetch("/api/correct?word=" + encodeURIComponent(word) + "&n=3")
    .then(r => r.json()).catch(() => null);
  const qaoaP = (ambiguous && word.length <= 12)
    ? fetch("/api/qaoa_word", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ positions: evidence }),
      }).then(r => r.json()).catch(() => null)
    : Promise.resolve(null);
  const [corr, qaoa] = await Promise.all([corrP, qaoaP]);
  qaoaLast = qaoa;
  if (currentWordBeforeSpace() !== word) return; // text moved on meanwhile
  const fixes = [];
  if (qaoa?.available && qaoa.changed && qaoa.word !== word) fixes.push(qaoa.word);
  if (corr && !corr.in_vocab) {
    for (const s of corr.suggestions || []) {
      if (s !== word && !fixes.includes(s)) fixes.push(s);
    }
  }
  if (fixes.length) renderFixChips(word, fixes.slice(0, 3));
}

function currentWordBeforeSpace() {
  const parts = text.trim().split(" ");
  return parts[parts.length - 1] || "";
}

function makeChip(label, cls, data, onPick) {
  const chip = document.createElement("div");
  chip.className = "chip " + cls;
  Object.assign(chip.dataset, data);
  chip.textContent = label;
  chip.addEventListener("click", () => commit(chip));
  chipsEl.appendChild(chip);
  return chip;
}

function renderChoiceChips(cands) {
  chipsMode = "choice";
  lastChipSig = "::choice::" + cands.map(c => c.letter || " ").join("|");
  chipsEl.innerHTML = "";
  chipEls = cands.map(c =>
    makeChip(c.letter ? c.letter.toUpperCase() : "SPACE", "choice", { choice: c.letter || " " }));
}

function renderFixChips(typed, words) {
  chipsMode = "fix";
  lastChipSig = "::fix::" + words.join("|");
  chipsEl.innerHTML = "";
  chipEls = [
    ...words.map(w => makeChip(w, "fix", { fix: w })),
    makeChip("keep “" + typed + "”", "fix keep", { fix: typed }),
  ];
}

function keyAt(x, y) {
  const el = document.elementFromPoint(x, y);
  return el ? el.closest(".key, .chip") : null;
}

/** One axis of a One-Euro filter. */
class OneEuroAxis {
  constructor() { this.x = null; this.dx = 0; this.t = null; }
  _alpha(cutoff, dt) { const tau = 1 / (2 * Math.PI * cutoff); return 1 / (1 + tau / dt); }
  filter(x, t) {
    if (this.x === null) { this.x = x; this.t = t; return x; }
    const dt = Math.max(1e-3, (t - this.t) / 1000);
    this.t = t;
    const raw = (x - this.x) / dt;
    this.dx += this._alpha(EURO_DCUTOFF, dt) * (raw - this.dx);
    const cutoff = EURO_MIN_CUTOFF + EURO_BETA * Math.abs(this.dx);
    this.x += this._alpha(cutoff, dt) * (x - this.x);
    return this.x;
  }
  reset() { this.x = null; this.dx = 0; this.t = null; }
}
const euroX = new OneEuroAxis(), euroY = new OneEuroAxis();

/* Points gathered since the dwell target last changed. The commit is decided
 * on their mean rather than on wherever the pointer happened to be when the
 * timer expired: one noisy final frame should not choose the letter, and a
 * 693 ms dwell supplies ~20 samples whose mean is far steadier than any one. */
let dwellPts = [];
function dwellCentroid() {
  if (!dwellPts.length) return point;
  let sx = 0, sy = 0;
  for (const p of dwellPts) { sx += p.x; sy += p.y; }
  return { x: sx / dwellPts.length, y: sy / dwellPts.length };
}

function moveCursor(x, y) {
  // The mouse is already exact, so filtering it only adds lag.
  if (mouseMode) {
    point = point
      ? { x: point.x + SMOOTHING * (x - point.x), y: point.y + SMOOTHING * (y - point.y) }
      : { x, y };
  } else {
    const now = performance.now();
    point = { x: euroX.filter(x, now), y: euroY.filter(y, now) };
  }
  cursor.style.display = "block";
  cursor.style.left = point.x + "px";
  cursor.style.top = point.y + "px";
  gazeX.textContent = "x: " + Math.round(point.x);
  gazeY.textContent = "y: " + Math.round(point.y);
  updateDwell();
}

/** Is the point still within `el`, allowing a margin outside its edges? */
function withinMargin(el, pt, margin) {
  if (!el) return false;
  const r = el.getBoundingClientRect();
  return pt.x >= r.left - margin && pt.x <= r.right + margin
      && pt.y >= r.top - margin && pt.y <= r.bottom + margin;
}

let outsideFrames = 0;

function updateDwell() {
  const now = performance.now();
  let key = keyAt(point.x, point.y);

  // Stickiness: a dwell in progress survives a pointer that strays briefly or
  // barely. Only a departure that is both beyond the margin and sustained for
  // several frames actually changes target.
  if (key !== dwellKey && dwellKey) {
    if (withinMargin(dwellKey, point, DWELL_EXIT_MARGIN)) {
      outsideFrames = 0;
      key = dwellKey;
    } else if (++outsideFrames < DWELL_EXIT_FRAMES) {
      key = dwellKey;
    }
  } else {
    outsideFrames = 0;
  }

  activeKeys().forEach(k => k.classList.toggle("active", k === key));
  chipEls.forEach(c => c.classList.toggle("active", c === key));

  if (key !== dwellKey) {
    dwellKey = key;
    dwellStart = now;
    dwellPts = [];
    outsideFrames = 0;
    ring.classList.remove("filling");
    void ring.offsetWidth; // restart the fill animation
    if (key && now >= rearmUntil) ring.classList.add("filling");
    return;
  }
  if (!key || now < rearmUntil) return;
  dwellPts.push({ x: point.x, y: point.y });
  if (dwellPts.length > 60) dwellPts.shift();

  if (!ring.classList.contains("filling")) {
    dwellStart = now;
    ring.classList.add("filling");
  }
  if (now - dwellStart >= DWELL_MS) {
    // chips commit directly; physical keys go through the fused posterior,
    // judged on the whole dwell window rather than its last frame
    if (key.classList.contains("chip")) commit(key);
    else smartCommit(key, dwellCentroid());
    rearmUntil = now + REARM_MS;
    dwellStart = now;
    dwellPts = [];
    ring.classList.remove("filling");
  }
}

function commit(key) {
  key.classList.add("flash");
  setTimeout(() => key.classList.remove("flash"), 200);

  if (key.dataset.fix) { replaceLastWord(key.dataset.fix); return; } // correction chip
  if (key.dataset.choice) {                                          // uncertain-letter chip
    if (key.dataset.choice === " ") applySpace();
    else applyLetter(key.dataset.choice);
    return;
  }
  if (key.dataset.word) { chooseWord(key.dataset.word); return; }    // suggestion chip
  const action = key.dataset.action;
  if (action === "backspace") { applyBackspace(); return; }
  if (action === "space") { applySpace(); return; }
  const letter = key.dataset.letter;
  if (letter) applyLetter(letter);
}

// ---------------- header controls ----------------

// Editing always dismisses a pending choice/correction prompt — otherwise the
// stale chips stay on screen and keep word suggestions suppressed.
document.getElementById("undo-btn").addEventListener("click", () => {
  lastChipSig = null;   // sentinel: an empty chip list ("") must still repaint
  applyBackspace();
});
document.getElementById("clear-btn").addEventListener("click", () => {
  chipsMode = "words";
  lastChipSig = null;   // sentinel: an empty chip list ("") must still repaint
  text = "";
  wordEvidence = [];
  renderLabel();
});
document.getElementById("reset-calib").addEventListener("click", () => {
  tracker?.resetCalibration();
  calibratedCorners.clear();
  document.querySelectorAll("button.calib").forEach(b => b.classList.remove("done"));
  showBanner("Calibration cleared — click each key (Q T/Y P · A G L · Z V M) and click its dot 5 times while looking at it", true);
  setTimeout(hideBanner, 4000);
});

// ---------------- calibration ----------------

// Corners plus interior spots (mid-left, center, mid-right). The extra points
// give the regression real data where the keys are, instead of extrapolating
// into the screen middle from the corners alone.
/* Calibration targets sit on key centres, not viewport corners.
 *
 * The old dots were inset 50 px from the viewport edges, so roughly a third of
 * the calibration span covered screen the user never looks at while typing —
 * the header and the suggestion strip. The fit spent its eight parameters
 * covering that dead area and was correspondingly looser over the keys.
 *
 * Measured on the two recorded sessions by refitting on only the in-keyboard
 * targets and scoring in-keyboard hold-out points:
 *
 *     session A   173.1 px -> 147.7 px   (4.96 deg -> 4.23 deg)   -14.7%
 *     session C   277.4 px -> 238.6 px   (6.36 deg -> 5.47 deg)   -14.0%
 *
 * A 3x3 grid on Q / T-Y / P, A / G / L, Z / V / M spans the key field almost
 * exactly, and every dot lands on a target the user will actually dwell on.
 * Positions are read from the live layout so they stay correct across window
 * resizes and display changes. */
const CALIB_KEYS = {
  TL: "q", TC: ["t", "y"], TR: "p",
  CL: "a", C:  "g",        CR: "l",
  BL: "z", BC: "v",        BR: "m",
};

function keyCentre(letter) {
  const el = keys.find(k => k.dataset.letter === letter);
  if (!el) return null;
  const r = el.getBoundingClientRect();
  return [r.left + r.width / 2, r.top + r.height / 2];
}

/** Centre of one key, or the midpoint between two when the row has no middle. */
function calibKeyPoint(spec) {
  if (Array.isArray(spec)) {
    const a = keyCentre(spec[0]), b = keyCentre(spec[1]);
    return a && b ? [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2] : (a || b);
  }
  return keyCentre(spec);
}

const CORNERS = Object.fromEntries(
  Object.entries(CALIB_KEYS).map(([id, spec]) => [id, () => calibKeyPoint(spec)]),
);
const TOTAL_CALIB = Object.keys(CORNERS).length;
let calibClicksLeft = 0;
let calibButton = null;
let calibPoint = null;

document.querySelectorAll("button.calib").forEach(btn => {
  btn.addEventListener("click", () => {
    if (!gazeReady) {
      showBanner("The gaze tracker isn't running (camera: " + cameraState +
                 (lastError ? ", error: “" + lastError + "”" : "") + "), so calibration " +
                 "clicks can't train it. Fix the camera first, then calibrate.");
      return;
    }
    if (faceState !== "ok") {
      showBanner("Your face isn't tracked right now (" + faceHint.textContent + "). " +
                 "Fix that first — the camera box border turns green when tracking is good.");
      return;
    }
    const pt = CORNERS[btn.dataset.corner]();
    if (!pt) { showBanner("That key isn't on screen — can't place the calibration dot there."); return; }
    const [x, y] = pt;
    calibPoint = { x, y };
    calibDot.style.left = x - 17 + "px";
    calibDot.style.top = y - 17 + "px";
    calibDot.style.display = "block";
    calibClicksLeft = CALIB_CLICKS;
    calibDot.dataset.count = calibClicksLeft;
    calibButton = btn;
    setStatus(`look at the dot and click it ${CALIB_CLICKS} times`);
  });
});

calibDot.addEventListener("click", () => {
  // each click records an iris-feature snapshot paired with the dot position
  const ok = tracker?.addCalibrationSample(calibPoint.x, calibPoint.y, calibButton?.dataset.corner);
  if (!ok) {
    showBanner("No face data right now — make sure the camera box border is green, then click again.");
    return;
  }
  calibClicksLeft--;
  calibDot.dataset.count = calibClicksLeft;
  if (calibClicksLeft <= 0) {
    calibDot.style.display = "none";
    if (calibButton) {
      calibButton.classList.add("done");
      calibratedCorners.add(calibButton.dataset.corner);
    }
    if (calibratedCorners.size >= TOTAL_CALIB) {
      showBanner("Calibration complete — gaze at a cluster to type. " +
                 "Clicking anywhere while looking at the pointer keeps refining it.", true);
      setTimeout(hideBanner, 5000);
      setStatus("eye tracking calibrated");
    } else {
      setStatus(`spot calibrated — ${TOTAL_CALIB - calibratedCorners.size} to go`);
    }
  }
});

// every ordinary click while gazing refines the model (click = ground truth)
document.addEventListener("click", (e) => {
  if (!gazeReady || mouseMode || !tracker?.fitted || faceState !== "ok") return;
  if (e.target.closest("button, label, input, #calib-dot")) return;
  tracker.addCalibrationSample(e.clientX, e.clientY);
});

// ---------------- input sources ----------------

mouseToggle.addEventListener("change", () => {
  mouseMode = mouseToggle.checked;
  autoFallback = false; // the user decided explicitly
  setStatus(mouseMode ? "mouse mode: cursor follows the pointer" : (gazeReady ? "eye tracking active" : "loading gaze tracker…"));
});

document.getElementById("show-dot").addEventListener("change", (e) => {
  rawDot.style.display = e.target.checked ? "block" : "none";
});

/* ---------------- camera exposure / contrast ----------------
 *
 * Off by default. What it fixes is tracking dropout: one recorded session lost
 * 67% of its frames because the landmarker could not hold the face in poor
 * light, which made that session unusable at any accuracy.
 *
 * On how much it can be worth, the honest answer turned out to depend on the
 * head. With a still head, sensor noise is 5% of the error budget and the dwell
 * centroid averages nearly all of it away. Past 10 degrees of roll it becomes
 * 58%, because per-frame landmark jitter grows 5.5x while the model's
 * amplification of it stays flat. An earlier version of this comment quoted the
 * still-head figure alone and concluded image tuning could never be worth more
 * than ~13%; that was measured before any session contained head movement.
 *
 * Two paths. If the camera exposes real controls we drive the sensor, which is
 * strictly better because it acts before quantisation. Most built-in Mac
 * cameras expose nothing, so the fallback filters the frame on its way to the
 * landmarker with a GPU-composited canvas draw. The panel says which is live
 * rather than leaving you guessing. */
const VIDEO_ADJUST_KEY = "gaze-video-adjust";
const vidToggle = document.getElementById("video-adjust");
const vidPanel = document.getElementById("cam-settings");
const camGear = document.getElementById("cam-gear");

/* The gear owns whether the panel is visible; the checkbox inside owns whether
   the adjustment is applied. Keeping those separate means opening the panel to
   look at it cannot change what the tracker sees. */
camGear.addEventListener("click", (e) => {
  e.stopPropagation();
  vidPanel.hidden = !vidPanel.hidden;
});
document.addEventListener("click", (e) => {
  if (!vidPanel.hidden && !vidPanel.contains(e.target) && e.target !== camGear) vidPanel.hidden = true;
});
const vidMode = document.getElementById("vid-mode");
const vidInputs = {
  exposure: document.getElementById("vid-exposure"),
  contrast: document.getElementById("vid-contrast"),
  brightness: document.getElementById("vid-brightness"),
};

function videoAdjustConfig() {
  return {
    enabled: vidToggle.checked,
    exposure: Number(vidInputs.exposure.value),
    contrast: Number(vidInputs.contrast.value),
    brightness: Number(vidInputs.brightness.value),
  };
}

function applyVideoAdjust() {
  const cfg = videoAdjustConfig();
  document.getElementById("vid-exposure-v").textContent = cfg.exposure > 0 ? "+" + cfg.exposure : cfg.exposure;
  document.getElementById("vid-contrast-v").textContent = cfg.contrast + "%";
  document.getElementById("vid-brightness-v").textContent = cfg.brightness + "%";

  // Mirror the adjustment onto the preview so you can see what you are setting.
  const cam = document.getElementById("cam");
  const gain = 1 + (cfg.exposure / 100) * 0.6;
  cam.style.filter = cfg.enabled
    ? `brightness(${(cfg.brightness / 100) * gain}) contrast(${cfg.contrast / 100})`
    : "";

  try { localStorage.setItem(VIDEO_ADJUST_KEY, JSON.stringify(cfg)); } catch (_) {}

  if (!tracker?.setVideoAdjust) { vidMode.textContent = ""; return; }
  const path = tracker.setVideoAdjust(cfg);
  const caps = Object.keys(tracker.cameraCaps || {});
  if (!cfg.enabled) {
    vidMode.className = ""; vidMode.textContent = "";
  } else if (path === "hardware") {
    vidMode.className = "";
    vidMode.textContent = "driving the camera directly (" + caps.join(", ") + ")";
  } else {
    vidMode.className = "warn";
    vidMode.textContent = caps.length
      ? "this camera exposes only: " + caps.join(", ") + " — filtering the frame in software instead"
      : "this camera exposes no manual controls — filtering the frame in software instead";
  }
}

vidToggle.addEventListener("change", applyVideoAdjust);
for (const el of Object.values(vidInputs)) el.addEventListener("input", applyVideoAdjust);
document.getElementById("vid-reset").addEventListener("click", () => {
  vidInputs.exposure.value = 0;
  vidInputs.contrast.value = 100;
  vidInputs.brightness.value = 100;
  applyVideoAdjust();
});

(function restoreVideoAdjust() {
  try {
    const saved = JSON.parse(localStorage.getItem(VIDEO_ADJUST_KEY) || "null");
    if (!saved) return;
    vidToggle.checked = !!saved.enabled;
    if (saved.exposure != null) vidInputs.exposure.value = saved.exposure;
    if (saved.contrast != null) vidInputs.contrast.value = saved.contrast;
    if (saved.brightness != null) vidInputs.brightness.value = saved.brightness;
  } catch (_) {}
  applyVideoAdjust();
})();

document.addEventListener("mousemove", e => {
  if (mouseMode) moveCursor(e.clientX, e.clientY);
});

// mousemove stops firing the moment the pointer rests on a key — which is
// exactly the dwell gesture — so a ticker keeps the dwell clock advancing
// between events. Gaze mode needs no ticker: frames stream while tracked.
setInterval(() => { if (mouseMode && point) updateDwell(); }, 50);

function fallbackToMouse(msg) {
  mouseToggle.checked = mouseMode = true;
  autoFallback = true;
  setStatus("mouse mode enabled");
  showBanner(msg);
}

async function startGaze() {
  gazeReady = false;
  lastError = "";
  if (typeof IrisTracker === "undefined") {
    cameraState = "no-tracker";
    fallbackToMouse("iris-tracker.js failed to load — mouse mode enabled.");
    return;
  }
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    cameraState = "insecure-context";
    fallbackToMouse("Camera requires a secure context — open the app at http://localhost:8321 " +
                    "on this machine (or via https), not a plain LAN IP.");
    return;
  }

  tracker = new IrisTracker({
    video: document.getElementById("cam"),
    overlay: document.getElementById("cam-overlay"),
    onReady: (delegate) => {
      applyVideoAdjust();          // push any restored setting to the camera
      cameraState = delegate === "CPU" ? "on (CPU — WebGL not needed)" : "on (GPU)";
      gazeReady = true;
      if (tracker.fitted) {
        for (const c of tracker.meta.corners || []) {
          calibratedCorners.add(c);
          document.getElementById("calib-" + c)?.classList.add("done");
        }
        showBanner("Calibration restored from your last session — the gaze cursor is live. " +
                   "Click Reset calibration to redo it.", true);
        setTimeout(hideBanner, 5000);
        setStatus("eye tracking active (restored calibration)");
      } else {
        setStatus("camera active — calibrate the 9 keys (Q T/Y P · A G L · Z V M)");
        showBanner("Face the camera until the box border turns green, then click each spot " +
                   "(Q T/Y P · A G L · Z V M) and click its dot 5 times while looking at it.", true);
      }
    },
    onFace: (state, hint) => {
      faceState = state;
      faceHint.textContent = hint;
      cameraBox.classList.toggle("ok", state === "ok");
    },
    onGaze: (x, y) => {
      predictionCount++;
      rawDot.style.left = x + "px";
      rawDot.style.top = y + "px";
      if (mouseMode && autoFallback) {
        autoFallback = false;
        mouseToggle.checked = mouseMode = false;
        showBanner("Gaze tracking recovered — mouse mode switched off", true);
        setTimeout(hideBanner, 3000);
      }
      if (!mouseMode) {
        if (predictionCount === 1) { hideBanner(); setStatus("eye tracking active"); }
        moveCursor(x, y);
      }
    },
  });
  tracker.loadSaved();

  try {
    await tracker.start();
  } catch (e) {
    lastError = (e && (e.message || e.name)) || String(e);
    console.error("IrisTracker startup failure:", e);
    tracker.stop(); // release the camera — the light must not stay on in fallback
    const msg = {
      NotAllowedError: "Camera blocked. Allow camera access for this site, and on macOS check " +
                       "System Settings → Privacy & Security → Camera. Then reload.",
      NotFoundError: "No camera detected on this device — mouse mode enabled.",
      NotReadableError: "The camera is in use by another app (Zoom, FaceTime…) — close it and reload.",
    }[e.name] || ("Gaze tracker failed to start: “" + lastError + "” — mouse mode enabled. " +
                  "(If this mentions the network, the MediaPipe CDN may be blocked.)");
    cameraState = e.name ? "blocked:" + e.name : "tracker-failed";
    fallbackToMouse(msg);
  }
}
window.addEventListener("load", startGaze);

// Live diagnostics: on-screen panel + a ping that lands in the server access
// log (GET with query params, so it is visible there even as a 404).
setInterval(() => {
  document.getElementById("diag-camera").textContent = "camera: " + cameraState +
    (lastError ? " (" + lastError.slice(0, 60) + ")" : "");
  document.getElementById("diag-pred").textContent =
    "face: " + faceState + " · gaze frames: " + predictionCount;
  document.getElementById("diag-calib").textContent = "calibration: " + calibratedCorners.size +
    "/" + TOTAL_CALIB + " (" + (tracker?.samples.length || 0) + " samples)" +
    (movementState && movementState.available ? " · eye: " + movementState.name : "");
}, 1000);

/* Quantum-kernel movement classifier (see quantum/movement_qsvm.py).
 * Polled on its own timer rather than called at commit: the request costs
 * ~15 ms, which is half a 33 ms frame, so it must never sit on the render or
 * commit path. 250 ms is well inside the 693 ms dwell, so the verdict consulted
 * by smartCommit is always fresh relative to a keystroke. A failure leaves
 * movementState null and typing falls back to the local threshold rule. */
/* Quantum gaze->letter stack (/api/gaze_letter).
 *
 * Polled on the same principle as the movement classifier: the request costs
 * tens of milliseconds, so it must never sit on the render or commit path.
 * 250 ms is well inside the 693 ms dwell, so whatever smartCommit reads is at
 * most one poll old. A failure leaves qgazeState null and the fused decision
 * proceeds exactly as before.
 *
 * Whether the answer may *influence* typing is the server's call
 * (info.gaze_letter_active, set by ENABLE_QGAZE=1). Off, this loop still runs
 * and records diagnostics, so the stack can be compared against the classical
 * decision on real gaze without risking a single keystroke. */
setInterval(() => {
  if (!tracker || mouseMode || qgazeBusy) return;
  const g = tracker.gazeFeatures?.();
  const m = tracker.movementFeatures?.();
  if (!g || !m) return;
  qgazeBusy = true;
  fetch("/api/gaze_letter", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...g, ...m }),
  })
    .then(r => r.json())
    .then(d => { qgazeState = d; })
    .catch(() => { qgazeState = null; })
    .finally(() => { qgazeBusy = false; });
}, 250);

fetch("/api/info").then(r => r.json())
  .then(d => { qgazeActive = !!d.gaze_letter_active; })
  .catch(() => {});

setInterval(() => {
  if (!tracker || mouseMode || movementBusy) return;
  const f = tracker.movementFeatures?.();
  if (!f) return;
  movementBusy = true;
  fetch("/api/movement", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(f),
  })
    .then(r => r.json())
    .then(d => { movementState = d; })
    .catch(() => { movementState = null; })
    .finally(() => { movementBusy = false; });
}, 250);
setInterval(() => {
  fetch(`/api/diag?camera=${encodeURIComponent(cameraState)}&face=${faceState}&pred=${predictionCount}` +
        `&calib=${calibratedCorners.size}&mouse=${mouseMode}` +
        `&err=${encodeURIComponent(lastError.slice(0, 120))}`).catch(() => {});
}, 5000);

// Programmatic hooks for automated testing
window.__gaze = {
  setPoint: (x, y) => moveCursor(x, y),
  state: () => ({ text, preview: labelEl.textContent,
                  layout: layoutMode, clusterCode, codeWords: [...codeWords],
                  clusterCands: clusterCands.map(c => c.word), lastDecode,
                  chips: chipEls.map(c => c.dataset.word || c.dataset.choice || c.dataset.fix),
                  chipsMode, lastFused, priorPrefix: letterPrior?.prefix ?? null,
                  mouseMode, gazeReady, cameraState, faceState, lastError, predictionCount,
                  calibrated: calibratedCorners.size, banner: banner.textContent }),
  commitLetter: (ch) => { const k = keys.find(k => k.dataset.letter === ch); if (k) commit(k); },
  commitCluster: (d) => { const k = keys.find(k => k.dataset.cluster === d); if (k) smartCommit(k); },
  setLayout: (m) => { layoutMode = m; applyLayout(); },
  commitAction: (a) => { const k = keys.find(k => k.dataset.action === a); if (k) commit(k); },
  smartCommitAt: (x, y) => { point = { x, y }; smartCommit(keyAt(x, y) || keys[0]); },
  setSigma: (px) => { sigmaOverride = px == null ? null : { sx: px, sy: px }; },
  chooseWord,
  backspace: () => { applyBackspace(); },
  getEvidence: () => wordEvidence.map(p => p.slice()),
  qaoaLast: () => qaoaLast,
  qgazeLast: () => qgazeState,
  setMovement: (state) => { movementState = state; },
  qgazeActive: () => qgazeActive,
  setQgaze: (state, active) => { qgazeState = state; qgazeActive = !!active; },
  // commit a letter with an explicit posterior (integration tests)
  applyLetterEv: (ch, ev) => applyLetter(ch, ev),
  // movement-gate hooks for automated testing
  getMovement: () => movementState,
  gateDisabled: () => gateDisabled,
  resetGate: () => { gateRefusals = 0; gateDisabled = false; },
  // route a key through the gated fusion path, from the centre of that key
  commitKey: (ch) => {
    const k = keys.find(k => k.dataset.letter === ch);
    if (!k) return;
    const r = k.getBoundingClientRect();
    point = { x: r.left + r.width / 2, y: r.top + r.height / 2 };
    rearmUntil = 0;
    smartCommit(k);
  },
  startGaze,
  getTracker: () => tracker,
};

applyLayout();   // sets the keyboard to the stored layout and renders the label
renderLabel(); // paint the initial placeholder
