"""Head-to-head: quantum conditional selection rule vs the deployed rule.

Two questions, measured separately because they are different claims.

**Does interference type more accurately?**  Both arms see *identical* gaze
noise — same draws, same key-rectangle integrals — so every difference is the
selection rule and nothing else.

Accuracy alone cannot answer this. The keyboard refuses to commit below
``THETA`` and asks instead, and a prompt always yields the right character, so
any arm can reach 100% accuracy by prompting constantly. The honest comparison
is the whole trade-off curve: sweep the threshold, and compare **commit
accuracy at matched prompt rate**. That also removes the threshold as a
confound, since neither arm gets to keep a threshold tuned in its own favour.
The phase scale is tuned on a training split and reported on a disjoint
held-out split; a gain that does not survive the split is not a gain.

**Does amplitude amplification shorten dwell?**  Dwell is evidence
accumulation: a noisy gaze sample per frame, committed once the fused posterior
clears the threshold. Frames-to-commit *is* the dwell length, and it is what
amplification would have to reduce.

Consecutive gaze frames are strongly correlated — the eye is still, the
landmark noise is not white — so treating each frame as an independent sample
would let any arm commit in one or two frames and leave nothing to measure.
``FRAME_ESS`` discounts each frame to a fraction of an independent sample,
which is the standard effective-sample-size correction and is applied
identically to both arms.

The gaze likelihood is the exact 2-D Gaussian integral over key rectangles, the
same computation the deployed keyboard runs in ``static/app.js::keyPosterior``,
so the baseline arm is the shipping rule rather than an approximation of it.

Run: ``python3 eval/selection_rule_bench.py``
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decoder import HybridDecoder                      # noqa: E402
from letters import LetterModel                        # noqa: E402
from quantum.conditional_selection import (            # noqa: E402
    END, coherence, loglinear, phases, select,
)
from tune import VOCAB, load_sentences                 # noqa: E402

# Deployed constants (static/app.js) — the incumbent is its real self here.
LM_BETA = 0.70
LM_FLOOR = 1e-4
THETA = 0.80

KEY_W, KEY_H = 100.0, 110.0
ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
SIGMAS = (30.0, 40.0, 50.0, 60.0, 75.0)
PHASE_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
THETA_GRID = tuple(np.round(np.arange(0.05, 1.00, 0.025), 3))
MATCH_RATES = (0.02, 0.05, 0.10, 0.20)
N_HYP = 32
FRAME_ESS = 0.25        # a frame carries a quarter of an independent sample


def key_rects() -> Dict[str, Tuple[float, float, float, float]]:
    """Key rectangles in pixels, matching the deployed QWERTY stagger."""
    out = {}
    for r, row in enumerate(ROWS):
        indent = (10 - len(row)) * KEY_W / 2.0
        for i, ch in enumerate(row):
            x = indent + i * KEY_W
            out[ch] = (x, r * KEY_H, x + KEY_W, (r + 1) * KEY_H)
    return out


RECTS = key_rects()
CENTRES = {k: ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2) for k, r in RECTS.items()}


def _ncdf(x: float, mu: float, s: float) -> float:
    return 0.5 * (1.0 + math.erf((x - mu) / (s * math.sqrt(2.0))))


def gaze_posterior(gx: float, gy: float, sigma: float) -> Dict[str, float]:
    """P(key | gaze): the Gaussian's exact integral over each key rectangle."""
    out: Dict[str, float] = {}
    tot = 0.0
    for ch, (x0, y0, x1, y1) in RECTS.items():
        p = ((_ncdf(x1, gx, sigma) - _ncdf(x0, gx, sigma)) *
             (_ncdf(y1, gy, sigma) - _ncdf(y0, gy, sigma)))
        if p > 1e-9:
            out[ch] = p
            tot += p
    return {k: v / tot for k, v in out.items()} if tot > 0 else {}


def fuse(pg: Dict[str, float], prior: Dict[str, float], typed: str,
         comps: Sequence[Tuple[str, float]], phase_scale: float,
         amplify_cap: int, stage: str = "pre") -> Dict[str, float]:
    """The selection rule, or the deployed rule when both knobs are off."""
    if phase_scale > 0.0 or amplify_cap > 0:
        return select(pg, prior, comps, prefix=typed, beta=LM_BETA,
                      floor=LM_FLOOR, phase_scale=phase_scale,
                      amplify_cap=amplify_cap, amplify_stage=stage)
    return loglinear(pg, prior, beta=LM_BETA, floor=LM_FLOOR)


# ==========================================================================
# Experiment 1 — the accuracy / prompt-rate trade-off
# ==========================================================================

def run_keystrokes(lm: LetterModel, pairs: Sequence[Tuple[str, str]],
                   sigma: float, phase_scale: float, amplify_cap: int,
                   stage: str = "pre", seed: int = 7) -> Dict[str, object]:
    """Type every sentence, recording ``(top_p, was_correct, sentence)`` per key.

    Typing proceeds under the deployed ``THETA`` so the prefix context evolves
    the way it really would, but the record keeps each keystroke's top posterior
    so any threshold can be scored afterwards from one pass.
    """
    rng = np.random.default_rng(seed)          # same seed => same noise per arm
    top_p: List[float] = []
    ok: List[bool] = []
    sent: List[int] = []

    for si, (_code, en) in enumerate(pairs):
        for word in en.split():
            typed = ""
            for ch in word:
                if ch not in CENTRES:
                    continue
                cx, cy = CENTRES[ch]
                pg = gaze_posterior(cx + rng.normal(0, sigma),
                                    cy + rng.normal(0, sigma), sigma)
                if not pg:
                    continue
                prior = lm.letter_prior(typed)
                comps = (lm.completions(typed, n=N_HYP)
                         if (phase_scale > 0.0 or amplify_cap > 0) else [])
                post = fuse(pg, prior, typed, comps, phase_scale, amplify_cap, stage)

                best = max(post, key=post.get)
                top_p.append(post[best])
                ok.append(best == ch)
                sent.append(si)
                # the live system would prompt below THETA and get the right
                # character, so the prefix advances correctly either way
                typed += ch if (post[best] < THETA or best == END) else best

    return {"top_p": np.array(top_p), "ok": np.array(ok, dtype=bool),
            "sent": np.array(sent), "n": len(top_p)}


def curve(rec: Dict[str, object]) -> List[Dict[str, float]]:
    """Commit accuracy and prompt rate across the threshold grid."""
    p, ok = rec["top_p"], rec["ok"]
    out = []
    for th in THETA_GRID:
        commit = p >= th
        nc = int(commit.sum())
        out.append({"theta": float(th),
                    "prompt_rate": float(1.0 - nc / max(1, len(p))),
                    "commit_acc": float(ok[commit].mean()) if nc else 1.0,
                    "char_acc": float((ok[commit].sum() + (len(p) - nc)) / max(1, len(p))),
                    "n_commit": nc})
    return out


def acc_at_rate(cv: List[Dict[str, float]], rate: float) -> Optional[float]:
    """Commit accuracy at a given prompt rate, linearly interpolated."""
    pts = sorted(cv, key=lambda r: r["prompt_rate"])
    lo = [r for r in pts if r["prompt_rate"] <= rate]
    hi = [r for r in pts if r["prompt_rate"] >= rate]
    if not lo or not hi:
        return None
    a, b = lo[-1], hi[0]
    if abs(b["prompt_rate"] - a["prompt_rate"]) < 1e-12:
        return a["commit_acc"]
    t = (rate - a["prompt_rate"]) / (b["prompt_rate"] - a["prompt_rate"])
    return a["commit_acc"] + t * (b["commit_acc"] - a["commit_acc"])


def paired_by_sentence(a: Dict[str, object], b: Dict[str, object],
                       theta: float) -> Tuple[float, float, float]:
    """Per-sentence commit accuracy for each arm, and the paired Wilcoxon p."""
    xs, ys = [], []
    for s in np.unique(a["sent"]):
        ma, mb = (a["sent"] == s), (b["sent"] == s)
        ca, cb = a["top_p"][ma] >= theta, b["top_p"][mb] >= theta
        if ca.sum() and cb.sum():
            xs.append(float(a["ok"][ma][ca].mean()))
            ys.append(float(b["ok"][mb][cb].mean()))
    if len(xs) < 5:
        return float("nan"), float("nan"), 1.0
    x, y = np.array(xs), np.array(ys)
    p = 1.0 if np.allclose(x, y) else float(
        wilcoxon(x, y, zero_method="zsplit").pvalue)
    return float(x.mean()), float(y.mean()), p


# ==========================================================================
# Experiment 2 — frames to commit (the dwell claim)
# ==========================================================================

def run_dwell(lm: LetterModel, pairs: Sequence[Tuple[str, str]], sigma: float,
              phase_scale: float, amplify_cap: int, stage: str = "pre",
              max_frames: int = 60, seed: int = 23) -> Dict[str, object]:
    """Frames of accumulated evidence before the posterior clears THETA.

    Per-frame gaze posteriors are multiplied in, each discounted by
    ``FRAME_ESS`` to reflect that consecutive frames are correlated rather than
    independent. A keystroke that never clears within ``max_frames`` is counted
    as censored and reported rather than folded into the mean.
    """
    rng = np.random.default_rng(seed)
    frames: List[int] = []
    keys: List[Tuple[int, int]] = []
    censored = 0
    correct = 0
    ki = 0

    for _code, en in pairs:
        for word in en.split():
            typed = ""
            for ch in word:
                if ch not in CENTRES:
                    continue
                cx, cy = CENTRES[ch]
                prior = lm.letter_prior(typed)
                comps = (lm.completions(typed, n=N_HYP)
                         if (phase_scale > 0.0 or amplify_cap > 0) else [])
                acc = {k: 1.0 for k in CENTRES}
                hit = None
                for f in range(1, max_frames + 1):
                    pg = gaze_posterior(cx + rng.normal(0, sigma),
                                        cy + rng.normal(0, sigma), sigma)
                    for k in acc:
                        acc[k] *= max(pg.get(k, 1e-12), 1e-12) ** FRAME_ESS
                    z = sum(acc.values()) or 1.0
                    ev = {k: v / z for k, v in acc.items()}
                    post = fuse(ev, prior, typed, comps, phase_scale, amplify_cap, stage)
                    best = max(post, key=post.get)
                    if post[best] >= THETA:
                        hit = (f, best)
                        break
                if hit is None:
                    censored += 1
                else:
                    f, best = hit
                    frames.append(f)
                    keys.append((ki, f))
                    correct += (best == ch)
                typed += ch
                ki += 1

    arr = np.array(frames, dtype=float) if frames else np.array([0.0])
    return {"median_frames": float(np.median(arr)),
            "mean_frames": float(arr.mean()),
            "committed": len(frames), "censored": censored,
            "acc_at_commit": correct / max(1, len(frames)),
            "by_key": dict(keys)}



# ==========================================================================
# Experiment 3 — why the interference term behaves as it does
# ==========================================================================

def run_discrimination(lm: LetterModel, pairs: Sequence[Tuple[str, str]],
                       sigma: float = 40.0, seed: int = 7
                       ) -> List[Dict[str, float]]:
    """Coherence on the correct letter versus the wrong ones, by phase scale.

    For interference to help, the coherence penalty must fall on wrong letters
    more heavily than on right ones — the separation below must be positive.
    It is not, at any scale, and it worsens as the scale grows.

    The reason is structural rather than a matter of tuning. The correct letter
    after an English prefix is usually the one with the *richest* vocabulary
    support: many real continuations, hence many hypotheses, hence the widest
    spread of phases and the lowest coherence. A wrong letter is typically
    backed by one obscure completion and sits at coherence 1. Any phase derived
    from hypothesis likelihood therefore penalises exactly the evidence it
    should reward, and no re-tuning of the phase function changes that sign.
    """
    out = []
    for scale in (0.2, 0.4, 0.8, 1.0, 2.0):
        right: List[float] = []
        wrong: List[float] = []
        rng = np.random.default_rng(seed)
        for _code, en in pairs:
            for word in en.split():
                typed = ""
                for ch in word:
                    if ch not in CENTRES:
                        continue
                    cx, cy = CENTRES[ch]
                    pg = gaze_posterior(cx + rng.normal(0, sigma),
                                        cy + rng.normal(0, sigma), sigma)
                    if not pg:
                        typed += ch
                        continue
                    comps = lm.completions(typed, n=N_HYP * 2)
                    n = len(typed)
                    phi = phases(comps, scale)
                    co = coherence([(w[n] if len(w) > n else END, c, p)
                                    for (w, c), p in zip(comps, phi)])
                    for k, v in co.items():
                        (right if k == ch else wrong).append(abs(v))
                    typed += ch
        a = float(np.mean(right)) if right else float("nan")
        b = float(np.mean(wrong)) if wrong else float("nan")
        out.append({"scale": scale, "coh_correct": a, "coh_wrong": b,
                    "separation": a - b})
    return out


# ==========================================================================
# Driver
# ==========================================================================

def main() -> None:
    t0 = time.time()
    print("=" * 82)
    print("QUANTUM CONDITIONAL SELECTION RULE vs DEPLOYED LOG-LINEAR FUSION")
    print("=" * 82)

    dec = HybridDecoder(vocab_path=VOCAB, model_dir=None, num_clusters=9)
    lm = LetterModel(dec.unigrams)
    train = load_sentences(60, seed=3)
    test = load_sentences(60, seed=4, skip=2060)
    print("vocabulary %d words | train %d sentences | held-out %d sentences"
          % (len(dec.unigrams), len(train), len(test)))
    print("incumbent: LM_BETA=%.2f LM_FLOOR=%.0e THETA=%.2f (static/app.js)"
          % (LM_BETA, LM_FLOOR, THETA))

    # ---- tune the phase scale on train only ----
    print("\n" + "-" * 82)
    print("1. PHASE SCALE SWEEP  (train split, sigma=40px, commit acc at 20% prompts)")
    print("-" * 82)
    print("  %-8s %14s %14s" % ("scale", "acc@20%prompt", "acc@THETA"))
    best_scale, best_acc = 0.0, -1.0
    for ps in PHASE_GRID:
        cv = curve(run_keystrokes(lm, train, 40.0, ps, 0))
        a20 = acc_at_rate(cv, 0.20)
        at = [r for r in cv if abs(r["theta"] - 0.80) < 1e-9]
        flag = ""
        if a20 is not None and a20 > best_acc:
            best_acc, best_scale = a20, ps
            flag = "  <-"
        print("  %-8.2f %14s %14.4f%s"
              % (ps, "n/a" if a20 is None else "%.4f" % a20,
                 at[0]["commit_acc"] if at else float("nan"), flag))
    print("  -> best on train: phase_scale=%.2f (%.4f)" % (best_scale, best_acc))

    # ---- held-out, paired, across the noise sweep ----
    print("\n" + "-" * 82)
    print("2. HELD-OUT  (phase_scale=%.2f)  commit accuracy at matched prompt rate"
          % best_scale)
    print("-" * 82)
    hdr = "  %-7s" % "sigma" + "".join("%16s" % ("@%d%% prompts" % (r * 100))
                                       for r in MATCH_RATES) + "%10s" % "paired p"
    print(hdr)
    rows = []
    for sg in SIGMAS:
        b = run_keystrokes(lm, test, sg, 0.0, 0)
        q = run_keystrokes(lm, test, sg, best_scale, 0)
        cb, cq = curve(b), curve(q)
        cells, deltas = [], []
        for r in MATCH_RATES:
            x, y = acc_at_rate(cb, r), acc_at_rate(cq, r)
            if x is None or y is None:
                cells.append("%16s" % "n/a")
            else:
                cells.append("%16s" % ("%.4f %+.4f" % (x, y - x)))
                deltas.append(y - x)
        mb, mq, p = paired_by_sentence(b, q, THETA)
        rows.append({"sigma": sg, "deltas": deltas,
                     "median_delta": float(np.median(deltas)) if deltas else None,
                     "base_at_theta": mb, "rule_at_theta": mq, "p": p,
                     "n": b["n"]})
        print("  %-7.0f" % sg + "".join(cells) + "%10.4f" % p)
    print("  (cell = baseline accuracy, then the rule's delta)")

    # ---- amplification and dwell ----
    print("\n" + "-" * 82)
    print("3. AMPLITUDE AMPLIFICATION — frames to commit (held-out, sigma=40px,"
          " frame ESS=%.2f)" % FRAME_ESS)
    print("-" * 82)
    dwell = {}
    for label, ps, cap, stg in (("log-linear (deployed)", 0.0, 0, "pre"),
                                ("interference", best_scale, 0, "pre"),
                                ("amplification (pre-conditioning)", 0.0, 2, "pre"),
                                ("amplification (post-conditioning)", 0.0, 2, "post"),
                                ("interference + amplification", best_scale, 2, "pre")):
        r = run_dwell(lm, test[:25], 40.0, ps, cap, stg)
        dwell[label] = r
        print("  %-34s median=%4.1f mean=%5.2f frames  acc=%.4f  censored=%d"
              % (label, r["median_frames"], r["mean_frames"],
                 r["acc_at_commit"], r["censored"]))

    base_by = dwell["log-linear (deployed)"]["by_key"]
    amp_by = dwell["interference + amplification"]["by_key"]
    shared = sorted(set(base_by) & set(amp_by))
    dp = None
    if len(shared) >= 10:
        x = [base_by[i] for i in shared]
        y = [amp_by[i] for i in shared]
        dp = 1.0 if np.allclose(x, y) else float(
            wilcoxon(x, y, zero_method="zsplit").pvalue)
        print("  paired over %d keystrokes committed by both: p=%.4f"
              % (len(shared), dp))

    # ---- why ----
    print("\n" + "-" * 82)
    print("4. WHY — coherence on the correct letter vs the wrong ones")
    print("-" * 82)
    print("  %-8s %13s %13s %13s" % ("scale", "|c| correct", "|c| wrong", "separation"))
    disc = run_discrimination(lm, test[:40])
    for d in disc:
        print("  %-8.1f %13.4f %13.4f %+13.4f"
              % (d["scale"], d["coh_correct"], d["coh_wrong"], d["separation"]))
    print("  a positive separation would mean interference protects the correct"
          " letter")

    # ---- verdict ----
    print("\n" + "=" * 82)
    sig_win = [r for r in rows if r["rule_at_theta"] > r["base_at_theta"] and r["p"] < 0.05]
    sig_loss = [r for r in rows if r["rule_at_theta"] < r["base_at_theta"] and r["p"] < 0.05]
    all_d = [d for r in rows for d in r["deltas"]]
    med = float(np.median(all_d)) if all_d else float("nan")
    print("VERDICT")
    print("  accuracy at matched prompt rate: median delta %+.4f over %d"
          " (sigma, rate) cells" % (med, len(all_d)))
    print("  significant at THETA: %d wins, %d losses across %d noise levels"
          % (len(sig_win), len(sig_loss), len(rows)))
    d_base = dwell["log-linear (deployed)"]["median_frames"]
    d_amp = dwell["interference + amplification"]["median_frames"]
    print("  dwell: %.1f -> %.1f frames (%+.1f%%)%s"
          % (d_base, d_amp, 100.0 * (d_amp - d_base) / max(d_base, 1e-9),
             "" if dp is None else ", p=%.4f" % dp))
    print("=" * 82)

    out = {"phase_scale": best_scale, "train_acc_at_20pct": best_acc,
           "match_rates": list(MATCH_RATES), "sweep": rows,
           "dwell": {k: {kk: vv for kk, vv in v.items() if kk != "by_key"}
                     for k, v in dwell.items()},
           "dwell_paired_p": dp, "discrimination": disc,
           "theta": THETA, "lm_beta": LM_BETA,
           "frame_ess": FRAME_ESS, "median_delta": med,
           "seconds": round(time.time() - t0, 1)}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "selection_rule_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote %s  (%.0fs)" % (path, out["seconds"]))


if __name__ == "__main__":
    main()
