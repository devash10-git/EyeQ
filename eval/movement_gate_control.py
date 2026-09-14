"""Does the quantum movement gate earn its place, or is it a threshold in disguise?

The gate (``quantum/movement_qsvm.py``, a fidelity-kernel SVM) refuses to commit
a keystroke sampled during a saccade or blink. Its reported effect is large:
wrong commits fall from 12.67 to 0.47 per 100 fixations. But that number has a
confound, and the confound is not small — the gate also drops the commit rate
from 0.733 to 0.576. Refusing more keystrokes reduces wrong keystrokes by
definition. The keyboard can already refuse more by raising ``THETA``, with no
classifier at all and no 9.76 ms per frame.

So the question is not "does gating help" but **"does the gate beat simply
asking more often?"** That means comparing the whole frontier of
wrong-commits against commit-rate, and checking whether the gated curve lies
strictly below the ungated one rather than being the same curve at a different
point.

There is a second problem. The gate was fitted and scored entirely on synthetic
kinematics. Nothing in ``movement_qsvm_report.json`` touches a recorded human.
So this script runs the comparison twice:

* **Real data** — kinematics pulled from the recorded sessions
  (``ear_l``/``ear_r``, ``gaze_velocity``, ``quality_geometric``), with
  leave-one-target-out gaze predictions and the recorded target as ground
  truth. The decision is which of the 21 targets the user was looking at.
* **Synthetic** — the same generator the gate's own report used, so the
  published numbers are reproduced and the control is added to them directly.

Run: ``python3 eval/movement_gate_control.py [real|synthetic|all]``
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "movement_gate_control.json"

# Commit rates at which the arms are compared head to head.
MATCH_RATES = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75)


# ==========================================================================
# Frontier machinery
# ==========================================================================

def frontier(post: np.ndarray, truth: np.ndarray, allow: np.ndarray,
             thetas: Sequence[float]) -> List[dict]:
    """Wrong-commits per 100 against commit rate, as THETA is swept.

    ``allow`` is the gate's verdict; an all-true mask gives the ungated arm.
    Both arms are scored on identical posteriors and identical ground truth, so
    the only difference between their curves is the gate.
    """
    top = post.argmax(axis=1)
    conf = post.max(axis=1)
    n = len(truth)
    out = []
    for th in thetas:
        commit = (conf >= th) & allow
        nc = int(commit.sum())
        wrong = int(np.sum(commit & (top != truth)))
        right = nc - wrong
        out.append({"theta": float(th),
                    "commit_rate": nc / max(1, n),
                    "wrong_per_100": 100.0 * wrong / max(1, n),
                    "acc_when_committed": right / max(1, nc) if nc else 1.0})
    return out


def at_commit_rate(curve: List[dict], rate: float) -> Optional[dict]:
    """Interpolate the curve to a given commit rate."""
    pts = sorted(curve, key=lambda r: r["commit_rate"])
    lo = [r for r in pts if r["commit_rate"] <= rate]
    hi = [r for r in pts if r["commit_rate"] >= rate]
    if not lo or not hi:
        return None
    a, b = lo[-1], hi[0]
    span = b["commit_rate"] - a["commit_rate"]
    t = 0.0 if abs(span) < 1e-12 else (rate - a["commit_rate"]) / span
    return {k: a[k] + t * (b[k] - a[k]) for k in
            ("wrong_per_100", "acc_when_committed", "theta")}


def compare(post, truth, usable, label: str, thetas) -> dict:
    """Ungated against gated at matched commit rate."""
    un = frontier(post, truth, np.ones(len(truth), dtype=bool), thetas)
    ga = frontier(post, truth, usable, thetas)

    print("\n  %s" % label)
    print("  %-12s %14s %14s %12s" % ("commit rate", "ungated w/100",
                                      "gated w/100", "gate helps?"))
    print("  " + "-" * 60)
    rows, wins, losses = [], 0, 0
    for r in MATCH_RATES:
        a, b = at_commit_rate(un, r), at_commit_rate(ga, r)
        if a is None or b is None:
            print("  %-12.2f %14s %14s %12s" % (r, "n/a", "n/a", "-"))
            continue
        d = a["wrong_per_100"] - b["wrong_per_100"]
        verdict = "yes" if d > 0.01 else ("no" if d < -0.01 else "tie")
        wins += verdict == "yes"
        losses += verdict == "no"
        rows.append({"commit_rate": r, "ungated": a["wrong_per_100"],
                     "gated": b["wrong_per_100"], "delta": d,
                     "ungated_theta": a["theta"], "gated_theta": b["theta"]})
        print("  %-12.2f %14.2f %14.2f %12s" % (r, a["wrong_per_100"],
                                                b["wrong_per_100"], verdict))
    print("  -> gate wins %d, loses %d, of %d matched rates" % (wins, losses, len(rows)))
    return {"label": label, "matched": rows, "wins": wins, "losses": losses,
            "ungated_curve": un, "gated_curve": ga}


# ==========================================================================
# Real recorded data
# ==========================================================================

def kinematics(sess) -> np.ndarray:
    """The gate's four inputs, built from what the sessions actually recorded.

    Order matches ``MOVEMENT_FEATURES``: ear, gaze velocity, stability, history.
    ``history`` is not stored, so it is reconstructed the way the tracker
    maintains it — a short trailing mean of recent velocity.
    """
    ex = sess.extras
    ear = np.array([(e.get("ear_l", 0.0) + e.get("ear_r", 0.0)) / 2.0 for e in ex])
    vel = np.array([e.get("gaze_velocity", 0.0) for e in ex], float)
    sta = np.array([e.get("quality_geometric", 0.0) for e in ex], float)
    hist = np.convolve(vel, np.ones(5) / 5.0, mode="same")
    return np.column_stack([ear, vel, sta, hist])


def real_data():
    """Per-sample posteriors over the 21 recorded targets, plus kinematics."""
    from eval.feature_ablation import columns
    from eval.gaze_bench import fit_ridge, load_sessions, session_quality

    S = [s for s in load_sessions() if session_quality(s)["usable"]]
    P_all, y_all, K_all = [], [], []
    for s in S:
        X = columns(s, "canthus")
        kin = kinematics(s)
        tids = np.unique(s.groups)
        centres = np.array([s.T[s.groups == g].mean(axis=0) for g in tids])
        pred = np.empty_like(s.T)
        sd = np.empty(len(s.T))
        for g in tids:                       # leave-one-target-out predictions
            te = s.groups == g
            Xtr, Ttr = X[~te], s.T[~te]
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                A = Xtr.T @ Xtr
                lam = 1e-4 * (np.trace(A) / A.shape[0])
                Ainv = np.linalg.inv(A + lam * np.eye(A.shape[0]))
                W = Ainv @ (Xtr.T @ Ttr)
                pred[te] = X[te] @ W
                resid = Ttr - Xtr @ W
                s2 = float((resid ** 2).sum() / max(1, len(resid) - X.shape[1])) / 2.0
                lev = np.einsum("ij,jk,ik->i", X[te], Ainv, X[te])
            sd[te] = np.sqrt(np.maximum(s2 * (1.0 + lev), 1.0))

        # posterior over the 21 targets: Gaussian on distance, per-sample width
        d2 = ((pred[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
        logits = -0.5 * d2 / (sd[:, None] ** 2)
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)

        truth = np.searchsorted(tids, s.groups)
        P_all.append(p)
        y_all.append(truth)
        K_all.append(kin)

    return (np.concatenate(P_all), np.concatenate(y_all),
            np.concatenate(K_all), [s.label for s in S])


def run_real() -> dict:
    from quantum.movement_qsvm import MovementQSVM

    post, truth, kin, labels = real_data()
    print("REAL RECORDED DATA")
    print("  sessions: %s" % ", ".join(labels))
    print("  %d samples, %d targets" % (len(truth), post.shape[1]))

    qsvm = MovementQSVM.load()
    lo, hi = np.asarray(qsvm.scaler.lo), np.asarray(qsvm.scaler.hi)
    names = ["ear", "velocity", "stability", "history"]
    print("\n  distribution check against the gate's training range:")
    oob = []
    for i, nm in enumerate(names):
        col = kin[:, i]
        frac = float(np.mean((col < lo[i]) | (col > hi[i])))
        oob.append(frac)
        print("    %-10s recorded [%8.3f, %8.3f]   trained [%6.3f, %6.3f]"
              "   %5.1f%% outside" % (nm, col.min(), col.max(), lo[i], hi[i],
                                      100 * frac))

    verdict = np.array([qsvm.classify_one(r)["usable"] for r in kin])
    print("\n  gate calls %.1f%% of recorded frames usable" % (100 * verdict.mean()))
    if verdict.all() or not verdict.any():
        print("  NOTE: the gate is saturated on this data — it makes no"
              " distinction, so it cannot help or hurt.")

    thetas = np.unique(np.concatenate([np.linspace(0.0, 0.999, 200),
                                       np.quantile(post.max(axis=1),
                                                   np.linspace(0, 1, 100))]))
    res = compare(post, truth, verdict, "real data, 21-target decision", thetas)
    res["usable_fraction"] = float(verdict.mean())
    res["out_of_range_fraction"] = dict(zip(names, oob))
    res["sessions"] = labels
    return res


# ==========================================================================
# Synthetic — reproduce the gate's own report, with the control added
# ==========================================================================

def run_synthetic(noise: float) -> dict:
    from quantum.gaze_letter import (
        GazeLetterStack, KINEMATIC_IDX, make_commit_dataset,
    )
    from quantum.movement_qsvm import MovementQSVM

    stack = GazeLetterStack.load()
    Xt, yt, st = make_commit_dataset(1500, noise=noise, seed=31)
    kin = Xt[:, list(KINEMATIC_IDX)]
    post = stack.geom.posterior(Xt)
    qsvm = MovementQSVM.load()
    usable = np.array([qsvm.classify_one(r)["usable"] for r in kin])

    print("\nSYNTHETIC (the generator the gate's own report used), noise=%.3f" % noise)
    print("  %d fixations, %.0f%% corrupted, gate calls %.1f%% usable"
          % (len(yt), 100 * float(np.mean(st > 0)), 100 * usable.mean()))

    # the published operating point, reproduced
    for nm, allow in (("ungated", np.ones(len(yt), bool)), ("qsvm_gate", usable)):
        top, conf = post.argmax(1), post.max(1)
        c = (conf >= 0.80) & allow
        w = int(np.sum(c & (top != yt)))
        print("    published point  %-10s commit %.3f  wrong/100 %6.2f"
              % (nm, c.mean(), 100.0 * w / len(yt)))

    thetas = np.unique(np.concatenate([np.linspace(0.0, 0.999, 200),
                                       np.quantile(post.max(axis=1),
                                                   np.linspace(0, 1, 100))]))
    res = compare(post, yt, usable, "synthetic, noise=%.3f" % noise, thetas)
    res["usable_fraction"] = float(usable.mean())
    res["noise"] = noise
    return res


# ==========================================================================
# Driver
# ==========================================================================

def main() -> None:
    which = (sys.argv[1] if len(sys.argv) > 1 else "all").lower()
    out = {}
    print("=" * 74)
    print("MOVEMENT GATE — does it beat simply asking more often?")
    print("=" * 74)

    if which in ("real", "all"):
        out["real"] = run_real()

    if which in ("synthetic", "all"):
        out["synthetic"] = [run_synthetic(n) for n in (0.02, 0.03)]

    print("\n" + "=" * 74)
    print("VERDICT")
    for key, res in (("real", out.get("real")),):
        if res:
            print("  real data     : gate wins %d / loses %d of %d matched"
                  " commit rates" % (res["wins"], res["losses"],
                                     len(res["matched"])))
    for res in out.get("synthetic", []):
        print("  %-14s: gate wins %d / loses %d of %d matched commit rates"
              % (res["label"], res["wins"], res["losses"], len(res["matched"])))
    print("=" * 74)

    RESULTS.write_text(json.dumps(out, indent=2, default=float))
    print("wrote %s" % RESULTS)


if __name__ == "__main__":
    main()
