"""Retrain the movement gate on recorded human kinematics.

Why the target changed
----------------------
The deployed gate is a four-way movement-state classifier (fixation, saccade,
blink, drift) and it cannot be retrained on the recorded sessions, because
three of those four classes have no examples in them. Capture discards the
settling window after every target change -- the gap between retained windows
is a flat ~701 ms -- so every sample on disk is already a fixation. There is
nothing for a saccade class to learn from.

What *is* available for every retained frame is the thing the gate exists to
decide: whether the gaze evidence from that frame can be trusted. Leave-one-
target-out gives a held-out prediction error per sample, and the worst quartile
of that error is a real, recorded definition of "unreliable" -- chosen to match
the 25 % corruption rate the synthetic generator used, so the two settings stay
comparable. The classifier therefore predicts *reliability* directly instead of
predicting a movement state and inferring reliability from it.

The features are unchanged: ear, gaze velocity, stability, trailing velocity.
None of them sees the gaze error.

What this answers
-----------------
1. Does a gate trained on real kinematics beat one trained on synthetic ones?
2. Does it beat simply raising ``THETA`` until the keyboard asks as often --
   the control from ``eval/movement_gate_control.py``, which is the only
   comparison that separates a useful classifier from a threshold in disguise.
3. **Is the quantum kernel doing the work?** A classical RBF SVM on identical
   features, identical splits, answers that directly. If it matches the
   fidelity kernel, the gate is worth keeping and the quantum part is not.

Splits are grouped by target throughout: the five fixations recorded at one
position are one observation measured five times, and splitting by sample would
leave near-duplicates on both sides.

Run: ``python3 eval/retrain_movement_gate.py``
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval.feature_ablation import columns                       # noqa: E402
from eval.gaze_bench import load_sessions, session_quality      # noqa: E402
from eval.movement_gate_control import (                        # noqa: E402
    MATCH_RATES, at_commit_rate, frontier, kinematics,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "retrain_movement_gate.json"

UNRELIABLE_QUANTILE = 0.75      # worst quartile == "unreliable", as in synthetic
N_QSVM_TRAIN = 240              # the quantum Gram is O(n^2) circuits; this is the ceiling


# ==========================================================================
# Build the real-data training set
# ==========================================================================

def build() -> dict:
    """Per-sample kinematics, reliability labels, target posteriors, groups."""
    S = [s for s in load_sessions() if session_quality(s)["usable"]]
    K, y, P, G, SESS, ERR = [], [], [], [], [], []

    for si, s in enumerate(S):
        X = columns(s, "canthus")
        kin = kinematics(s)
        tids = np.unique(s.groups)
        centres = np.array([s.T[s.groups == g].mean(axis=0) for g in tids])

        pred = np.empty_like(s.T)
        sd = np.empty(len(s.T))
        for g in tids:
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

        err = np.linalg.norm(pred - s.T, axis=1) / s.px_per_degree
        # "unreliable" is defined per session, so a session that was simply
        # harder throughout does not swamp the label with its own difficulty
        thr = np.quantile(err, UNRELIABLE_QUANTILE)

        d2 = ((pred[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
        logits = -0.5 * d2 / (sd[:, None] ** 2)
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)

        K.append(kin)
        y.append((err > thr).astype(int))          # 1 == unreliable
        P.append(p)
        G.append(np.searchsorted(tids, s.groups))
        SESS.append(np.full(len(err), si))
        ERR.append(err)

    return {"K": np.concatenate(K), "y": np.concatenate(y),
            "P": np.concatenate(P), "G": np.concatenate(G),
            "sess": np.concatenate(SESS), "err": np.concatenate(ERR),
            "truth": np.concatenate([np.searchsorted(np.unique(s.groups), s.groups)
                                     for s in S]),
            "labels": [s.label for s in S]}


# ==========================================================================
# Models
# ==========================================================================

def models():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    def qsvm(Xtr, ytr, Xte):
        from quantum.movement_qsvm import MovementQSVM
        if len(Xtr) > N_QSVM_TRAIN:            # O(n^2) circuits: subsample
            idx = np.random.default_rng(0).choice(len(Xtr), N_QSVM_TRAIN, replace=False)
            Xtr, ytr = Xtr[idx], ytr[idx]
        m = MovementQSVM().fit(Xtr, ytr)
        return m.predict(Xte)

    def rbf(Xtr, ytr, Xte):
        m = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))
        m.fit(Xtr, ytr)
        return m.predict(Xte)

    def logistic(Xtr, ytr, Xte):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        m.fit(Xtr, ytr)
        return m.predict(Xte)

    def deployed(Xtr, ytr, Xte):
        """The shipped gate, unretrained — 1 means it refuses the frame."""
        from quantum.movement_qsvm import MovementQSVM
        g = MovementQSVM.load()
        return np.array([0 if g.classify_one(r)["usable"] else 1 for r in Xte])

    def thresh(Xtr, ytr, Xte):
        from quantum.movement_qsvm import threshold_rule
        return (np.asarray(threshold_rule(Xte)) != 0).astype(int)

    return [("QSVM fidelity kernel (retrained)", qsvm),
            ("classical RBF SVM (retrained)", rbf),
            ("logistic regression (retrained)", logistic),
            ("deployed gate (synthetic, as shipped)", deployed),
            ("threshold rule (pre-quantum baseline)", thresh)]


# ==========================================================================
# Evaluation
# ==========================================================================

def grouped_predict(fn, K, y, G, sess) -> np.ndarray:
    """Leave-one-target-out within each session, so no near-duplicate leaks."""
    out = np.zeros(len(y), dtype=int)
    for s in np.unique(sess):
        m = sess == s
        Ks, ys, Gs = K[m], y[m], G[m]
        pred = np.zeros(len(ys), dtype=int)
        for g in np.unique(Gs):
            te = Gs == g
            if ys[~te].sum() == 0 or ys[~te].sum() == len(ys[~te]):
                pred[te] = 0
                continue
            pred[te] = fn(Ks[~te], ys[~te], Ks[te])
        out[m] = pred
    return out


def main() -> None:
    t0 = time.time()
    print("=" * 78)
    print("RETRAINING THE MOVEMENT GATE ON RECORDED KINEMATICS")
    print("=" * 78)

    d = build()
    K, y, P, G, sess, truth = d["K"], d["y"], d["P"], d["G"], d["sess"], d["truth"]
    print("sessions: %s" % ", ".join(d["labels"]))
    print("%d recorded frames, %d labelled unreliable (%.1f%%), worst-quartile rule"
          % (len(y), y.sum(), 100 * y.mean()))
    print("NOTE: every recorded frame is a retained fixation. Capture discards the")
    print("      settling window, so there are no saccade or blink examples to")
    print("      learn from; the label is reliability, not movement state.\n")

    print("  %-40s %9s %9s %9s %8s"
          % ("model", "bal.acc", "recall", "precision", "seconds"))
    print("  " + "-" * 78)
    rows = []
    preds = {}
    for name, fn in models():
        t = time.time()
        try:
            p = grouped_predict(fn, K, y, G, sess)
            tp = int(np.sum((p == 1) & (y == 1)))
            fp = int(np.sum((p == 1) & (y == 0)))
            fn_ = int(np.sum((p == 0) & (y == 1)))
            tn = int(np.sum((p == 0) & (y == 0)))
            rec = tp / max(1, tp + fn_)
            spec = tn / max(1, tn + fp)
            prec = tp / max(1, tp + fp)
            preds[name] = p
            rows.append({"name": name, "balanced_acc": 0.5 * (rec + spec),
                         "recall": rec, "precision": prec,
                         "refuse_rate": float((p == 1).mean()),
                         "sec": time.time() - t})
            print("  %-40s %9.4f %9.4f %9.4f %8.1f"
                  % (name, 0.5 * (rec + spec), rec, prec, time.time() - t),
                  flush=True)
        except Exception as exc:
            rows.append({"name": name, "failed": "%s: %s" % (type(exc).__name__, exc)})
            print("  %-40s %9s  %s" % (name, "FAILED", str(exc)[:36]), flush=True)
    print("  (chance balanced accuracy is 0.5000)")

    # ---- the decisive control: does gating beat asking more often? ----
    print("\n" + "-" * 78)
    print("CONTROL: gate against simply raising THETA, at matched commit rate")
    print("-" * 78)
    thetas = np.unique(np.concatenate([np.linspace(0.0, 0.999, 200),
                                       np.quantile(P.max(axis=1),
                                                   np.linspace(0, 1, 100))]))
    un = frontier(P, truth, np.ones(len(truth), bool), thetas)
    control = []
    for name, p in preds.items():
        allow = p == 0                       # usable == predicted reliable
        ga = frontier(P, truth, allow, thetas)
        cells, wins, losses = [], 0, 0
        for r in MATCH_RATES:
            a, b = at_commit_rate(un, r), at_commit_rate(ga, r)
            if a is None or b is None:
                cells.append("  n/a")
                continue
            delta = a["wrong_per_100"] - b["wrong_per_100"]
            wins += delta > 0.01
            losses += delta < -0.01
            cells.append("%+6.2f" % delta)
        control.append({"name": name, "wins": wins, "losses": losses,
                        "deltas": cells})
        print("  %-40s %s   (%d win / %d loss)"
              % (name, " ".join(cells), wins, losses))
    print("  cells = wrong-commits/100 saved against raising THETA to the same"
          " commit rate")
    print("  positive means the gate beats the threshold; 6 cells per row")

    out = {"n_frames": int(len(y)), "unreliable_frac": float(y.mean()),
           "sessions": d["labels"], "classification": rows, "control": control,
           "seconds": round(time.time() - t0, 1)}
    RESULTS.write_text(json.dumps(out, indent=2, default=float))
    print("\nwrote %s  (%.0fs)" % (RESULTS, out["seconds"]))


if __name__ == "__main__":
    main()
