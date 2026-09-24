"""Four regressors at the calibration size the system actually uses.

``eval/regression_benchmark.py`` trains leave-one-target-out, so each fit sees
20 of the 21 recorded targets. The deployed keyboard calibrates on **9**. That
difference decides the answer: kernel ridge beats ridge at 20 targets and loses
to it at 9, because its alpha and gamma have to be tuned by cross-validation
and at 9 targets there is not enough data to tune them on. Ridge's lambda is a
fixed heuristic with nothing to fit, so it degrades gracefully instead.

This script measures every arm across calibration sizes, and reports the
deployed size separately with a paired test, because that is the operating
point any deployment claim has to be made at.

Protocol
--------
For each session and each draw, ``n`` targets are sampled for training and
every remaining target is the test set. All four arms get the **identical**
split, so the comparison is paired and a Wilcoxon signed-rank test over draws
is the right significance test. Splits are by target, never by sample: the
~90 frames at one target are near-duplicates and splitting by sample lets a
high-capacity model memorise the position it is about to be tested on.

Error is mean Euclidean distance in degrees of visual angle, using each
session's own px_per_degree.

Run: python3 eval/calibration_size_benchmark.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from typing import Dict, List

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval.feature_ablation import columns                              # noqa: E402
from eval.gaze_bench import fit_ridge, load_sessions, session_quality  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "calibration_size_benchmark.json")

FEATURES = "canthus"
DEPLOYED_TARGETS = 9          # static/app.js TOTAL_CALIB
SIZES = (4, 6, 9, 12, 16, 20)
DRAWS_SWEEP = 5               # draws per session per size, for the curve
DRAWS_DEPLOYED = 15           # more draws at the size the claim is made at


def arm_ridge(Xtr, Ttr, Xte):
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return Xte @ fit_ridge(Xtr, Ttr)


def arm_ols(Xtr, Ttr, Xte):
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return Xte @ np.linalg.lstsq(Xtr, Ttr, rcond=None)[0]


def arm_krr(Xtr, Ttr, Xte):
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    gs = GridSearchCV(KernelRidge(kernel="rbf"),
                      {"alpha": np.logspace(-4, 2, 7),
                       "gamma": np.logspace(-3, 1, 5)},
                      cv=3, scoring="neg_mean_squared_error")
    m = make_pipeline(StandardScaler(), gs).fit(Xtr[:, 1:], Ttr)
    return np.asarray(m.predict(Xte[:, 1:]), dtype=float)


def arm_mlp(Xtr, Ttr, Xte):
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(StandardScaler(), MLPRegressor(
        hidden_layer_sizes=(16, 8), alpha=1e-2, max_iter=6000,
        random_state=0, early_stopping=True, n_iter_no_change=40))
    m.fit(Xtr[:, 1:], Ttr)
    return np.asarray(m.predict(Xte[:, 1:]), dtype=float)


ARMS = [("Ridge", "linear", arm_ridge),
        ("OLS", "linear", arm_ols),
        ("KRR-RBF", "non-linear", arm_krr),
        ("MLP", "non-linear", arm_mlp)]


def run(sizes, draws, seed=0) -> Dict[int, Dict[str, List[float]]]:
    """Per size, per arm, one error per (session, draw) — paired across arms."""
    S = [s for s in load_sessions() if session_quality(s)["usable"]]
    rng = np.random.default_rng(seed)
    out: Dict[int, Dict[str, List[float]]] = {n: {a: [] for a, _f, _fn in ARMS}
                                              for n in sizes}
    for n in sizes:
        for s in S:
            X, T, G = columns(s, FEATURES), s.T, s.groups
            tids = np.unique(G)
            if n >= len(tids):
                continue
            for _ in range(draws):
                tr = rng.choice(tids, n, replace=False)
                m = np.isin(G, tr)
                if m.sum() < 10 or (~m).sum() < 10:
                    continue
                for name, _fam, fn in ARMS:            # identical split per arm
                    P = np.asarray(fn(X[m], T[m], X[~m]), dtype=float)
                    e = float(np.linalg.norm(P - T[~m], axis=1).mean())
                    out[n][name].append(e / s.px_per_degree)
        print("    size %2d done" % n, flush=True)
    return out


def main() -> None:
    t0 = time.time()
    print("=" * 76)
    print("CALIBRATION-SIZE BENCHMARK — deployed calibration is %d targets"
          % DEPLOYED_TARGETS)
    print("=" * 76)

    print("\n1. Sweep across calibration sizes (%d draws/session/size)" % DRAWS_SWEEP)
    sweep = run(SIZES, DRAWS_SWEEP, seed=0)

    print("\n  %-6s" % "targets" + "".join("%12s" % a for a, _f, _fn in ARMS)
          + "%14s" % "best")
    print("  " + "-" * 68)
    for n in SIZES:
        means = {a: float(np.mean(sweep[n][a])) for a, _f, _fn in ARMS}
        best = min(means, key=means.get)
        print("  %-7d" % n + "".join("%12.3f" % means[a] for a, _f, _fn in ARMS)
              + "%14s" % best)
    print("  (mean gaze error in degrees, lower is better)")

    print("\n2. At the deployed %d targets (%d draws/session), paired"
          % (DEPLOYED_TARGETS, DRAWS_DEPLOYED))
    dep = run((DEPLOYED_TARGETS,), DRAWS_DEPLOYED, seed=1)[DEPLOYED_TARGETS]

    from scipy.stats import wilcoxon
    base = np.array(dep["Ridge"])
    print("\n  %-10s %-11s %9s %8s %11s %10s"
          % ("model", "family", "deg", "sd", "vs ridge", "paired p"))
    print("  " + "-" * 64)
    rows = []
    for name, fam, _fn in ARMS:
        v = np.array(dep[name])
        gain = 100.0 * (np.mean(v) - np.mean(base)) / np.mean(base)
        if name == "Ridge":
            p, txt = float("nan"), "  baseline"
        else:
            p = float(wilcoxon(base, v).pvalue)
            txt = "%+9.1f%%" % gain
        rows.append({"model": name, "family": fam, "deg": float(np.mean(v)),
                     "sd": float(np.std(v)), "vs_ridge_pct": float(gain),
                     "paired_p": p, "n_draws": int(len(v))})
        print("  %-10s %-11s %9.3f %8.3f %11s %10s"
              % (name, fam, np.mean(v), np.std(v), txt,
                 "-" if name == "Ridge" else "%.4f" % p))
    print("\n  'vs ridge' is the change in error: positive means WORSE than ridge.")
    print("  n = %d paired draws (%d sessions x %d)"
          % (len(base), len(base)//DRAWS_DEPLOYED, DRAWS_DEPLOYED))

    wins = sum(1 for r in rows if r["model"] != "Ridge"
               and r["deg"] > np.mean(base) and r["paired_p"] < 0.05)
    print("\n  ridge beats %d of 3 competitors significantly (p < 0.05) at %d targets"
          % (wins, DEPLOYED_TARGETS))

    out = {"features": FEATURES, "deployed_targets": DEPLOYED_TARGETS,
           "sizes": list(SIZES), "draws_sweep": DRAWS_SWEEP,
           "draws_deployed": DRAWS_DEPLOYED,
           "sweep_mean_deg": {str(n): {a: float(np.mean(sweep[n][a]))
                                       for a, _f, _fn in ARMS} for n in SIZES},
           "sweep_raw_deg": {str(n): {a: sweep[n][a] for a, _f, _fn in ARMS}
                             for n in SIZES},
           "deployed": rows, "seconds": round(time.time() - t0, 1)}
    with open(RESULTS, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print("\nwrote %s  (%.0fs)" % (RESULTS, out["seconds"]))


if __name__ == "__main__":
    main()
