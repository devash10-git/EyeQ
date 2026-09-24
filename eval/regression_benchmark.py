"""Four regressors for gaze calibration, judged on what the project needs.

Two linear (ridge, ordinary least squares) and two non-linear (RBF kernel
ridge, MLP), on the deployed canthus feature set, scored on four axes rather
than one:

    ACCURACY    mean gaze error in degrees of visual angle, leave-one-target-out
    PROJECT     what that error does to typing: commit accuracy, prompt rate,
                and entry rate through the real keyboard geometry
    EFFICIENCY  fit time and per-sample inference latency
    COST        hyperparameters that must be searched, parameters stored, and
                whether the fit can run in the browser at calibration time

A regressor can win one axis and lose another, and the table reports the winner
of each rather than collapsing them into a single verdict. That is the point:
the deployed choice is not only "whichever has the lowest error".

Nothing here modifies the shipped pipeline. ``fit_ridge`` is imported from
``eval/gaze_bench.py`` exactly as the deployed calibration solves it -- normal
equations with an escalating ridge term, mirroring the Gauss-Jordan solve in
``static/iris-tracker.js`` -- so the ridge arm is the real one, not a
reimplementation. The other three exist only in this file.

Why leave-one-target-out. ~1200 fit samples come from 21 screen positions, and
the frames at one position are near-duplicates. A model with enough capacity
memorises those positions from almost identical features and scores near zero
on a per-sample split. Held-out targets are the only honest test, and every
arm gets the identical folds.

How gaze error becomes typing error. Each arm's leave-one-target-out residuals
are its real, measured error. Those residuals -- not a fitted Gaussian -- are
resampled onto key centres to drive the same keystroke simulation the rest of
the project uses, so the typing numbers inherit each regressor's actual error
distribution including its tails. Residuals are scaled from session screen
pixels into keyboard pixels by assuming the keyboard spans the screen width,
which is how it is laid out in ``static/style.css``.

Run: python3 eval/regression_benchmark.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings
from typing import Dict, List, Sequence, Tuple

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval.feature_ablation import columns                       # noqa: E402
from eval.gaze_bench import fit_ridge, load_sessions, session_quality  # noqa: E402
from eval.selection_rule_bench import (                         # noqa: E402
    CENTRES, KEY_W, LM_BETA, LM_FLOOR, THETA, gaze_posterior,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "regression_benchmark.json")

FEATURES = "canthus"        # the deployed 12-dim set (paper Table: 4.37 deg)
DWELL_S = 0.693             # static/app.js DWELL_MS, for the entry-rate model

# Typing is scored over several resampling seeds, not one. A single seed put
# ridge at 96.63% and kernel ridge at 95.79% and the ordering looked decisive;
# across five seeds ridge is 95.33 +- 1.74 and kernel ridge 95.56 +- 0.93, so
# that ordering was a draw of the residual sampler and not a property of either
# regressor. Anything inside the combined spread is reported as a tie.
TYPING_SEEDS = (7, 11, 13, 17, 19)


# ==========================================================================
# The four arms
# ==========================================================================

def arm_ridge(Xtr, Ttr, Xte):
    """The deployed calibration, imported rather than reimplemented."""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return Xte @ fit_ridge(Xtr, Ttr)


def arm_ols(Xtr, Ttr, Xte):
    """Ordinary least squares: ridge with the regulariser removed.

    The informative contrast with ridge, because it isolates what the penalty
    buys. The design matrix is near-singular at 20 training targets, so this is
    where a linear fit goes wrong without one.
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        W = np.linalg.lstsq(Xtr, Ttr, rcond=None)[0]
        return Xte @ W


def _sk_krr():
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), GridSearchCV(
        KernelRidge(kernel="rbf"),
        {"alpha": np.logspace(-4, 2, 7), "gamma": np.logspace(-3, 1, 5)},
        cv=3, scoring="neg_mean_squared_error"))


def _sk_mlp(seed=0):
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), MLPRegressor(
        hidden_layer_sizes=(16, 8), alpha=1e-2, max_iter=6000,
        random_state=seed, early_stopping=True, n_iter_no_change=40))


def _sk_fit_predict(make, Xtr, Ttr, Xte):
    # column 0 is the constant 1.0; standardising it divides by a zero standard
    # deviation, and both models carry their own intercept.
    m = make()
    m.fit(Xtr[:, 1:], Ttr)
    return np.asarray(m.predict(Xte[:, 1:]), dtype=float)


ARMS = [
    ("Ridge (deployed)",      "linear",     arm_ridge,
     {"hyperparams_searched": 0, "closed_form": True}),
    ("Ordinary least squares", "linear",    arm_ols,
     {"hyperparams_searched": 0, "closed_form": True}),
    ("Kernel ridge (RBF)",    "non-linear",
     lambda a, b, c: _sk_fit_predict(_sk_krr, a, b, c),
     {"hyperparams_searched": 35, "closed_form": False}),
    ("MLP (16-8)",            "non-linear",
     lambda a, b, c: _sk_fit_predict(_sk_mlp, a, b, c),
     {"hyperparams_searched": 0, "closed_form": False}),
]


# ==========================================================================
# Stage 1 — gaze error, leave-one-target-out
# ==========================================================================

def run_gaze() -> Dict[str, dict]:
    S = [s for s in load_sessions() if session_quality(s)["usable"]]
    out: Dict[str, dict] = {}

    print("sessions: %s" % ", ".join(s.label for s in S))
    print("features: %s (%d dims)\n" % (FEATURES, columns(S[0], FEATURES).shape[1]))
    print("  %-24s %-11s %9s %9s %10s %11s"
          % ("model", "family", "deg", "px", "fit ms", "predict us"))
    print("  " + "-" * 78)

    for name, family, fn, cost in ARMS:
        per_deg, per_px, fit_ms, pred_us, resid = [], [], [], [], []
        for s in S:
            X, T = columns(s, FEATURES), s.T
            W, H = s.meta["screen"]["w_px"], s.meta["screen"]["h_px"]
            pred = np.empty_like(T)
            for g in np.unique(s.groups):
                te = s.groups == g
                t0 = time.perf_counter()
                P = np.asarray(fn(X[~te], T[~te], X[te]), dtype=float)
                dt = time.perf_counter() - t0
                fit_ms.append(dt * 1000.0)
                pred_us.append(dt * 1e6 / max(1, te.sum()))
                pred[te] = P
            pred[:, 0] = np.clip(pred[:, 0], 0, W)
            pred[:, 1] = np.clip(pred[:, 1], 0, H)
            err = np.linalg.norm(pred - T, axis=1)
            per_px.append(float(err.mean()))
            per_deg.append(float(err.mean()) / s.px_per_degree)
            # residuals in KEYBOARD pixels: the keyboard spans the screen width
            resid.append((pred - T) * (10.0 * KEY_W / W))

        out[name] = {
            "family": family,
            "deg": float(np.mean(per_deg)), "px": float(np.mean(per_px)),
            "per_session_deg": per_deg,
            "fit_ms": float(np.mean(fit_ms)),
            "predict_us": float(np.mean(pred_us)),
            "residuals_kbpx": np.concatenate(resid),
            **cost,
        }
        print("  %-24s %-11s %9.3f %9.1f %10.2f %11.2f"
              % (name, family, out[name]["deg"], out[name]["px"],
                 out[name]["fit_ms"], out[name]["predict_us"]), flush=True)
    return out


# ==========================================================================
# Stage 2 — what that error does to typing
# ==========================================================================

def run_typing(residuals: np.ndarray, lm, seed: int = 7) -> dict:
    """Type the benchmark sentences with this regressor's measured error.

    The gaze lands at the key centre plus a residual drawn from the arm's own
    leave-one-target-out error, so an arm with heavy tails is penalised for
    them rather than being represented by a fitted sigma it does not have.
    """
    from tune import load_sentences
    pairs = load_sentences(60, seed=4, skip=2060)
    rng = np.random.default_rng(seed)
    sigma = float(np.sqrt((residuals ** 2).sum(axis=1).mean() / 2.0))

    presented = committed = correct = 0
    for _code, en in pairs:
        for word in en.split():
            typed = ""
            for ch in word:
                if ch not in CENTRES:
                    continue
                cx, cy = CENTRES[ch]
                dx, dy = residuals[rng.integers(len(residuals))]
                pg = gaze_posterior(cx + dx, cy + dy, sigma)
                presented += 1
                if not pg:
                    continue
                prior = lm.letter_prior(typed)
                post = {k: v * max(prior.get(k, 0.0), LM_FLOOR) ** LM_BETA
                        for k, v in pg.items()}
                z = sum(post.values()) or 1.0
                post = {k: v / z for k, v in post.items()}
                best = max(post, key=post.get)
                if post[best] >= THETA:
                    committed += 1
                    correct += best == ch
                    typed += best
                else:
                    typed += ch
    prompts = presented - committed
    mins = DWELL_S * (presented + prompts) / 60.0
    return {"commit_accuracy_pct": 100.0 * correct / max(1, committed),
            "committed": committed, "correct": correct,
            "incorrect": committed - correct,
            "presented": presented, "prompts": prompts,
            "prompt_rate_pct": 100.0 * prompts / max(1, presented),
            "wpm": (presented / 5.0) / mins if mins > 0 else float("nan"),
            "sigma_kbpx": sigma}


# ==========================================================================
# Reporting
# ==========================================================================

def winners(res: Dict[str, dict]) -> List[Tuple[str, str, str]]:
    """Best arm on each axis, with the value that won it."""
    def best(key, lo=True, fmt="%.3f", src=None):
        pick = min if lo else max
        n = pick(res, key=lambda k: (res[k][src] if src else res[k])[key])
        v = (res[n][src] if src else res[n])[key]
        return n, fmt % v

    rows = []
    n, v = best("deg"); rows.append(("Accuracy — mean gaze error (deg)", n, v))
    ranked = sorted(res, key=lambda k: -res[k]["typing"]["commit_accuracy_pct"])
    top, second = ranked[0], ranked[1]
    gap = (res[top]["typing"]["commit_accuracy_pct"]
           - res[second]["typing"]["commit_accuracy_pct"])
    spread = (res[top]["typing"]["commit_accuracy_sd"]
              + res[second]["typing"]["commit_accuracy_sd"])
    if gap < spread:
        rows.append(("Project — typing commit accuracy",
                     "TIE (%s / %s)" % (top.split()[0], second.split()[0]),
                     "%.2f%% vs %.2f%%, gap %.2f < spread %.2f"
                     % (res[top]["typing"]["commit_accuracy_pct"],
                        res[second]["typing"]["commit_accuracy_pct"], gap, spread)))
    else:
        rows.append(("Project — typing commit accuracy", top,
                     "%.2f%%" % res[top]["typing"]["commit_accuracy_pct"]))
    n, v = best("prompt_rate_pct", fmt="%.1f%%", src="typing")
    rows.append(("Project — prompt rate (lower better)", n, v))
    n, v = best("wpm", lo=False, fmt="%.2f wpm", src="typing")
    rows.append(("Project — entry rate", n, v))
    n, v = best("fit_ms", fmt="%.2f ms"); rows.append(("Efficiency — fit time", n, v))
    n, v = best("predict_us", fmt="%.2f us"); rows.append(("Efficiency — inference", n, v))
    n, v = best("hyperparams_searched", fmt="%.0f")
    rows.append(("Cost — hyperparameters to search", n, v))
    return rows


def main() -> None:
    t0 = time.time()
    print("=" * 80)
    print("REGRESSION BENCHMARK — 2 linear, 2 non-linear, on the deployed features")
    print("=" * 80)

    res = run_gaze()

    print("\n" + "-" * 80)
    print("END-TO-END: each regressor's MEASURED error driving the typing pipeline")
    print("-" * 80)
    from decoder import HybridDecoder
    from letters import LetterModel
    from tune import VOCAB
    lm = LetterModel(HybridDecoder(vocab_path=VOCAB, model_dir=None,
                                   num_clusters=9).unigrams)
    print("  %-24s %16s %11s %9s %9s"
          % ("model", "commit acc (5 seeds)", "prompt rate", "wpm", "sigma px"))
    print("  " + "-" * 74)
    for name in res:
        runs = [run_typing(res[name]["residuals_kbpx"], lm, seed=sd)
                for sd in TYPING_SEEDS]
        agg = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
        agg["commit_accuracy_sd"] = float(np.std([r["commit_accuracy_pct"]
                                                  for r in runs]))
        agg["seeds"] = list(TYPING_SEEDS)
        agg["per_seed_accuracy"] = [r["commit_accuracy_pct"] for r in runs]
        res[name]["typing"] = agg
        print("  %-24s %8.2f%% +-%.2f %10.1f%% %9.2f %9.1f"
              % (name, agg["commit_accuracy_pct"], agg["commit_accuracy_sd"],
                 agg["prompt_rate_pct"], agg["wpm"], agg["sigma_kbpx"]),
              flush=True)

    print("\n" + "=" * 80)
    print("WINNER BY AXIS")
    print("=" * 80)
    wins = winners(res)
    for axis, who, val in wins:
        print("  %-38s %-24s %s" % (axis, who, val))

    tally: Dict[str, int] = {}
    for _a, who, _v in wins:
        if who.startswith("TIE"):
            continue          # a tie is not a win for anybody
        tally[who] = tally.get(who, 0) + 1
    print("\n  axes won: " + ", ".join("%s %d" % (k, v) for k, v in
                                       sorted(tally.items(), key=lambda x: -x[1])))

    out = {"features": FEATURES, "protocol": "leave-one-target-out, grouped",
           "arms": {n: {k: v for k, v in d.items() if k != "residuals_kbpx"}
                    for n, d in res.items()},
           "winner_by_axis": [{"axis": a, "model": m, "value": v} for a, m, v in wins],
           "axes_won": tally, "seconds": round(time.time() - t0, 1)}
    with open(RESULTS, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print("\nwrote %s  (%.0fs)" % (RESULTS, out["seconds"]))


if __name__ == "__main__":
    main()
