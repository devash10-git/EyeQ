"""Does a nonlinear regressor beat the deployed ridge on real gaze?

Answers one question and changes nothing in the project: on the recorded
sessions, using identical grouped splits, is an MLP (or any other nonlinear
model) more accurate than the linear ridge currently in iris-tracker.js?

    python3 eval/compare_models.py

Method notes that decide whether the answer means anything:

* **Splits are grouped by target.** ~1200 fit samples come from only 13 distinct
  screen positions, and the ~90 samples at one position are near-duplicates. A
  model with enough capacity will memorise those 13 positions from almost
  identical features and score near zero on a per-sample split. Held-out targets
  are the only honest test, and they are the headline column here.
* **The constant column is dropped for nonlinear models.** f[0] is always 1.0,
  so standardising it divides by a zero standard deviation; the MLP gets its own
  intercept anyway.
* **Two training regimes are reported.** Per-sample uses every frame. Per-target
  averages the frames at each target first, which yields 13 clean points and
  removes the duplicate-weighting problem. Which is better is an empirical
  question, so both are measured.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from gaze_bench import (Session, by_region, fit_ridge, grouped_kfold, load_sessions,
                        metrics, predict, session_quality)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
SEEDS = (0, 1, 2, 3, 4)


# --------------------------------------------------------------- models

def make_models(seed: int, small: bool = False) -> Dict[str, object]:
    """Every comparator, built fresh per seed so results are independent.

    ``small`` is set for the per-target regime, where training is 13 points.
    Early stopping needs a validation slice and 10% of 13 points is not one, so
    there the regulariser ``alpha`` does the work instead.
    """
    stop = dict(early_stopping=not small, n_iter_no_change=40)
    return {
        # nonlinear candidates the user asked about
        "mlp_8":       make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(8,), alpha=1e-2, max_iter=6000,
                                     random_state=seed, **stop)),
        "mlp_16_8":    make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(16, 8), alpha=1e-2, max_iter=6000,
                                     random_state=seed, **stop)),
        "mlp_32_16":   make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-1, max_iter=6000,
                                     random_state=seed, **stop)),
        # the baselines that make the answer interpretable
        "krr_rbf":     make_pipeline(StandardScaler(), GridSearchCV(
                            KernelRidge(kernel="rbf"),
                            {"alpha": np.logspace(-4, 2, 7), "gamma": np.logspace(-3, 1, 5)},
                            cv=(2 if small else 3), scoring="neg_mean_squared_error")),
        "poly2_ridge": make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False),
                                     GridSearchCV(KernelRidge(kernel="linear"),
                                                  {"alpha": np.logspace(-4, 2, 7)}, cv=(2 if small else 3),
                                                  scoring="neg_mean_squared_error")),
        "poly3_ridge": make_pipeline(StandardScaler(), PolynomialFeatures(3, include_bias=False),
                                     GridSearchCV(KernelRidge(kernel="linear"),
                                                  {"alpha": np.logspace(-4, 2, 7)}, cv=(2 if small else 3),
                                                  scoring="neg_mean_squared_error")),
    }


def fit_predict(name: str, model, Xtr, Ttr, Xte, screen) -> np.ndarray:
    """Nonlinear models see the 7 informative features; ridge keeps all 8."""
    if name == "ridge_linear":
        return predict(fit_ridge(Xtr, Ttr), Xte, screen)
    model.fit(Xtr[:, 1:], Ttr)              # drop the constant column
    P = np.asarray(model.predict(Xte[:, 1:]), dtype=float)
    P[:, 0] = np.clip(P[:, 0], 0, screen[0])
    P[:, 1] = np.clip(P[:, 1], 0, screen[1])
    return P


# --------------------------------------------------------------- regimes

def per_target_mean(X, T, groups):
    """Collapse each target's frames to one point: 13 clean observations."""
    uniq = np.unique(groups)
    Xm = np.array([X[groups == g].mean(axis=0) for g in uniq])
    Tm = np.array([T[groups == g].mean(axis=0) for g in uniq])
    return Xm, Tm, uniq


def evaluate(sess: Session) -> dict:
    screen = (sess.meta["screen"]["w_px"], sess.meta["screen"]["h_px"])
    fm, hm = sess.split == "fit", sess.split == "holdout"
    Xf, Tf, Gf = sess.X[fm], sess.T[fm], sess.groups[fm]
    Xh, Th = sess.X[hm], sess.T[hm]

    names = ["ridge_linear"] + list(make_models(0).keys())
    out: Dict[str, dict] = {}

    for regime in ("per_sample", "per_target"):
        small = regime == "per_target"
        Xtr, Ttr = per_target_mean(Xf, Tf, Gf)[:2] if small else (Xf, Tf)

        for name in names:
            runs = []
            for seed in SEEDS:
                P = fit_predict(name, make_models(seed, small).get(name), Xtr, Ttr, Xh, screen)
                runs.append(metrics(P, Th, sess))
                if name == "ridge_linear":
                    break          # deterministic; one run is all of them
            agg = lambda k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs])))
            hp, hs = agg("rmse_px")
            entry = {"holdout_rmse_px": hp, "holdout_rmse_px_sd": hs,
                     "holdout_rmse_deg": agg("rmse_deg")[0], "n_seeds": len(runs)}
            if "mean_keypitch" in runs[0]:
                entry["holdout_keypitch"] = agg("mean_keypitch")[0]
            Pf = fit_predict(name, make_models(0, small).get(name), Xtr, Ttr, Xh, screen)
            entry["by_region"] = {r: round(v["rmse_deg"], 2) for r, v in by_region(Pf, Th, sess).items()}
            out[f"{regime}/{name}"] = entry

    return out


def main() -> None:
    sessions = [s for s in load_sessions() if session_quality(s)["usable"]]
    if not sessions:
        print("No usable recorded sessions. Record with eval/capture.py first.")
        return

    report = {"data_source": "recorded_human", "n_sessions": len(sessions), "seeds": len(SEEDS),
              "sessions": {}}
    pooled: Dict[str, List[float]] = {}

    for s in sessions:
        res = evaluate(s)
        report["sessions"][s.label] = res
        print(f"\n=== {s.label} — held-out targets, mean of {len(SEEDS)} seeds ===")
        print(f"  {'model':<24} {'px':>9}  {'+/-':>6}  {'deg':>6}  {'keys':>5}   by region (deg)")
        base = res["per_sample/ridge_linear"]["holdout_rmse_px"]
        for key, e in sorted(res.items(), key=lambda kv: kv[1]["holdout_rmse_px"]):
            pooled.setdefault(key, []).append(e["holdout_rmse_px"])
            mark = "  <- deployed" if key == "per_sample/ridge_linear" else (
                   f"  {(1 - e['holdout_rmse_px'] / base) * 100:+.0f}%" if e["holdout_rmse_px"] != base else "")
            reg = " ".join(f"{r[:3]} {v}" for r, v in sorted(e["by_region"].items()))
            print(f"  {key:<24} {e['holdout_rmse_px']:9.1f}  {e['holdout_rmse_px_sd']:6.1f}  "
                  f"{e['holdout_rmse_deg']:6.2f}  {e.get('holdout_keypitch', float('nan')):5.2f}   {reg}{mark}")

    print(f"\n=== pooled over {len(sessions)} sessions (mean held-out RMSE) ===")
    base = float(np.mean(pooled["per_sample/ridge_linear"]))
    for key, vals in sorted(pooled.items(), key=lambda kv: np.mean(kv[1])):
        m = float(np.mean(vals))
        delta = (1 - m / base) * 100
        verdict = "deployed baseline" if key == "per_sample/ridge_linear" else \
                  (f"{delta:+.1f}% vs deployed" + ("  BETTER" if delta > 0 else "  worse"))
        print(f"  {key:<24} {m:8.1f} px   {verdict}")
    report["pooled"] = {k: float(np.mean(v)) for k, v in pooled.items()}

    Path("model_comparison.json").write_text(json.dumps(report, indent=2))
    print("\nwrote model_comparison.json")


if __name__ == "__main__":
    main()
