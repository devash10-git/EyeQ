"""Honest validation of the regressor candidates: leave-one-session-out.

``algorithm_trials.py`` scores every candidate by leave-one-target-out over all
three sessions and reports the best. That number cannot be believed. Twenty-one
candidates were scored on one metric and the best was taken, so the winner's
margin contains however much of the noise happened to point its way — the
classic multiple-comparison trap, and with three sessions the noise is not
small.

This script answers the question that matters instead: *if you had to choose a
regressor without seeing the session you will be judged on, what would you get?*

For each held-out session, every candidate is scored by leave-one-target-out on
the other two sessions only; the winner of that selection is then evaluated on
the held-out session. The held-out number is never used to choose anything. The
gap between the two numbers is the selection bias, and reporting it is the whole
point of the exercise.

Run: ``python3 eval/nested_validation.py``
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.algorithm_trials import loto_error, ridge_baseline, sessions  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "nested_validation_results.json"


def candidates() -> List[Tuple[str, Callable]]:
    """The incumbent, the A1 winners, and the tuned-lambda variant."""
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.linear_model import HuberRegressor, Ridge, RidgeCV, TheilSenRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.svm import SVR

    def wrap(make, multi=False):
        def f(Xtr, Ttr, Xte):
            m = MultiOutputRegressor(make()) if multi else make()
            m.fit(Xtr, Ttr)
            return m.predict(Xte)
        return f

    def plain_ridge(lam):
        def f(Xtr, Ttr, Xte):
            A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
            return Xte @ np.linalg.solve(A, Xtr.T @ Ttr)
        return f

    return [
        ("deployed fit_ridge (incumbent)", ridge_baseline),
        ("ridge, lambda=1e-4 unscaled", plain_ridge(1e-4)),
        ("ridge, lambda=1e-5 unscaled", plain_ridge(1e-5)),
        ("sklearn RidgeCV", wrap(lambda: RidgeCV(alphas=np.logspace(-6, 3, 40)))),
        ("KernelRidge RBF", wrap(lambda: KernelRidge(kernel="rbf", alpha=1e-2,
                                                     gamma=0.5))),
        ("Polynomial d=2 + Ridge",
         wrap(lambda: make_pipeline(PolynomialFeatures(2, include_bias=False),
                                    StandardScaler(), Ridge(alpha=1.0)))),
        ("SVR RBF", wrap(lambda: SVR(kernel="rbf", C=1e3, gamma="scale"), multi=True)),
        ("TheilSen (robust)",
         wrap(lambda: TheilSenRegressor(max_subpopulation=2000), multi=True)),
        ("HuberRegressor", wrap(HuberRegressor, multi=True)),
    ]


def main() -> None:
    S = sessions()
    cands = candidates()
    print("=" * 84)
    print("NESTED VALIDATION — leave-one-session-out selection")
    print("=" * 84)
    print("sessions: %s" % ", ".join(s.label for s in S))
    print("candidates: %d\n" % len(cands))

    # full-data scores, for the selection-bias comparison
    full: Dict[str, float] = {}
    for name, fn in cands:
        try:
            full[name], _ = loto_error(fn, S)
        except Exception:
            full[name] = float("nan")

    rows = []
    picked = []
    for i, held in enumerate(S):
        inner = [s for j, s in enumerate(S) if j != i]
        scores = {}
        for name, fn in cands:
            try:
                scores[name], _ = loto_error(fn, inner)
            except Exception:
                scores[name] = float("inf")
        best = min(scores, key=scores.get)
        fn = dict(cands)[best]
        outer, _ = loto_error(fn, [held])
        base_outer, _ = loto_error(ridge_baseline, [held])
        picked.append(best)
        rows.append({"held_out": held.label, "picked": best,
                     "inner_deg": scores[best], "outer_deg": outer,
                     "incumbent_outer_deg": base_outer,
                     "gain_pct": 100.0 * (base_outer - outer) / base_outer})
        print("  held out %-24s picked %-28s" % (held.label, best))
        print("      chosen on the other two: %.3f deg" % scores[best])
        print("      on the held-out session: %.3f deg   (incumbent %.3f, %+.1f%%)"
              % (outer, base_outer, rows[-1]["gain_pct"]))

    print("\n" + "-" * 84)
    print("  %-34s %11s %11s %11s" % ("candidate", "full-data", "honest", "bias"))
    print("  " + "-" * 80)

    # honest per-candidate number: each candidate evaluated on each session
    # alone, averaged — no selection involved at all
    honest = {}
    for name, fn in cands:
        per = []
        for s in S:
            try:
                d, _ = loto_error(fn, [s])
            except Exception:
                d = float("nan")
            per.append(d)
        honest[name] = float(np.mean(per))

    base_full = full["deployed fit_ridge (incumbent)"]
    base_hon = honest["deployed fit_ridge (incumbent)"]
    for name, _ in cands:
        print("  %-34s %11.3f %11.3f %11s"
              % (name, full[name], honest[name],
                 "%+.3f" % (honest[name] - full[name])))

    sel_gain = float(np.mean([r["gain_pct"] for r in rows]))
    print("\n  selection under leave-one-session-out gains %+.1f%% over the incumbent"
          % sel_gain)
    print("  selections made: %s" % ", ".join(sorted(set(picked))))
    if len(set(picked)) > 1:
        print("  NOTE: the selection is not stable across held-out sessions,")
        print("        which is itself evidence that the margins are inside the noise.")

    out = {"sessions": [s.label for s in S], "folds": rows,
           "full_data": full, "honest": honest,
           "mean_gain_pct": sel_gain, "picked": picked}
    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("\nwrote %s" % RESULTS)


if __name__ == "__main__":
    main()
