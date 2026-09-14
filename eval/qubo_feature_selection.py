"""Feature selection as a minimum-eigenvalue problem, against exact enumeration.

This is the one place in this pipeline where a qiskit minimum-eigensolver has a
well-posed job: a binary keep/drop decision per feature, an objective that is
the held-out gaze error in degrees, and a search space small enough to
enumerate exactly. Exact enumeration therefore supplies ground truth, and the
only open question is whether QAOA and SamplingVQE find it.

Two things must stay distinguishable, so both are reported:

* the **true** objective, which is not quadratic in the selection bits; and
* the **degree-2 Ising surrogate** fitted to it, which is what the solvers
  actually minimise. A solver can only be as good as its surrogate, so the
  surrogate's own optimum is reported alongside. Without that, "the solver
  failed" and "the surrogate was a poor proxy" are indistinguishable.

Scoring is cheap because the Gram matrices ``X'X`` and ``X'T`` are precomputed
per (session, held-out target) fold; a feature mask is then a submatrix index
and one small solve.

Imports numpy and qiskit only: qiskit 2.5.2 on this machine segfaults during
circuit construction once scikit-learn and pandas are loaded (see
``eval/quantum_kernel_gaze.py``).

Run: ``python3 eval/qubo_feature_selection.py <features.npz> [n_features] [out.json]``
"""

from __future__ import annotations

import itertools
import json
import sys
import time

import numpy as np
from qiskit.circuit.library import real_amplitudes
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver, SamplingVQE
from qiskit_algorithms.optimizers import COBYLA, NELDER_MEAD, POWELL, SPSA

FEATURES = ["bias", "hL", "hR", "vL", "vR", "scale", "tx", "ty",
            "can0", "can1", "can2", "can3"]


def build_folds(npz, n_feat: int):
    """Precompute per-fold Gram matrices over the first ``n_feat`` features."""
    folds, labels = [], []
    for k in range(int(npz["n"][0])):
        X, T, g = npz["X%d" % k][:, :n_feat], npz["T%d" % k], npz["g%d" % k]
        W, H, ppd = npz["m%d" % k]
        labels.append(k)
        for t in np.unique(g):
            te = g == t
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                A, B = X[~te].T @ X[~te], X[~te].T @ T[~te]
            folds.append({"A": A, "B": B, "Xte": X[te], "Tte": T[te],
                          "W": W, "H": H, "ppd": ppd, "sess": k})
    return folds, labels


def make_score(folds, labels, n_feat):
    cache = {}

    def score(mask: int) -> float:
        if mask in cache:
            return cache[mask]
        cols = np.flatnonzero([(mask >> i) & 1 for i in range(n_feat)])
        if cols.size == 0:
            cache[mask] = 99.0
            return 99.0
        per = {l: [] for l in labels}
        for f in folds:
            A = f["A"][np.ix_(cols, cols)]
            lam = 1e-4 * (np.trace(A) / A.shape[0])
            try:
                Wt = np.linalg.solve(A + lam * np.eye(A.shape[0]), f["B"][cols])
            except np.linalg.LinAlgError:
                cache[mask] = 99.0
                return 99.0
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                P = f["Xte"][:, cols] @ Wt
            P[:, 0] = np.clip(P[:, 0], 0, f["W"])
            P[:, 1] = np.clip(P[:, 1], 0, f["H"])
            per[f["sess"]].append(
                np.sqrt((((P - f["Tte"]) ** 2).sum(axis=1)).mean()) / f["ppd"])
        d = float(np.mean([np.mean(v) for v in per.values()]))
        cache[mask] = d
        return d
    return score


def surrogate(all_deg: np.ndarray, n: int):
    """Least-squares degree-2 Ising fit to the enumerated landscape."""
    idx = np.arange(2 ** n)
    bits = ((idx[:, None] >> np.arange(n)) & 1).astype(float)
    spins = 1.0 - 2.0 * bits                 # bit 1 (kept) -> spin -1
    pairs = list(itertools.combinations(range(n), 2))
    design = ([np.ones(2 ** n)] + [spins[:, i] for i in range(n)]
              + [spins[:, i] * spins[:, j] for i, j in pairs])
    A = np.column_stack(design)
    coef, *_ = np.linalg.lstsq(A, all_deg, rcond=None)
    fit = A @ coef
    r2 = 1.0 - np.sum((all_deg - fit) ** 2) / np.sum((all_deg - all_deg.mean()) ** 2)

    terms = []
    for i in range(n):
        p = ["I"] * n
        p[n - 1 - i] = "Z"
        terms.append(("".join(p), float(coef[1 + i])))
    for k, (i, j) in enumerate(pairs):
        p = ["I"] * n
        p[n - 1 - i] = "Z"
        p[n - 1 - j] = "Z"
        terms.append(("".join(p), float(coef[1 + n + k])))
    return SparsePauliOp.from_list(terms), fit, float(r2)


def mask_of(bitstring: str, n: int) -> int:
    """Qiskit prints qubit n-1 first; spin -1 (measured |1>) means kept."""
    bs = bitstring[-n:].rjust(n, "0")
    m = 0
    for i in range(n):
        if bs[n - 1 - i] == "1":
            m |= 1 << i
    return m


def main() -> None:
    npz = np.load(sys.argv[1] if len(sys.argv) > 1 else "gaze_sub.npz")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    out_path = sys.argv[3] if len(sys.argv) > 3 else "qubo_feature_selection.json"

    print("=" * 78)
    print("FEATURE SELECTION AS A QUBO — %d features, %d subsets" % (n, 2 ** n))
    print("=" * 78)

    folds, labels = build_folds(npz, n)
    score = make_score(folds, labels, n)

    t = time.time()
    all_deg = np.array([score(m) for m in range(2 ** n)])
    enum_sec = time.time() - t
    best = int(np.argmin(all_deg))
    full = float(all_deg[2 ** n - 1])
    bits = lambda m: "".join("1" if (m >> i) & 1 else "." for i in range(n))

    print("exhaustive enumeration: %.2fs" % enum_sec)
    print("  all %d features        : %.4f deg" % (n, full))
    print("  best subset            : %.4f deg  (%+.2f%%)  %s"
          % (all_deg[best], 100.0 * (full - all_deg[best]) / full, bits(best)))
    print("  kept: %s" % ", ".join(FEATURES[i] for i in range(n) if (best >> i) & 1))

    H, fit, r2 = surrogate(all_deg, n)
    sur = int(np.argmin(fit))
    print("\ndegree-2 Ising surrogate: R^2 = %.4f, %d terms" % (r2, len(H)))
    print("  its optimum %s scores %.4f deg on the true objective"
          % (bits(sur), score(sur)))

    sv = lambda: StatevectorSampler(default_shots=1024, seed=7)
    solvers = [
        ("NumPyMinimumEigensolver (exact on surrogate)",
         lambda: NumPyMinimumEigensolver().compute_minimum_eigenvalue(H)),
        ("QAOA p=1, COBYLA",
         lambda: QAOA(sv(), COBYLA(maxiter=100), reps=1).compute_minimum_eigenvalue(H)),
        ("QAOA p=2, COBYLA",
         lambda: QAOA(sv(), COBYLA(maxiter=120), reps=2).compute_minimum_eigenvalue(H)),
        ("QAOA p=3, COBYLA",
         lambda: QAOA(sv(), COBYLA(maxiter=150), reps=3).compute_minimum_eigenvalue(H)),
        ("QAOA p=2, SPSA",
         lambda: QAOA(sv(), SPSA(maxiter=50), reps=2).compute_minimum_eigenvalue(H)),
        ("QAOA p=2, NELDER_MEAD",
         lambda: QAOA(sv(), NELDER_MEAD(maxiter=100), reps=2).compute_minimum_eigenvalue(H)),
        ("QAOA p=2, POWELL",
         lambda: QAOA(sv(), POWELL(maxiter=100), reps=2).compute_minimum_eigenvalue(H)),
        ("SamplingVQE real_amplitudes r=1, COBYLA",
         lambda: SamplingVQE(sv(), real_amplitudes(n, reps=1),
                             COBYLA(maxiter=150)).compute_minimum_eigenvalue(H)),
        ("SamplingVQE real_amplitudes r=2, SPSA",
         lambda: SamplingVQE(sv(), real_amplitudes(n, reps=2),
                             SPSA(maxiter=50)).compute_minimum_eigenvalue(H)),
    ]

    print("\n  %-44s %9s %9s %8s" % ("solver", "degrees", "optimal", "seconds"))
    print("  " + "-" * 74)
    rows = []
    for name, run in solvers:
        t = time.time()
        try:
            res = run()
            bm = getattr(res, "best_measurement", None)
            if bm:
                m = mask_of(bm["bitstring"], n)
            else:
                pr = res.eigenstate.probabilities()
                m = mask_of(format(int(np.argmax(pr)), "0%db" % n), n)
            d = score(m)
            rows.append({"name": name, "deg": d, "mask": m,
                         "optimal": m == best, "sec": time.time() - t})
            print("  %-44s %9.4f %9s %8.1f"
                  % (name, d, "yes" if m == best else "no", time.time() - t),
                  flush=True)
        except Exception as exc:
            rows.append({"name": name, "failed": "%s: %s" % (type(exc).__name__, exc)})
            print("  %-44s %9s  %s" % (name, "FAILED", str(exc)[:32]), flush=True)

    # classical controls on the identical objective
    for label, fn in [("greedy forward selection", "fwd"),
                      ("random search, 100 draws", "rand")]:
        t = time.time()
        if fn == "fwd":
            m, cur, improved = 0, 99.0, True
            while improved:
                improved = False
                for i in range(n):
                    c = m | (1 << i)
                    if c != m and score(c) < cur - 1e-9:
                        cur, bestc, improved = score(c), c, True
                if improved:
                    m = bestc
        else:
            rng = np.random.default_rng(0)
            cands = rng.integers(1, 2 ** n, size=100)
            m = int(min(cands, key=score))
            cur = score(m)
        rows.append({"name": label, "deg": cur, "mask": int(m),
                     "optimal": m == best, "sec": time.time() - t})
        print("  %-44s %9.4f %9s %8.1f"
              % (label, cur, "yes" if m == best else "no", time.time() - t))

    print("\n  exhaustive search found %.4f deg in %.2fs" % (all_deg[best], enum_sec))

    out = {"n_features": n, "exact_best_deg": float(all_deg[best]),
           "exact_best_mask": best, "all_features_deg": full,
           "enum_seconds": enum_sec, "surrogate_r2": r2,
           "surrogate_best_mask": sur, "surrogate_best_deg": score(sur),
           "solvers": rows}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("wrote %s" % out_path)


if __name__ == "__main__":
    main()
