"""Systematic trial of qiskit-algorithms and scikit-learn candidates.

Every candidate is measured against the deployed incumbent on the *same*
recorded human sessions, under the same leave-one-target-out protocol, and
reported in degrees of visual angle. Nothing here is scored on synthetic data.

Part A is the one that can move the gaze number. Gaze is a regression from 12
canthus-normalised features to a screen coordinate, so only a candidate that
replaces or reshapes that regression can change the error in degrees. Most of
the qiskit-algorithms catalogue cannot: Grover, the amplitude estimators, the
phase estimators and the time evolvers are not regressors, and no amount of
wrapping makes them one. Where a legitimate mapping exists we construct it and
report the result; where it does not we say so rather than manufacturing a
number.

  A1  Classical regressors (scikit-learn)     -> degrees
  A2  Quantum fidelity kernel (ComputeUncompute) -> degrees
  A3  Feature selection as a QUBO, solved by QAOA / SamplingVQE /
      NumPyMinimumEigensolver, against exhaustive search over all 4096 subsets
  A4  qiskit-algorithms optimizers tuning the ridge penalty -> degrees

Run: ``python3 eval/algorithm_trials.py [part]``  (part in a1 a2 a3 a4 all)
"""

from __future__ import annotations

import itertools
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# eval/feature_ablation.py imports gaze_bench by bare name, so eval/ must be
# importable too or any caller outside this directory fails on import.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval.feature_ablation import columns                      # noqa: E402
from eval.gaze_bench import (                                  # noqa: E402
    fit_ridge, load_sessions, session_quality,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "algorithm_trials_results.json"


def _save(path, new: dict) -> None:
    """Merge into the results file rather than overwriting it.

    Each part of this script can be run on its own, and a plain write means the
    last part run is the only one on disk. Merging keeps every part's results
    together however they were produced.
    """
    old = {}
    if path.exists():
        try:
            old = json.loads(path.read_text())
        except Exception:
            old = {}
    old.update(new)
    path.write_text(json.dumps(old, indent=2, default=float))



# ==========================================================================
# Shared evaluation
# ==========================================================================

def sessions():
    return [s for s in load_sessions() if session_quality(s)["usable"]]


def loto_error(fit_predict, S, kind: str = "canthus") -> Tuple[float, List[float]]:
    """Leave-one-target-out error in degrees, pooled and per session.

    ``fit_predict(Xtr, Ttr, Xte) -> (m, 2)`` is the only thing a candidate must
    supply. Predictions are clipped to the screen exactly as the deployed
    pipeline clips them, so no candidate is scored on coordinates the system
    could never emit.
    """
    per = []
    for s in S:
        X = columns(s, kind)
        W, H = s.meta["screen"]["w_px"], s.meta["screen"]["h_px"]
        errs = []
        for g in np.unique(s.groups):
            te = s.groups == g
            P = np.asarray(fit_predict(X[~te], s.T[~te], X[te]), dtype=float)
            P[:, 0] = np.clip(P[:, 0], 0, W)
            P[:, 1] = np.clip(P[:, 1], 0, H)
            errs.append(np.sqrt((((P - s.T[te]) ** 2).sum(axis=1)).mean()))
        per.append(float(np.mean(errs)) / s.px_per_degree)
    return float(np.mean(per)), per


def ridge_baseline(Xtr, Ttr, Xte):
    """The deployed regressor: linear ridge by normal equations."""
    return Xte @ fit_ridge(Xtr, Ttr)


def report(rows: List[dict], base: float, title: str) -> None:
    print("\n  %-42s %9s %10s %10s" % (title, "degrees", "vs base", "seconds"))
    print("  " + "-" * 74)
    for r in rows:
        if r.get("failed"):
            print("  %-42s %9s %10s %10s" % (r["name"], "FAILED", "--", "--"))
            print("  %-42s   %s" % ("", r["failed"][:64]))
            continue
        d = r["deg"]
        rel = 100.0 * (base - d) / base
        mark = "" if abs(rel) < 0.05 else ("  <-" if rel > 0 else "")
        print("  %-42s %9.3f %+9.1f%% %10.1f%s"
              % (r["name"], d, rel, r.get("sec", 0.0), mark))


# ==========================================================================
# A1 — classical regressors
# ==========================================================================

def part_a1(S, base: float) -> List[dict]:
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.ensemble import (
        ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor,
    )
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.linear_model import (
        ARDRegression, BayesianRidge, ElasticNet, HuberRegressor, Lasso,
        LinearRegression, Ridge, RidgeCV, TheilSenRegressor,
    )
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.svm import SVR

    def wrap(make, multi=False):
        def f(Xtr, Ttr, Xte):
            m = make()
            if multi:
                m = MultiOutputRegressor(m)
            m.fit(Xtr, Ttr)
            return m.predict(Xte)
        return f

    cands = [
        ("sklearn Ridge (alpha=1e-4)", wrap(lambda: Ridge(alpha=1e-4))),
        ("sklearn RidgeCV (auto alpha)",
         wrap(lambda: RidgeCV(alphas=np.logspace(-6, 3, 40)))),
        ("OLS (no penalty)", wrap(LinearRegression)),
        ("BayesianRidge", wrap(BayesianRidge, multi=True)),
        ("ARDRegression", wrap(ARDRegression, multi=True)),
        ("Lasso", wrap(lambda: Lasso(alpha=1e-3, max_iter=5000))),
        ("ElasticNet", wrap(lambda: ElasticNet(alpha=1e-3, max_iter=5000))),
        ("HuberRegressor", wrap(HuberRegressor, multi=True)),
        ("TheilSen (robust)", wrap(lambda: TheilSenRegressor(max_subpopulation=2000),
                                   multi=True)),
        ("PLSRegression (n=6)", wrap(lambda: PLSRegression(n_components=6))),
        ("KernelRidge RBF", wrap(lambda: KernelRidge(kernel="rbf", alpha=1e-2,
                                                     gamma=0.5))),
        ("KernelRidge Laplacian",
         wrap(lambda: KernelRidge(kernel="laplacian", alpha=1e-2, gamma=0.5))),
        ("KernelRidge polynomial d=2",
         wrap(lambda: KernelRidge(kernel="polynomial", degree=2, alpha=1e-2))),
        ("Polynomial d=2 + Ridge",
         wrap(lambda: make_pipeline(PolynomialFeatures(2, include_bias=False),
                                    StandardScaler(), Ridge(alpha=1.0)))),
        ("SVR RBF", wrap(lambda: SVR(kernel="rbf", C=1e3, gamma="scale"),
                         multi=True)),
        ("KNeighbors (k=5)", wrap(lambda: KNeighborsRegressor(n_neighbors=5))),
        ("RandomForest", wrap(lambda: RandomForestRegressor(
            n_estimators=120, random_state=0, n_jobs=-1))),
        ("ExtraTrees", wrap(lambda: ExtraTreesRegressor(
            n_estimators=120, random_state=0, n_jobs=-1))),
        ("GradientBoosting", wrap(lambda: GradientBoostingRegressor(
            random_state=0), multi=True)),
        ("MLP (32,16)", wrap(lambda: make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=3000,
                         random_state=0)))),
        ("GaussianProcess RBF+White", wrap(lambda: GaussianProcessRegressor(
            kernel=RBF(1.0) + WhiteKernel(1e-2), normalize_y=True,
            random_state=0))),
    ]

    rows = []
    for name, fn in cands:
        t = time.time()
        try:
            deg, per = loto_error(fn, S)
            rows.append({"name": name, "deg": deg, "per": per,
                         "sec": time.time() - t})
        except Exception as exc:
            rows.append({"name": name, "failed": "%s: %s" % (type(exc).__name__, exc)})
    return rows


# ==========================================================================
# A2 — quantum fidelity kernel
# ==========================================================================

def part_a2(S, base: float, n_sub: int = 420) -> Tuple[List[dict], float]:
    """Kernel ridge on a quantum fidelity Gram matrix.

    ``ComputeUncompute`` evaluates one circuit per *pair*, so a Gram matrix over
    N samples costs O(N^2) circuit executions — about a million for a single
    session here, which is not a measurement, it is a week. The same quantity
    has an O(N) route: build each feature state once and take
    ``|<phi(x)|phi(z)>|^2`` from the inner products. That is exactly what
    ComputeUncompute converges to in the noiseless limit, and we verify the two
    agree on a small block before using the fast route at scale.

    Sessions are subsampled per target so every position survives; the classical
    RBF arm is scored on the identical subsample, so the comparison is fair even
    though these absolute numbers are not comparable with the full-data rows of
    A1.
    """
    import copy
    from qiskit import QuantumCircuit
    # Qiskit 2.5.2 on this machine segfaults inside its own feature-map builder
    # (both the deprecated ZZFeatureMap class and the functional
    # pauli_feature_map) once scikit-learn and pandas have been imported; it
    # works only in a fresh interpreter. The maps are a few lines of standard
    # gates, so we build them directly and avoid the crashing path. The
    # construction is verified against qiskit's own output in a clean process:
    # state overlap agrees to 1e-15 for Z and ZZ at 4-6 qubits, reps 1-2.
    from qiskit.circuit import ParameterVector

    def z_feature_map(n, reps=1):
        x = ParameterVector("x", n)
        qc = QuantumCircuit(n)
        for _ in range(reps):
            for i in range(n):
                qc.h(i)
            for i in range(n):
                qc.p(2.0 * x[i], i)
        return qc

    def zz_feature_map(n, reps=1):
        x = ParameterVector("x", n)
        qc = QuantumCircuit(n)
        for _ in range(reps):
            for i in range(n):
                qc.h(i)
            for i in range(n):
                qc.p(2.0 * x[i], i)
            for i in range(n):
                for j in range(i + 1, n):
                    qc.cx(i, j)
                    qc.p(2.0 * (np.pi - x[i]) * (np.pi - x[j]), j)
                    qc.cx(i, j)
        return qc
    from qiskit.primitives import StatevectorSampler
    from qiskit.quantum_info import Statevector
    from qiskit_algorithms.state_fidelities import ComputeUncompute
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import rbf_kernel
    from sklearn.preprocessing import MinMaxScaler

    def states(fmap, Xs):
        """One statevector per sample."""
        return np.array([Statevector(fmap.assign_parameters(x)).data for x in Xs])

    def fid(Va, Vb):
        return np.abs(Va.conj() @ Vb.T) ** 2

    print("    verifying the fast route against ComputeUncompute ...", flush=True)
    # ---- agreement check against the qiskit primitive itself ----
    print("      [1] building map", flush=True)
    rng0 = np.random.default_rng(0)
    fmap_chk = zz_feature_map(4, reps=1)
    print("      [2] map built", flush=True)
    Xc = rng0.uniform(0, np.pi, size=(6, 4))
    print("      [3] making statevectors", flush=True)
    Vc = states(fmap_chk, Xc)
    print("      [4] statevectors done", flush=True)
    exact_block = fid(Vc, Vc)
    print("      [5] constructing ComputeUncompute", flush=True)
    cu = ComputeUncompute(sampler=StatevectorSampler(default_shots=20000, seed=3))
    print("      [6] constructed", flush=True)
    xa = np.repeat(Xc, len(Xc), axis=0)
    xb = np.tile(Xc, (len(Xc), 1))
    sampled = np.asarray(
        cu.run([fmap_chk] * len(xa), [fmap_chk] * len(xa), xa, xb
               ).result().fidelities).reshape(len(Xc), len(Xc))
    print("    ComputeUncompute vs exact statevector fidelity: max |diff| = %.4f"
          % np.abs(sampled - exact_block).max(), flush=True)

    def make_quantum(fmap_factory, n_q, alpha=1e-2):
        def fn(Xtr, Ttr, Xte):
            pca = PCA(n_components=n_q, random_state=0).fit(Xtr)
            sc = MinMaxScaler((0, np.pi)).fit(pca.transform(Xtr))
            A = sc.transform(pca.transform(Xtr))
            B = sc.transform(pca.transform(Xte))
            fmap = fmap_factory(n_q)
            Va, Vb = states(fmap, A), states(fmap, B)
            K, Kt = fid(Va, Va), fid(Vb, Va)
            W = np.linalg.solve(K + alpha * np.eye(len(K)), Ttr)
            return Kt @ W
        return fn

    def make_rbf(alpha=1e-2):
        def fn(Xtr, Ttr, Xte):
            from scipy.spatial.distance import pdist
            g = 1.0 / (2.0 * np.median(pdist(Xtr)) ** 2 + 1e-12)
            K = rbf_kernel(Xtr, Xtr, gamma=g)
            Kt = rbf_kernel(Xte, Xtr, gamma=g)
            W = np.linalg.solve(K + alpha * np.eye(len(K)), Ttr)
            return Kt @ W
        return fn

    rng = np.random.default_rng(0)
    sub = []
    for s in S:
        keep = []
        ng = len(np.unique(s.groups))
        for g in np.unique(s.groups):
            idx = np.flatnonzero(s.groups == g)
            k = max(2, n_sub // max(1, ng))
            keep.append(rng.choice(idx, size=min(k, len(idx)), replace=False))
        keep = np.sort(np.concatenate(keep))
        s2 = copy.copy(s)
        s2.X, s2.T, s2.groups = s.X[keep], s.T[keep], s.groups[keep]
        s2.extras = [s.extras[i] for i in keep]
        sub.append(s2)
    print("    subsampled to %s samples per session" % [len(s.T) for s in sub])

    cands = [
        ("ridge (same subsample)", ridge_baseline),
        ("classical RBF kernel ridge", make_rbf()),
        ("ZFeatureMap fidelity, 4q", make_quantum(lambda n: z_feature_map(n, reps=1), 4)),
        ("ZFeatureMap fidelity, 8q", make_quantum(lambda n: z_feature_map(n, reps=1), 8)),
        ("ZZFeatureMap fidelity, 4q reps=1",
         make_quantum(lambda n: zz_feature_map(n, reps=1), 4)),
        ("ZZFeatureMap fidelity, 4q reps=2",
         make_quantum(lambda n: zz_feature_map(n, reps=2), 4)),
        ("ZZFeatureMap fidelity, 6q reps=1",
         make_quantum(lambda n: zz_feature_map(n, reps=1), 6)),
        ("ZZFeatureMap fidelity, 8q reps=1",
         make_quantum(lambda n: zz_feature_map(n, reps=1), 8)),
        ("ZZFeatureMap fidelity, 10q reps=1",
         make_quantum(lambda n: zz_feature_map(n, reps=1), 10)),
    ]

    rows, sub_base = [], None
    for name, fn in cands:
        t = time.time()
        try:
            deg, per = loto_error(fn, sub)
            if sub_base is None:
                sub_base = deg
            rows.append({"name": name, "deg": deg, "per": per, "sec": time.time() - t})
            print("      %-40s %.3f deg (%.0fs)" % (name, deg, time.time() - t),
                  flush=True)
        except Exception as exc:
            rows.append({"name": name, "failed": "%s: %s" % (type(exc).__name__, exc)})
            print("      %-40s FAILED %s" % (name, exc), flush=True)

    # diagnose concentration: how close does the Gram get to the identity?
    conc = []
    for nq in (4, 6, 8, 10, 12):
        X = columns(sub[0], "canthus")
        pca = PCA(n_components=nq, random_state=0).fit(X)
        sc = MinMaxScaler((0, np.pi)).fit(pca.transform(X))
        A = sc.transform(pca.transform(X))[:120]
        V = states(zz_feature_map(nq, reps=1), A)
        K = fid(V, V)
        off = K[~np.eye(len(K), dtype=bool)]
        conc.append({"qubits": nq, "mean_offdiag": float(off.mean()),
                     "max_offdiag": float(off.max())})
        print("      ZZ concentration at %2dq: mean off-diagonal fidelity %.5f"
              % (nq, off.mean()), flush=True)
    return rows, sub_base, conc


# ==========================================================================
# A3 — feature selection as a QUBO
# ==========================================================================

def part_a3(S, base: float) -> Tuple[List[dict], dict]:
    """Choose which of the 12 features to keep, as a minimum-eigenvalue problem.

    This is the one place a qiskit minimum-eigensolver has a well-posed job in
    this pipeline: a binary decision per feature, an objective that is the
    held-out gaze error, and a search space of $2^{12} = 4096$ that can be
    enumerated exactly. Exact enumeration therefore supplies ground truth, and
    the question is only whether QAOA and SamplingVQE find it.

    Scoring is made cheap by precomputing, for every (session, held-out target)
    fold, the full Gram matrices ``X'X`` and ``X'T`` on that fold's training
    rows. A feature mask is then a submatrix index and a 12x12 solve, so all
    4096 subsets cost seconds rather than hours.
    """
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.primitives import StatevectorSampler
    from qiskit_algorithms import QAOA, SamplingVQE, NumPyMinimumEigensolver
    from qiskit_algorithms.optimizers import COBYLA, SPSA, NELDER_MEAD, POWELL
    from qiskit.circuit.library import RealAmplitudes

    n = columns(S[0], "canthus").shape[1]
    print("    %d features -> %d subsets, enumerated exactly" % (n, 2 ** n))

    # ---- precompute folds ----
    folds = []
    for s in S:
        X = columns(s, "canthus")
        W, H = s.meta["screen"]["w_px"], s.meta["screen"]["h_px"]
        for g in np.unique(s.groups):
            te = s.groups == g
            Xtr, Ttr, Xte, Tte = X[~te], s.T[~te], X[te], s.T[te]
            folds.append({"A": Xtr.T @ Xtr, "B": Xtr.T @ Ttr, "Xte": Xte,
                          "Tte": Tte, "W": W, "H": H,
                          "ppd": s.px_per_degree, "sess": s.label})
    labels = [s.label for s in S]

    cache: Dict[int, float] = {}

    def score(mask: int) -> float:
        if mask in cache:
            return cache[mask]
        cols = np.flatnonzero([(mask >> i) & 1 for i in range(n)])
        if cols.size == 0:
            cache[mask] = 99.0
            return 99.0
        per = {lab: [] for lab in labels}
        for f in folds:
            A = f["A"][np.ix_(cols, cols)]
            lam = 1e-4 * (np.trace(A) / max(1, A.shape[0]))
            try:
                Wt = np.linalg.solve(A + lam * np.eye(A.shape[0]), f["B"][cols])
            except np.linalg.LinAlgError:
                cache[mask] = 99.0
                return 99.0
            P = f["Xte"][:, cols] @ Wt
            P[:, 0] = np.clip(P[:, 0], 0, f["W"])
            P[:, 1] = np.clip(P[:, 1], 0, f["H"])
            e = np.sqrt((((P - f["Tte"]) ** 2).sum(axis=1)).mean())
            per[f["sess"]].append(e / f["ppd"])
        d = float(np.mean([np.mean(v) for v in per.values()]))
        cache[mask] = d
        return d

    t = time.time()
    all_deg = np.array([score(m) for m in range(2 ** n)])
    best_mask = int(np.argmin(all_deg))
    best_deg = float(all_deg[best_mask])
    enum_sec = time.time() - t
    print("    exact optimum %.3f deg, mask %s (%d of %d features kept), %.1fs"
          % (best_deg, format(best_mask, "0%db" % n),
             bin(best_mask).count("1"), n, enum_sec))
    print("    all %d features (the deployed choice): %.3f deg"
          % (n, all_deg[2 ** n - 1]))

    # ---- degree-2 Ising surrogate of the landscape ----
    # The true objective is not quadratic in the selection bits, so the
    # operator a minimum-eigensolver can consume is a fitted surrogate. We
    # report its own optimum separately, so "the solver failed" and "the
    # surrogate was a poor proxy" stay distinguishable.
    idx = np.arange(2 ** n)
    bits = ((idx[:, None] >> np.arange(n)) & 1).astype(float)
    spins = 1.0 - 2.0 * bits                     # bit 1 (kept) -> spin -1
    design = [np.ones(2 ** n)] + [spins[:, i] for i in range(n)]
    pairs = list(itertools.combinations(range(n), 2))
    design += [spins[:, i] * spins[:, j] for i, j in pairs]
    A = np.column_stack(design)
    coef, *_ = np.linalg.lstsq(A, all_deg, rcond=None)
    fit = A @ coef
    r2 = 1.0 - np.sum((all_deg - fit) ** 2) / np.sum((all_deg - all_deg.mean()) ** 2)
    sur_mask = int(np.argmin(fit))
    print("    degree-2 Ising surrogate R^2 = %.4f; its optimum scores %.3f deg"
          % (r2, score(sur_mask)))

    terms = []
    for i in range(n):
        if abs(coef[1 + i]) > 1e-12:
            p = ["I"] * n
            p[n - 1 - i] = "Z"
            terms.append(("".join(p), float(coef[1 + i])))
    for k, (i, j) in enumerate(pairs):
        c = float(coef[1 + n + k])
        if abs(c) > 1e-12:
            p = ["I"] * n
            p[n - 1 - i] = "Z"
            p[n - 1 - j] = "Z"
            terms.append(("".join(p), c))
    H = SparsePauliOp.from_list(terms)

    def mask_of(bitstring: str) -> int:
        """Qiskit prints qubit n-1 first; spin -1 (measured |1>) means kept."""
        bs = bitstring[-n:].rjust(n, "0")
        m = 0
        for i in range(n):
            if bs[n - 1 - i] == "1":
                m |= 1 << i
        return m

    solvers = [
        ("NumPyMinimumEigensolver (exact on surrogate)",
         lambda: NumPyMinimumEigensolver().compute_minimum_eigenvalue(H)),
        ("QAOA p=1, COBYLA",
         lambda: QAOA(StatevectorSampler(default_shots=2048, seed=7), COBYLA(maxiter=120), reps=1
                      ).compute_minimum_eigenvalue(H)),
        ("QAOA p=2, COBYLA",
         lambda: QAOA(StatevectorSampler(default_shots=2048, seed=7), COBYLA(maxiter=150), reps=2
                      ).compute_minimum_eigenvalue(H)),
        ("QAOA p=3, COBYLA",
         lambda: QAOA(StatevectorSampler(default_shots=2048, seed=7), COBYLA(maxiter=180), reps=3
                      ).compute_minimum_eigenvalue(H)),
        ("QAOA p=2, SPSA",
         lambda: QAOA(StatevectorSampler(default_shots=2048, seed=7), SPSA(maxiter=60), reps=2
                      ).compute_minimum_eigenvalue(H)),
        ("QAOA p=2, NELDER_MEAD",
         lambda: QAOA(StatevectorSampler(default_shots=2048, seed=7), NELDER_MEAD(maxiter=120), reps=2
                      ).compute_minimum_eigenvalue(H)),
        ("QAOA p=2, POWELL",
         lambda: QAOA(StatevectorSampler(default_shots=2048, seed=7), POWELL(maxiter=120), reps=2
                      ).compute_minimum_eigenvalue(H)),
        ("SamplingVQE RealAmplitudes r=2, COBYLA",
         lambda: SamplingVQE(StatevectorSampler(default_shots=2048, seed=7), RealAmplitudes(n, reps=1),
                             COBYLA(maxiter=200)).compute_minimum_eigenvalue(H)),
        ("SamplingVQE RealAmplitudes r=3, SPSA",
         lambda: SamplingVQE(StatevectorSampler(default_shots=2048, seed=7), RealAmplitudes(n, reps=2),
                             SPSA(maxiter=60)).compute_minimum_eigenvalue(H)),
    ]

    rows = []
    for name, run in solvers:
        t = time.time()
        print("      running %s ..." % name, flush=True)
        try:
            res = run()
            bm = getattr(res, "best_measurement", None)
            if bm:
                m = mask_of(bm["bitstring"])
            else:                       # NumPyMinimumEigensolver -> statevector
                pr = res.eigenstate.probabilities()
                m = mask_of(format(int(np.argmax(pr)), "0%db" % n))
            rows.append({"name": name, "deg": score(m), "mask": m,
                         "optimal": m == best_mask,
                         "n_kept": bin(m).count("1"), "sec": time.time() - t})
        except Exception as exc:
            rows.append({"name": name,
                         "failed": "%s: %s" % (type(exc).__name__, exc)})

    # classical search baselines on the identical objective
    for nm, fn in [("greedy forward selection", "greedy")]:
        t = time.time()
        m, cur = 0, 99.0
        improved = True
        while improved:
            improved = False
            for i in range(n):
                cand = m | (1 << i)
                if cand == m:
                    continue
                d = score(cand)
                if d < cur - 1e-9:
                    cur, bestc, improved = d, cand, True
            if improved:
                m = bestc
        rows.append({"name": nm, "deg": cur, "mask": m, "optimal": m == best_mask,
                     "n_kept": bin(m).count("1"), "sec": time.time() - t})

    meta = {"n_features": n, "exact_best_deg": best_deg, "exact_best_mask": best_mask,
            "enum_seconds": enum_sec, "surrogate_r2": float(r2),
            "surrogate_best_mask": sur_mask, "surrogate_best_deg": score(sur_mask),
            "all_features_deg": float(all_deg[2 ** n - 1])}
    return rows, meta


# ==========================================================================
# A4 — qiskit optimizers tuning the ridge penalty
# ==========================================================================

def part_a4(S, base: float) -> List[dict]:
    """Every classical optimizer shipped in qiskit_algorithms.optimizers,
    pointed at the one continuous hyperparameter the regressor actually has.

    These are classical optimizers that happen to live in a quantum package.
    Using them here is legitimate and they are cheap; what they cannot do is
    make the regressor something other than ridge.
    """
    from qiskit_algorithms.optimizers import (
        ADAM, AQGD, CG, COBYLA, GSLS, GradientDescent, L_BFGS_B, NELDER_MEAD,
        NFT, P_BFGS, POWELL, SLSQP, SPSA, TNC, UMDA,
    )

    def objective(theta) -> float:
        lam = float(10.0 ** np.clip(np.atleast_1d(theta)[0], -8, 4))
        def fp(Xtr, Ttr, Xte):
            n = Xtr.shape[1]
            A = Xtr.T @ Xtr + lam * np.eye(n)
            return Xte @ np.linalg.solve(A, Xtr.T @ Ttr)
        try:
            d, _ = loto_error(fp, S)
        except Exception:
            d = 99.0
        return d

    x0 = np.array([-4.0])          # the deployed lam0 = 1e-4
    print("    deployed lambda=1e-4 scores %.3f deg" % objective(x0))

    opts = [
        ("COBYLA", COBYLA(maxiter=40)),
        ("NELDER_MEAD", NELDER_MEAD(maxiter=40)),
        ("POWELL", POWELL(maxiter=40)),
        ("SPSA", SPSA(maxiter=30)),
        ("L_BFGS_B", L_BFGS_B(maxiter=30)),
        ("SLSQP", SLSQP(maxiter=30)),
        ("CG", CG(maxiter=20)),
        ("TNC", TNC(maxiter=20)),
        ("P_BFGS", P_BFGS(maxfun=30)),
        ("ADAM", ADAM(maxiter=30)),
        ("GradientDescent", GradientDescent(maxiter=30, learning_rate=0.3)),
        ("AQGD", AQGD(maxiter=20)),
        ("GSLS", GSLS(maxiter=30)),
        ("NFT", NFT(maxiter=30)),
        ("UMDA", UMDA(maxiter=20, size_gen=12)),
    ]

    rows = []
    for name, opt in opts:
        t = time.time()
        try:
            res = opt.minimize(fun=objective, x0=x0.copy())
            lam = float(10.0 ** np.clip(np.atleast_1d(res.x)[0], -8, 4))
            rows.append({"name": "%s (lambda=%.2e)" % (name, lam),
                         "deg": float(res.fun), "lam": lam,
                         "nfev": int(res.nfev or 0), "sec": time.time() - t})
        except Exception as exc:
            rows.append({"name": name,
                         "failed": "%s: %s" % (type(exc).__name__, exc)})
    return rows


# ==========================================================================
# Driver
# ==========================================================================

def main() -> None:
    part = (sys.argv[1] if len(sys.argv) > 1 else "all").lower()
    S = sessions()
    print("=" * 80)
    print("ALGORITHM TRIALS — gaze accuracy in degrees, leave-one-target-out")
    print("=" * 80)
    print("sessions: %s" % ", ".join(s.label for s in S))

    base, base_per = loto_error(ridge_baseline, S)
    print("\nINCUMBENT  linear ridge on canthus-12: %.3f deg  (per session %s)"
          % (base, ", ".join("%.2f" % d for d in base_per)))

    out = {"baseline_deg": base, "baseline_per_session": base_per,
           "sessions": [s.label for s in S]}

    if part in ("a1", "all"):
        print("\n" + "=" * 80)
        print("A1. CLASSICAL REGRESSORS (scikit-learn)")
        print("=" * 80)
        r = part_a1(S, base)
        report(r, base, "candidate")
        out["a1"] = r

    if part in ("a2", "all"):
        print("\n" + "=" * 80)
        print("A2. QUANTUM FIDELITY KERNEL (qiskit_algorithms.state_fidelities)")
        print("=" * 80)
        r, sub_base, conc = part_a2(S, base)
        report(r, sub_base or base, "candidate (subsampled)")
        out["a2"] = r
        out["a2_subsample_baseline"] = sub_base
        out["a2_concentration"] = conc

    if part in ("a3", "all"):
        print("\n" + "=" * 80)
        print("A3. FEATURE SELECTION AS A QUBO (minimum eigensolvers)")
        print("=" * 80)
        r, meta = part_a3(S, base)
        print("\n  %-46s %9s %7s %8s" % ("solver", "degrees", "optimal", "seconds"))
        print("  " + "-" * 74)
        for row in r:
            if row.get("failed"):
                print("  %-46s %9s   %s" % (row["name"], "FAILED", row["failed"][:40]))
            else:
                print("  %-46s %9.3f %7s %8.1f"
                      % (row["name"], row["deg"],
                         "yes" if row["optimal"] else "no", row["sec"]))
        print("\n  exact optimum over all 4096 subsets: %.3f deg"
              % meta["exact_best_deg"])
        print("  all 12 features (the deployed choice): %.3f deg"
              % meta["all_features_deg"])
        out["a3"], out["a3_meta"] = r, meta

    if part in ("a4", "all"):
        print("\n" + "=" * 80)
        print("A4. QISKIT OPTIMIZERS TUNING THE RIDGE PENALTY")
        print("=" * 80)
        r = part_a4(S, base)
        report(r, base, "optimizer")
        out["a4"] = r

    _save(RESULTS, out)
    print("\nwrote %s" % RESULTS)


if __name__ == "__main__":
    main()
