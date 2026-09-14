"""Quantum fidelity-kernel ridge for gaze, in a clean interpreter.

Why this is a separate script. Qiskit 2.5.2 on this machine segfaults inside
its own circuit construction once scikit-learn and pandas have been imported —
reproducibly, and in the deprecated ``ZZFeatureMap`` class, the functional
``pauli_feature_map``, and even a hand-built circuit of plain gates. It only
survives in an interpreter that has not loaded that stack. So the quantum arm
runs here, importing nothing but numpy and qiskit; PCA and min-max scaling are
done with numpy rather than sklearn for the same reason. The main trial script
writes the features to an ``.npz`` and shells out to this one.

What it measures. Kernel ridge regression on a fidelity Gram matrix
``K(x,z) = |<phi(x)|phi(z)>|^2``, scored by leave-one-target-out on the same
recorded human sessions as every other candidate, in degrees of visual angle.
The classical RBF arm and the deployed linear ridge are scored on the identical
subsample, so the three are directly comparable.

A note on cost. ``ComputeUncompute`` evaluates one circuit per *pair*, so a Gram
matrix over N samples is O(N^2) circuit executions — about a million per session
here. The same quantity is available in O(N) by building each feature state once
and taking inner products, which is exactly what ComputeUncompute converges to
without shot noise. We verify the two agree on a small block, then use the fast
route at scale.

Run: ``python3 eval/quantum_kernel_gaze.py <features.npz> [out.json]``
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector


# ==========================================================================
# Feature maps, built from gates
# ==========================================================================

def z_feature_map(n: int, reps: int = 1) -> QuantumCircuit:
    """First-order Pauli-Z map: H, then a phase proportional to each feature."""
    x = ParameterVector("x", n)
    qc = QuantumCircuit(n)
    for _ in range(reps):
        for i in range(n):
            qc.h(i)
        for i in range(n):
            qc.p(2.0 * x[i], i)
    return qc


def zz_feature_map(n: int, reps: int = 1) -> QuantumCircuit:
    """Second-order map: the Z map plus an entangling pairwise phase.

    The pairwise term is what makes the induced kernel conjectured hard to
    simulate classically, and also what drives the fidelity between distinct
    samples towards zero as the qubit count rises.
    """
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


def states(fmap: QuantumCircuit, Xs: np.ndarray) -> np.ndarray:
    return np.array([Statevector(fmap.assign_parameters(v)).data for v in Xs])


def fidelity(Va: np.ndarray, Vb: np.ndarray) -> np.ndarray:
    # numpy 2.x on Apple Accelerate raises divide/overflow/invalid flags from the
    # vectorised matmul path even on bounded inputs. The project already
    # established these are spurious (agreement with an explicit dot-product
    # reference to 2.8e-13); they are silenced here and nowhere wider, and the
    # caller asserts the result is finite.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        K = np.abs(Va.conj() @ Vb.T) ** 2
    assert np.all(np.isfinite(K)), "fidelity produced non-finite values"
    return K


# ==========================================================================
# numpy stand-ins for the sklearn pieces
# ==========================================================================

def pca_fit(X: np.ndarray, k: int):
    mu = X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[:k].T


def pca_apply(X: np.ndarray, fit) -> np.ndarray:
    mu, W = fit
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        Z = (X - mu) @ W
    assert np.all(np.isfinite(Z)), "PCA produced non-finite values"
    return Z


def minmax_fit(X: np.ndarray):
    lo, hi = X.min(axis=0), X.max(axis=0)
    return lo, np.where(hi - lo < 1e-12, 1.0, hi - lo)


def minmax_apply(X: np.ndarray, fit, hi: float = np.pi) -> np.ndarray:
    lo, span = fit
    return np.clip((X - lo) / span, 0.0, 1.0) * hi


# ==========================================================================
# Regressors
# ==========================================================================

def fit_ridge(X, T, lam0=1e-4):
    """The deployed calibration, reproduced exactly (eval/gaze_bench.py)."""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        A0, B = X.T @ X, X.T @ T
    lam = lam0 * (np.trace(A0) / max(1, A0.shape[0]))
    for _ in range(8):
        try:
            W = np.linalg.solve(A0 + lam * np.eye(A0.shape[0]), B)
            if np.all(np.isfinite(W)):
                return W
        except np.linalg.LinAlgError:
            pass
        lam *= 10
    return np.linalg.lstsq(X, T, rcond=None)[0]


def arm_ridge(Xtr, Ttr, Xte):
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return Xte @ fit_ridge(Xtr, Ttr)


def arm_rbf(alpha=1e-2):
    def fn(Xtr, Ttr, Xte):
        d2 = ((Xtr[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
        med = np.median(np.sqrt(d2[np.triu_indices(len(Xtr), 1)]))
        g = 1.0 / (2.0 * med ** 2 + 1e-12)
        K = np.exp(-g * d2)
        Kt = np.exp(-g * ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1))
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            return Kt @ np.linalg.solve(K + alpha * np.eye(len(K)), Ttr)
    return fn


def arm_quantum(make_map, n_q, alpha=1e-2):
    def fn(Xtr, Ttr, Xte):
        p = pca_fit(Xtr, n_q)
        m = minmax_fit(pca_apply(Xtr, p))
        A = minmax_apply(pca_apply(Xtr, p), m)
        B = minmax_apply(pca_apply(Xte, p), m)
        fm = make_map(n_q)
        Va, Vb = states(fm, A), states(fm, B)
        K, Kt = fidelity(Va, Va), fidelity(Vb, Va)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            return Kt @ np.linalg.solve(K + alpha * np.eye(len(K)), Ttr)
    return fn


# ==========================================================================
# Evaluation
# ==========================================================================

def loto(fn, data):
    per = []
    for X, T, g, meta in data:
        W, H, ppd = meta
        errs = []
        for t in np.unique(g):
            te = g == t
            P = np.asarray(fn(X[~te], T[~te], X[te]), dtype=float)
            P[:, 0] = np.clip(P[:, 0], 0, W)
            P[:, 1] = np.clip(P[:, 1], 0, H)
            errs.append(np.sqrt((((P - T[te]) ** 2).sum(axis=1)).mean()))
        per.append(float(np.mean(errs)) / ppd)
    return float(np.mean(per)), per


def verify_against_computeuncompute() -> float:
    """Confirm the O(N) statevector route matches the qiskit primitive."""
    from qiskit.primitives import StatevectorSampler
    from qiskit_algorithms.state_fidelities import ComputeUncompute
    fm = zz_feature_map(4, reps=1)
    X = np.random.default_rng(0).uniform(0, np.pi, size=(6, 4))
    exact = fidelity(states(fm, X), states(fm, X))
    cu = ComputeUncompute(sampler=StatevectorSampler(default_shots=20000, seed=3))
    xa = np.repeat(X, len(X), axis=0)
    xb = np.tile(X, (len(X), 1))
    got = np.asarray(cu.run([fm] * len(xa), [fm] * len(xa), xa, xb
                            ).result().fidelities).reshape(len(X), len(X))
    return float(np.abs(got - exact).max())


def concentration(data, qubits=(4, 6, 8, 10, 12)) -> list:
    """How close does the ZZ Gram matrix get to the identity as n grows?

    Exponential concentration is the standard failure mode of fidelity kernels:
    past a few qubits the overlap between distinct samples collapses, the Gram
    matrix approaches the identity, and the model can only memorise.
    """
    X = data[0][0]
    out = []
    for nq in qubits:
        p = pca_fit(X, nq)
        m = minmax_fit(pca_apply(X, p))
        A = minmax_apply(pca_apply(X, p), m)[:120]
        K = fidelity(*(states(zz_feature_map(nq), A),) * 2)
        off = K[~np.eye(len(K), dtype=bool)]
        out.append({"qubits": nq, "mean_offdiag": float(off.mean()),
                    "max_offdiag": float(off.max())})
        print("    ZZ at %2dq: mean off-diagonal fidelity %.6f  (max %.4f)"
              % (nq, off.mean(), off.max()), flush=True)
    return out


def main() -> None:
    npz = np.load(sys.argv[1] if len(sys.argv) > 1 else "gaze_sub.npz")
    n = int(npz["n"][0])
    data = [(npz["X%d" % k], npz["T%d" % k], npz["g%d" % k], npz["m%d" % k])
            for k in range(n)]
    print("=" * 78)
    print("QUANTUM FIDELITY-KERNEL RIDGE FOR GAZE (clean interpreter)")
    print("=" * 78)
    print("sessions: %s samples" % [len(d[1]) for d in data])

    d = verify_against_computeuncompute()
    print("ComputeUncompute vs O(N) statevector route: max |diff| = %.4f"
          " (shot noise at 20k shots)\n" % d)

    arms = [
        ("linear ridge (deployed)", arm_ridge),
        ("classical RBF kernel ridge", arm_rbf()),
        ("Z feature map, 4q", arm_quantum(lambda q: z_feature_map(q), 4)),
        ("Z feature map, 8q", arm_quantum(lambda q: z_feature_map(q), 8)),
        ("ZZ feature map, 4q reps=1", arm_quantum(lambda q: zz_feature_map(q, 1), 4)),
        ("ZZ feature map, 4q reps=2", arm_quantum(lambda q: zz_feature_map(q, 2), 4)),
        ("ZZ feature map, 6q reps=1", arm_quantum(lambda q: zz_feature_map(q, 1), 6)),
        ("ZZ feature map, 8q reps=1", arm_quantum(lambda q: zz_feature_map(q, 1), 8)),
        ("ZZ feature map, 10q reps=1", arm_quantum(lambda q: zz_feature_map(q, 1), 10)),
    ]

    rows, base = [], None
    print("  %-34s %9s %10s %9s" % ("arm", "degrees", "vs ridge", "seconds"))
    print("  " + "-" * 66)
    for name, fn in arms:
        t = time.time()
        try:
            deg, per = loto(fn, data)
            if base is None:
                base = deg
            rows.append({"name": name, "deg": deg, "per": per,
                         "sec": time.time() - t})
            print("  %-34s %9.3f %+9.1f%% %9.1f"
                  % (name, deg, 100.0 * (base - deg) / base, time.time() - t),
                  flush=True)
        except Exception as exc:
            rows.append({"name": name, "failed": "%s: %s" % (type(exc).__name__, exc)})
            print("  %-34s %9s  %s" % (name, "FAILED", str(exc)[:30]), flush=True)

    print("\n  exponential concentration of the ZZ kernel:")
    conc = concentration(data)

    out = {"arms": rows, "baseline_deg": base, "concentration": conc,
           "computeuncompute_agreement": d}
    path = sys.argv[2] if len(sys.argv) > 2 else "quantum_kernel_gaze.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("\nwrote %s" % path)


if __name__ == "__main__":
    main()
