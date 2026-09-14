"""Calibration: raw gaze features -> screen coordinates. Is quantum justified?

The question, stated precisely
-----------------------------
Calibration fits ``w`` in ``screen = F w`` from the user's calibration clicks.
The deployed solver is ridge regression:

    w = (F^T F + lam I)^-1 F^T t      minimising   ||F w - t||^2 + lam ||w||^2

Three candidates are compared on the *actual* problem this system solves:

1. **Ridge regression** (incumbent) — closed form, exact global optimum.
2. **Quantum linear regression** (HHL-family linear solvers).
3. **Quantum optimization** (variational minimisation of the same objective).

This module measures the problem's real dimensions and conditioning, then tests
whether a variational quantum optimiser can reach the closed-form optimum. HHL
is not simulated — it is analysed against its own published requirements, since
running it on a 5x5 system would prove nothing either way.

DATA PROVENANCE — synthetic, from the forward model in ``features.py``, using
the deployed calibration protocol (7 spots x 5 clicks).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from .gaze_letter import GEOM_IDX, make_intent_dataset

#: the deployed calibration protocol
CALIB_SPOTS = 7
CLICKS_PER_SPOT = 5
N_CALIB = CALIB_SPOTS * CLICKS_PER_SPOT      # 35 samples
N_COEFF = len(GEOM_IDX) + 1                  # 5 coefficients (bias + 4 features)


# ==========================================================================
# The real problem
# ==========================================================================

def calibration_problem(noise: float = 0.02, seed: int = 11,
                        collinear: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Design matrix ``F`` (n x 5) and targets ``t`` (n x 2) for one user.

    ``collinear=True`` reproduces the realistic degenerate case: a user who
    holds their head perfectly still, so the yaw/pitch columns carry almost no
    variance and the normal equations become ill-conditioned. This is the
    condition under which solver choice actually matters.
    """
    X, _y, pts = make_intent_dataset(N_CALIB, noise=noise, seed=seed)
    if collinear:
        X[:, 4] = X[:, 4].mean() + 1e-4 * np.random.default_rng(seed).normal(size=len(X))
        X[:, 5] = X[:, 5].mean() + 1e-4 * np.random.default_rng(seed + 1).normal(size=len(X))
    F = np.column_stack([np.ones(len(X)), X[:, list(GEOM_IDX)]])
    return F, pts


def ridge_exact(F: np.ndarray, t: np.ndarray, lam: float = 1e-3
                ) -> Tuple[np.ndarray, float, float]:
    """Closed-form ridge. Returns ``(w, objective, seconds)``."""
    t0 = time.perf_counter()
    A = F.T @ F + lam * np.eye(F.shape[1])
    w = np.linalg.solve(A, F.T @ t)
    dt = time.perf_counter() - t0
    obj = float(np.sum((F @ w - t) ** 2) + lam * np.sum(w ** 2))
    return w, obj, dt


def conditioning(F: np.ndarray, lam: float = 1e-3) -> Dict[str, float]:
    """Condition numbers with and without regularisation.

    ``kappa`` is the quantity every quantum linear solver's runtime depends on,
    so it is the number that decides whether such a solver could ever help.
    """
    A = F.T @ F
    return {"n_samples": int(F.shape[0]), "n_coefficients": int(F.shape[1]),
            "kappa_normal_equations": float(np.linalg.cond(A)),
            "kappa_ridge": float(np.linalg.cond(A + lam * np.eye(F.shape[1]))),
            "qubits_to_index_5x5": float(np.ceil(np.log2(F.shape[1])))}


# ==========================================================================
# Candidate 3 — quantum optimization of the same objective
# ==========================================================================

class VariationalLeastSquares:
    """Minimise ``||F w - t||^2 + lam ||w||^2`` with a variational circuit.

    The solution vector is amplitude-encoded in a 3-qubit register (8 amplitudes,
    5 used) and scaled by a learned norm, so ``w = r * psi(theta)``. This is the
    VQLS-style formulation (Bravo-Prieto et al., 2019) applied to the actual
    calibration objective.

    The measurement of interest is not "does it work" but "does it reach the
    closed-form optimum, and at what cost" — because the closed-form optimum is
    already available in microseconds.
    """

    def __init__(self, n_coeff: int = N_COEFF, steps: int = 300, lr: float = 0.06,
                 seed: int = 13) -> None:
        self.n_coeff = n_coeff
        self.n_qubits = int(np.ceil(np.log2(n_coeff)))
        self.steps, self.lr, self.seed = steps, lr, seed
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.n_layers = 3

        @qml.qnode(self.dev, interface="autograd")
        def state(theta):
            qml.StronglyEntanglingLayers(theta, wires=range(self.n_qubits))
            return qml.probs(wires=range(self.n_qubits))

        self._probs = state

        @qml.qnode(self.dev, interface="autograd")
        def amps(theta):
            qml.StronglyEntanglingLayers(theta, wires=range(self.n_qubits))
            return qml.state()

        self._state = amps

    def fit(self, F: np.ndarray, t: np.ndarray, lam: float = 1e-3
            ) -> Dict[str, object]:
        """Optimise one target column. Returns objective trace and timings."""
        F = pnp.array(np.asarray(F, dtype=float), requires_grad=False)
        t = pnp.array(np.asarray(t, dtype=float), requires_grad=False)
        rng = np.random.default_rng(self.seed)
        shape = qml.StronglyEntanglingLayers.shape(self.n_layers, self.n_qubits)
        theta = pnp.array(rng.normal(0, 0.3, size=shape), requires_grad=True)
        # scale factor: the circuit produces a unit vector, the solution is not
        r = pnp.array(float(np.linalg.norm(np.linalg.lstsq(
            np.asarray(F), np.asarray(t), rcond=None)[0])), requires_grad=True)

        def w_of(theta, r):
            psi = pnp.real(self._state(theta))[: self.n_coeff]
            return r * psi

        def cost(theta, r):
            w = w_of(theta, r)
            return pnp.sum((F @ w - t) ** 2) + lam * pnp.sum(w ** 2)

        opt = qml.AdamOptimizer(self.lr)
        trace: List[float] = []
        t0 = time.perf_counter()
        for _ in range(self.steps):
            theta, r = opt.step(cost, theta, r)
            trace.append(float(cost(theta, r)))
        dt = time.perf_counter() - t0
        return {"w": np.asarray(w_of(theta, r), dtype=float),
                "objective": trace[-1], "trace": trace, "seconds": dt,
                "steps": self.steps}


# ==========================================================================
# Study
# ==========================================================================

def run(verbose: bool = True) -> Dict[str, object]:
    out: Dict[str, object] = {}

    for label, collinear in (("well-conditioned", False), ("collinear head pose", True)):
        F, t = calibration_problem(collinear=collinear)
        cond = conditioning(F)
        w_ex, obj_ex, sec_ex = ridge_exact(F, t)

        # exact solve, repeated, to get a stable timing
        reps = 2000
        t0 = time.perf_counter()
        for _ in range(reps):
            ridge_exact(F, t)
        per_solve_us = (time.perf_counter() - t0) / reps * 1e6

        vqls = VariationalLeastSquares()
        vq = vqls.fit(F, t[:, 0])
        obj_ex_x = float(np.sum((F @ w_ex[:, 0] - t[:, 0]) ** 2)
                         + 1e-3 * np.sum(w_ex[:, 0] ** 2))

        rec = {
            **cond,
            "exact_objective_x": obj_ex_x,
            "exact_solve_us": per_solve_us,
            "vqls_objective_x": vq["objective"],
            "vqls_seconds": vq["seconds"],
            "vqls_steps": vq["steps"],
            "gap_percent": 100.0 * (vq["objective"] - obj_ex_x) / max(obj_ex_x, 1e-12),
            "slowdown_x": vq["seconds"] * 1e6 / per_solve_us,
        }
        out[label] = rec

        if verbose:
            print("\n--- %s ---" % label)
            print("  problem size          %d samples x %d coefficients"
                  % (rec["n_samples"], rec["n_coefficients"]))
            print("  kappa(F^T F)          %.3e" % rec["kappa_normal_equations"])
            print("  kappa(F^T F + lam I)  %.3e   <- ridge fixes conditioning"
                  % rec["kappa_ridge"])
            print("  exact ridge solve     %.1f us   objective %.6e"
                  % (rec["exact_solve_us"], rec["exact_objective_x"]))
            print("  variational (VQLS)    %.1f s    objective %.6e"
                  % (rec["vqls_seconds"], rec["vqls_objective_x"]))
            print("  gap to optimum        %+.2f%%      slowdown %.0fx"
                  % (rec["gap_percent"], rec["slowdown_x"]))
    return out


if __name__ == "__main__":
    import json
    import warnings
    warnings.filterwarnings("ignore")
    np.seterr(all="ignore")
    print("calibration solver study — raw gaze features -> screen coordinates")
    res = run()
    with open("quantum/calibration_study.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\nresults -> quantum/calibration_study.json")
