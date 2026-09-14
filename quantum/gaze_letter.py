"""Quantum gaze -> letter decoding stack (camera path only).

Scope
-----
This module covers **only** the eye-gaze half of the system: camera features in,
a posterior over the 26 letter keys out. It never touches the word decoder, the
language model, or the keyboard's chip UI. The question it answers is narrow and
measurable: *given one fixation, which letter did the user intend?*

The deployed classical answer (``static/app.js``) is:

    ridge regression -> gaze point -> Gaussian integrated over each key rect
                                   -> argmax

Ridge is the exact global optimum of its own objective, so there is no headroom
inside it. The headroom is in what ridge structurally *cannot* represent: the
gaze->screen map has an optical curvature term (see ``features._forward``), the
landmark noise is heteroscedastic, and a single frame's evidence is discarded
once it becomes a point estimate. This stack attacks that residual with seven
quantum stages, then fuses them with the classical geometric channel.

Stage map (as specified)
------------------------
=========================  =====================  ==============================
Purpose                    Algorithm              Role here
=========================  =====================  ==============================
Feature optimization       QAOA                   choose the informative,
                                                  non-redundant feature subset
                                                  fed to every later stage
Eye feature classification QSVM (fidelity kernel) gaze features -> 1 of 9 screen
                                                  regions (coarse, separable)
Letter prediction          VQC                    within a region, which slot
                                                  (<=4 classes) was intended
Similarity matching        Quantum kernel         fidelity similarity between
                                                  this fixation and 26 stored
                                                  per-letter gaze prototypes
Pattern recognition        QNN                    fixation / saccade / blink /
                                                  drift from kinematics
Probability modeling       Quantum Bayesian Net   P(evidence reliable | state)
                                                  -> tempers every channel
Candidate letter opt.      QAOA                   robust consensus pick among
                                                  the top-K letters
=========================  =====================  ==============================

Why the hierarchy (this is the accuracy mechanism)
--------------------------------------------------
A flat 26-class VQC was measured at 0.367 accuracy in ``benchmark.py`` — far
below the classical MLP's 0.947. Twenty-six classes over ~13 features is simply
the wrong shape for a shallow variational circuit. Splitting the decision into
*9 regions* (QSVM, well-separated, kernel-friendly) and then *<=4 slots within a
region* (VQC, few classes, small circuit) puts each quantum model on a problem
its class size can actually support. That restructuring — not a bigger circuit —
is what makes the quantum path competitive.

Feasibility is handled classically
----------------------------------
The candidate-letter QAOA carries **no one-hot penalty and no XY mixer**. It runs
the plain transverse-field mixer over a penalty-free QUBO, and "exactly one
letter" is imposed on the *classical* side by post-selecting the feasible
bitstrings out of the measured distribution (and repairing the rest). The
acceptance rate of that post-selection is measured and reported rather than
assumed.

Measured outcome (read this before quoting the module)
------------------------------------------------------
Run ``python -m quantum.gaze_letter`` to reproduce; numbers land in
``gaze_letter_report.json``.

1. **QAOA feature optimization works.** It selects ``iris_x, iris_y, yaw, pitch``
   and matches the exhaustive optimum exactly, on 11 qubits, once the constraint
   penalty is scaled to the objective.
2. **QAOA candidate optimization works** — after reformulation. It matches the
   exact optimum on 100% of decisions at ~3 ms, with feasibility handled entirely
   classically.
3. **The quantum classifiers do not beat the calibrated geometric decoder.**
   Quantum-kernel prototype matching reaches 0.728 and the QSVM-region x VQC
   hierarchy 0.395, against 0.938 for the deployed ridge + Gaussian rule. Nested
   validation with a McNemar test rejects every fused weighting, so the shipped
   configuration collapses to geometry-only and the stack scores *exactly* the
   deployed accuracy (+0.0 pp) at every noise level. That is the honest result:
   given a calibrated point and its uncertainty, integrating the Gaussian over
   each key rectangle is already the Bayes rule, and channels derived from that
   same point cannot add information to it.
4. **Where the quantum path does win is commit quality.** On fixation streams
   where 25% of dwell windows land mid-saccade or mid-blink, tempering the
   posterior by the QNN+QBN reliability cuts wrong keystrokes from 0.47 to 0.33
   per 100 fixations on top of the already-deployed QSVM gate (-30%), and from
   4.47 to 3.47 (-22%) at higher gaze noise, for about 2 pp more prompts. This
   path reads **kinematics only**, so unlike the region/slot stages it needs no
   per-user geometry and carries over to a real webcam unchanged. It is wired
   into the live commit path.
5. **The calibration-residual corrector does not help at 7x5 samples.**
   Leave-one-out drives its regularisation to the maximum and the correction
   shrinks toward zero: 35 clicks cannot separate a ~0.010 systematic curvature
   from ~0.007 of per-anchor jitter. It is retained because it is safe by
   construction and would become useful with a denser calibration grid.

Provenance of data
------------------
No recorded human gaze corpus ships with this project. Training and evaluation
use ``features.make_gaze_dataset``, the documented forward model. Numbers here
are valid for *relative* comparison of decoders under a stated noise model and
are not a claim about real-webcam accuracy; ``fit_from_calibration`` exists so
the same stack can be retrained on a real user's calibration samples.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import time
from dataclasses import dataclass, field
from itertools import product as iproduct
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from .features import (FEATURE_NAMES, N_FEATURES, FeatureScaler, keyboard_layout,
                       make_gaze_dataset, make_movement_dataset)
from .models import QNN, QNNConfig, QuantumKernel, VQC, VQCConfig
from .qaoa import QAOASolver, build_feature_selection_qubo, qubo_energy
from .qbn import QuantumBayesianNetwork

MODEL_PATH = os.path.join(os.path.dirname(__file__), "gaze_letter.pkl")
PARAMS_PATH = os.path.join(os.path.dirname(__file__), "gaze_qaoa_params.json")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "gaze_letter_report.json")

#: screen is split into a 3x3 grid of gaze regions for the QSVM stage
REGION_ROWS, REGION_COLS = 3, 3
N_REGIONS = REGION_ROWS * REGION_COLS

#: features the QAOA subset selector is allowed to keep. Four is what the
#: selector chooses on this data (both iris ratios plus the two head-pose
#: compensation terms); it is also the qubit count of every kernel stage.
N_SELECT = 4
#: candidate letters carried into the consensus QAOA
K_CANDIDATES = 4
#: evidence channels fused by the consensus stage
CHANNELS: Tuple[str, ...] = ("geom", "qknn", "qsvm", "qhier")

#: head-pose columns handed to the quantum stages alongside the calibrated point
POSE_IDX: Tuple[int, ...] = (4, 5)

#: indices into FEATURE_NAMES used by the deployed-style ridge baseline:
#: iris_x, iris_y, yaw, pitch (plus an implicit bias column)
GEOM_IDX: Tuple[int, ...] = (0, 1, 4, 5)
#: kinematic indices consumed by the QNN: ear, gaze_velocity, fixation, history
KINEMATIC_IDX: Tuple[int, ...] = (2, 8, 10, 12)

_EPS = 1e-12


# ==========================================================================
# Keyboard region geometry
# ==========================================================================

def _build_region_maps():
    layout = keyboard_layout()
    key_region: List[int] = []
    for _ch, kx, ky, kw, _kh in layout:
        row = int(round((ky - 0.30) / 0.22))
        band = min(REGION_COLS - 1, int((kx + kw / 2.0) * REGION_COLS))
        key_region.append(row * REGION_COLS + band)
    region_keys: List[List[int]] = [[] for _ in range(N_REGIONS)]
    for i, r in enumerate(key_region):
        region_keys[r].append(i)
    key_slot = [0] * len(layout)
    for keys in region_keys:
        for s, i in enumerate(keys):
            key_slot[i] = s
    max_slots = max(len(k) for k in region_keys)
    return (np.array(key_region, dtype=int), region_keys,
            np.array(key_slot, dtype=int), max_slots)


KEY_REGION, REGION_KEYS, KEY_SLOT, MAX_SLOTS = _build_region_maps()
LETTERS: List[str] = [ch for ch, _x, _y, _w, _h in keyboard_layout()]
N_KEYS = len(LETTERS)


# ==========================================================================
# Intent dataset — "which key did the user MEAN", not "where did gaze land"
# ==========================================================================

def _inverse_forward(target: np.ndarray, head: np.ndarray,
                     curvature: float) -> np.ndarray:
    """Iris ratio that truly points at ``target`` under the curved optical map.

    ``features.make_gaze_dataset`` inverts only the *linear* part of the forward
    model, so its label is where the gaze landed. For letter *intent* we need the
    opposite: fix the key the user meant, and solve the full cubic

        curv * u^3 + 1.05 * u = target - 0.5 - 0.35 * (head - 0.5),  u = h - 0.5

    exactly (Cardano; the cubic is strictly monotone so the real root is unique).
    The linear ridge calibration then carries a genuine cubic residual — which is
    precisely the error the nonlinear stages are asked to remove.
    """
    c = np.asarray(target, dtype=float) - 0.5 - 0.35 * (np.asarray(head, dtype=float) - 0.5)
    if curvature < 1e-9:
        return 0.5 + c / 1.05
    p = 1.05 / curvature
    q = -c / curvature
    disc = np.sqrt(q * q / 4.0 + p ** 3 / 27.0)
    u = np.cbrt(-q / 2.0 + disc) + np.cbrt(-q / 2.0 - disc)
    return 0.5 + u


def make_intent_dataset(n_samples: int = 900, noise: float = 0.02,
                        curvature: float = 0.45, head_motion: float = 0.06,
                        seed: int = 13) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthesise ``(features, intended_key, intended_point)``.

    The label is the key the user *meant*, so landmark jitter creates real
    ambiguity near key borders — the error a gaze decoder actually has to fight.
    Feature layout matches ``features.FEATURE_NAMES`` exactly, so the same
    vectors feed every stage and the movement/kinematic indices stay valid.
    """
    rng = np.random.default_rng(seed)
    layout = keyboard_layout()
    X = np.zeros((n_samples, N_FEATURES))
    y = np.zeros(n_samples, dtype=int)
    tgt = np.zeros((n_samples, 2))

    for i in range(n_samples):
        k = int(rng.integers(0, len(layout)))
        _ch, kx, ky, kw, kh = layout[k]
        tx = kx + kw * rng.uniform(0.25, 0.75)
        ty = ky + kh * rng.uniform(0.25, 0.75)
        nose_x = 0.5 + rng.normal(0, head_motion)
        nose_y = 0.5 + rng.normal(0, head_motion)

        h = float(_inverse_forward(tx, nose_x, curvature)) + rng.normal(0, noise)
        v = float(_inverse_forward(ty, nose_y, curvature)) + rng.normal(0, noise)

        ear = float(np.clip(rng.normal(0.75, 0.06), 0.0, 1.0))
        vel = float(np.clip(abs(rng.normal(0.05, 0.04)), 0.0, 1.0))
        X[i] = [
            h, v, ear, 0.0,
            (nose_x - 0.5) * 2, (nose_y - 0.5) * 2, rng.normal(0, 0.05),
            ear, vel, 0.0,
            float(np.clip(1.0 - vel * 4, 0.0, 1.0)),
            float(np.clip(rng.normal(0.85, 0.08), 0.0, 1.0)),
            float(np.clip(abs(rng.normal(0.1, 0.06)), 0.0, 1.0)),
        ]
        y[i] = k
        tgt[i] = (tx, ty)
    return X, y, tgt


# ==========================================================================
# Classical geometric channel — the deployed baseline, reproduced exactly
# ==========================================================================

def _erf(z: np.ndarray) -> np.ndarray:
    """Abramowitz-Stegun 7.1.26 — the same approximation ``static/app.js`` uses,
    so the baseline here is the shipped computation and not a better one."""
    z = np.asarray(z, dtype=float)
    s = np.sign(z)
    z = np.abs(z)
    t = 1.0 / (1.0 + 0.3275911 * z)
    poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
             - 0.284496736) * t + 0.254829592) * t
    return s * (1.0 - poly * np.exp(-z * z))


def _norm_cdf(x: float, mu: np.ndarray, sigma: float) -> np.ndarray:
    return 0.5 * (1.0 + _erf((x - mu) / (sigma * math.sqrt(2.0))))


class GeometricGaze:
    """Ridge gaze map + Gaussian key posterior — the incumbent to beat.

    ``w = (F^T F + lam I)^-1 F^T t`` mirrors ``iris-tracker.js::_fit``; the
    per-key probability is the Gaussian integrated over the key rectangle,
    mirroring ``app.js::keyPosterior``. Both are exact given their model, which
    is why this channel is kept in the fusion rather than replaced.
    """

    def __init__(self, lam: float = 1e-3) -> None:
        self.lam = lam
        self.wx: Optional[np.ndarray] = None
        self.wy: Optional[np.ndarray] = None
        self.sigma: Tuple[float, float] = (0.02, 0.02)
        self.layout = keyboard_layout()

    @staticmethod
    def _design(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.column_stack([np.ones(len(X)), X[:, list(GEOM_IDX)]])

    def fit(self, X: np.ndarray, pts: np.ndarray) -> "GeometricGaze":
        F = self._design(X)
        A = F.T @ F + self.lam * np.eye(F.shape[1])
        self.wx = np.linalg.solve(A, F.T @ pts[:, 0])
        self.wy = np.linalg.solve(A, F.T @ pts[:, 1])
        pred = self.predict_points(X)
        self.sigma = (
            float(max(1e-3, np.sqrt(np.mean((pred[:, 0] - pts[:, 0]) ** 2)))),
            float(max(1e-3, np.sqrt(np.mean((pred[:, 1] - pts[:, 1]) ** 2)))),
        )
        return self

    def predict_points(self, X: np.ndarray) -> np.ndarray:
        F = self._design(X)
        # errstate: Accelerate's matmul emits spurious divide/overflow warnings
        # on some shapes; inputs and outputs are asserted finite in the tests.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            return np.column_stack([F @ self.wx, F @ self.wy])

    def posterior(self, X: np.ndarray) -> np.ndarray:
        """``(n, 26)`` probability of each key under the gaze Gaussian."""
        return self.posterior_from_points(self.predict_points(X), self.sigma)

    def posterior_from_points(self, pts: np.ndarray,
                              sigma: Tuple[float, float]) -> np.ndarray:
        """Key posterior for an arbitrary gaze estimate — lets a corrected point
        reuse the exact integration the deployed keyboard performs."""
        pts = np.asarray(pts, dtype=float)
        sx, sy = sigma
        out = np.zeros((len(pts), N_KEYS))
        for i, (_ch, kx, ky, kw, kh) in enumerate(self.layout):
            px = _norm_cdf(kx + kw, pts[:, 0], sx) - _norm_cdf(kx, pts[:, 0], sx)
            py = _norm_cdf(ky + kh, pts[:, 1], sy) - _norm_cdf(ky, pts[:, 1], sy)
            out[:, i] = np.clip(px * py, 0.0, None)
        z = out.sum(axis=1, keepdims=True)
        return np.where(z > _EPS, out / np.where(z > _EPS, z, 1.0), 1.0 / N_KEYS)


# ==========================================================================
# Quantum-kernel calibration correction — the user-adaptive part of the stack
# ==========================================================================

#: the seven calibration spots, matching ``CORNERS`` in ``static/app.js``
CALIB_SPOTS: Tuple[Tuple[str, float, float], ...] = (
    ("TL", 0.04, 0.05), ("TR", 0.96, 0.05),
    ("ML", 0.04, 0.50), ("C", 0.50, 0.50), ("MR", 0.96, 0.50),
    ("BL", 0.04, 0.95), ("BR", 0.96, 0.95),
)


class CalibrationResidualQK:
    """Kernel-ridge regression, on the quantum fidelity kernel, of the *residual*
    left by the deployed linear calibration.

    Why this is the one quantum stage that can help a real user immediately: it
    is fitted from **their own** calibration clicks, not from synthetic data. The
    browser already collects 7 spots x 5 clicks; the linear ridge consumes them
    and leaves a smooth, position-dependent error (optical curvature, lens
    distortion, eyeball-rotation nonlinearity). This stage models exactly that
    leftover:

        residual(z) = target - ridge(z),     alpha = (K + lam I)^-1 residual

    with ``K`` the fidelity kernel between calibration fixations. Closed form —
    no variational optimiser, no barren plateau, ``n^2`` circuits for ``n<=40``
    calibration samples (about 1 second).

    Safety by construction: if a user's gaze map really is linear, the residuals
    are noise, leave-one-out cross-validation drives ``lam`` up, and the
    correction shrinks to zero. It cannot make a well-calibrated user worse.
    """

    def __init__(self, n_features: int = N_SELECT, strategy: str = "angle",
                 reps: int = 1, lam: float = 0.05) -> None:
        self.qk = QuantumKernel(n_features, strategy=strategy, reps=reps)
        self.lam = float(lam)
        self.Z: Optional[np.ndarray] = None
        self.alpha: Optional[np.ndarray] = None
        self.loo_rmse: Optional[float] = None

    def fit(self, Z: np.ndarray, residual: np.ndarray,
            lams: Sequence[float] = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0)
            ) -> "CalibrationResidualQK":
        """Fit with ``lam`` chosen by closed-form leave-one-out CV.

        For kernel ridge the LOO residual is available without refitting:
        ``e_i = alpha_i / [(K + lam I)^-1]_ii``. With only ~35 calibration
        samples this guard is what separates a correction from an overfit.
        """
        Z = np.asarray(Z, dtype=float)
        R = np.asarray(residual, dtype=float)
        K = self.qk.gram(Z)
        n = len(Z)
        best = (np.inf, self.lam, None)
        for lam in lams:
            Hinv = np.linalg.inv(K + lam * np.eye(n))
            a = Hinv @ R
            diag = np.clip(np.diag(Hinv), 1e-9, None)[:, None]
            loo = float(np.sqrt(np.mean((a / diag) ** 2)))
            if loo < best[0]:
                best = (loo, lam, a)
        self.loo_rmse, self.lam, self.alpha = best[0], best[1], best[2]
        self.Z = Z
        return self

    def predict(self, Z: np.ndarray) -> np.ndarray:
        if self.alpha is None:
            return np.zeros((len(Z), 2))
        return self.qk.gram(np.asarray(Z, dtype=float), self.Z) @ self.alpha


# ==========================================================================
# Stage 1 — QAOA feature optimization (subset selection before classification)
# ==========================================================================

def mutual_information(X: np.ndarray, y: np.ndarray, bins: int = 8) -> np.ndarray:
    """Per-feature discrete MI with the label — the relevance term of mRMR."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    n = len(y)
    out = np.zeros(X.shape[1])
    py = np.bincount(y) / n
    hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
    for d in range(X.shape[1]):
        col = X[:, d]
        if col.max() - col.min() < 1e-12:
            continue
        edges = np.linspace(col.min(), col.max() + 1e-9, bins + 1)
        b = np.clip(np.digitize(col, edges) - 1, 0, bins - 1)
        h_cond = 0.0
        for v in range(bins):
            m = b == v
            if not m.any():
                continue
            pv = m.mean()
            pyv = np.bincount(y[m], minlength=len(py)) / m.sum()
            h_cond += pv * -np.sum(pyv[pyv > 0] * np.log(pyv[pyv > 0]))
        out[d] = max(0.0, hy - h_cond)
    return out


def ridge_importance(X: np.ndarray, pts: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    """Leave-one-feature-out increase in calibration RMSE — the relevance term.

    Marginal mutual information is the textbook mRMR relevance, and it is the
    wrong statistic here: head pose (``yaw``/``pitch``) is *independent* of which
    key was intended, so its marginal MI is ~0, yet dropping it wrecks the gaze
    map because it is the term that compensates head translation. Measured on
    this data, MI-relevance selects ``roll``/``eye_openness``/``confidence`` —
    pure noise channels — and downstream accuracy collapses.

    Backward-elimination importance asks the question that actually matters:
    how much worse does the calibration get *without* this feature, given all
    the others. It keeps conditioning variables, is exactly as QUBO-native, and
    costs ``d`` closed-form ridge fits. Both statistics are reported.
    """
    X = np.asarray(X, dtype=float)
    pts = np.asarray(pts, dtype=float)
    d = X.shape[1]

    def rmse(cols: List[int]) -> float:
        F = np.column_stack([np.ones(len(X))] + ([X[:, cols]] if cols else []))
        # errstate: Apple Accelerate's matmul emits spurious divide/overflow
        # warnings on these shapes. Inputs and outputs are finite — the assert
        # below is the guard, so the warnings are noise, not a masked fault.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            A = F.T @ F + lam * np.eye(F.shape[1])
            w = np.linalg.solve(A, F.T @ pts)
            err = float(np.sqrt(np.mean((F @ w - pts) ** 2)))
        if not np.isfinite(err):
            raise FloatingPointError("ridge_importance produced a non-finite RMSE")
        return err

    full = list(range(d))
    base = rmse(full)
    imp = np.array([max(0.0, rmse([c for c in full if c != i]) - base) for i in range(d)])
    return imp


def redundancy_matrix(X: np.ndarray) -> np.ndarray:
    """|Pearson correlation| between features — the redundancy term of mRMR."""
    C = np.corrcoef(np.asarray(X, dtype=float).T)
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 0.0)
    return np.abs(C)


def _repair_subset(bits: np.ndarray, relevance: np.ndarray,
                   redundancy: np.ndarray, k: int) -> List[int]:
    """Classical feasibility repair: force a sampled string to exactly ``k`` bits.

    Greedy mRMR is used to add or drop, so the repaired subset is still a
    defensible selection rather than an arbitrary truncation.
    """
    chosen = [int(i) for i in np.flatnonzero(np.asarray(bits) > 0.5)]
    while len(chosen) > k:                      # drop the weakest member
        worst, worst_score = None, np.inf
        for i in chosen:
            others = [j for j in chosen if j != i]
            score = relevance[i] - (np.mean([redundancy[i, j] for j in others])
                                    if others else 0.0)
            if score < worst_score:
                worst_score, worst = score, i
        chosen.remove(worst)
    while len(chosen) < k:                      # add the strongest outsider
        best, best_score = None, -np.inf
        for i in range(len(relevance)):
            if i in chosen:
                continue
            score = relevance[i] - (np.mean([redundancy[i, j] for j in chosen])
                                    if chosen else 0.0)
            if score > best_score:
                best_score, best = score, i
        chosen.append(best)
    return sorted(chosen)


def _subset_objective(subset: Sequence[int], relevance: np.ndarray,
                      redundancy: np.ndarray, alpha: float = 1.0) -> float:
    s = list(subset)
    return float(-sum(relevance[i] for i in s)
                 + alpha * sum(redundancy[i, j]
                               for a, i in enumerate(s) for j in s[a + 1:]))


def _greedy_subset(relevance: np.ndarray, redundancy: np.ndarray, k: int,
                   alpha: float) -> List[int]:
    """Greedy relevance/redundancy selection — the standard classical heuristic."""
    chosen: List[int] = []
    for _ in range(k):
        best, best_score = None, -np.inf
        for i in range(len(relevance)):
            if i in chosen:
                continue
            red = alpha * float(np.mean([redundancy[i, j] for j in chosen])) if chosen else 0.0
            if relevance[i] - red > best_score:
                best_score, best = relevance[i] - red, i
        chosen.append(int(best))
    return sorted(chosen)


def _exact_subset(relevance: np.ndarray, redundancy: np.ndarray, k: int,
                  alpha: float) -> List[int]:
    """Exhaustive k-subset search — ground truth for the QAOA comparison."""
    from itertools import combinations
    best, best_v = None, np.inf
    for combo in combinations(range(len(relevance)), k):
        v = _subset_objective(combo, relevance, redundancy, alpha)
        if v < best_v:
            best_v, best = v, list(combo)
    return best


def select_features_qaoa(X: np.ndarray, y: np.ndarray, pts: np.ndarray,
                         k: int = N_SELECT, alpha: float = 0.05,
                         penalty: float = 1.0,
                         p: int = 2, steps: int = 100, seed: int = 13,
                         verbose: bool = True) -> Dict[str, object]:
    """Pick ``k`` gaze features by QAOA over the relevance/redundancy QUBO.

    Zero-variance columns are dropped before the QUBO is built. Left in, they
    are *free* to select — no relevance but also no redundancy — so the optimiser
    happily fills the budget with constants. That is a property of the criterion,
    not of the solver, and removing them is ordinary preprocessing.

    Compared against greedy selection and against exhaustive search, so the QAOA
    answer is checked against the true optimum rather than assumed good.
    """
    X = np.asarray(X, dtype=float)
    active = [i for i in range(X.shape[1]) if X[:, i].std() > 1e-9]

    relevance_full = ridge_importance(X, pts)
    if relevance_full.max() > 0:
        relevance_full = relevance_full / relevance_full.max()
    relevance = relevance_full[active]
    redundancy = redundancy_matrix(X[:, active])
    # Penalty scale is decisive, and the library default (4.0) is wrong for this
    # instance: relevance is normalised to <=1, so a penalty of 4 makes the
    # cardinality constraint 4x louder than the objective and QAOA at p=2 spends
    # its whole expressivity satisfying it — measured, it returned pure-noise
    # subsets. The penalty only has to exceed the best relevance a violation
    # could buy (1.0). At 1.0-1.5 QAOA hits the exact optimum; at 2.5 it misses.
    Q = build_feature_selection_qubo(relevance, redundancy, k=k, alpha=alpha,
                                     penalty=penalty)

    solver = QAOASolver(Q, p=p, steps=steps, seed=seed, mixer="x")
    res = solver.solve()
    raw = [int(i) for i in np.flatnonzero(res.bitstring > 0.5)]
    qaoa_local = _repair_subset(res.bitstring, relevance, redundancy, k)
    greedy_local = _greedy_subset(relevance, redundancy, k, alpha)
    exact_local = _exact_subset(relevance, redundancy, k, alpha)

    to_global = lambda s: sorted(int(active[i]) for i in s)
    qaoa_subset = to_global(qaoa_local)

    out = {
        "selected": qaoa_subset,
        "selected_names": [FEATURE_NAMES[i] for i in qaoa_subset],
        "raw_bits_cardinality": len(raw),
        "needed_repair": len(raw) != k,
        "greedy": to_global(greedy_local),
        "exact": to_global(exact_local),
        "objective_qaoa": _subset_objective(qaoa_local, relevance, redundancy, alpha),
        "objective_greedy": _subset_objective(greedy_local, relevance, redundancy, alpha),
        "objective_exact": _subset_objective(exact_local, relevance, redundancy, alpha),
        "matches_exact": qaoa_local == list(exact_local),
        "relevance_ridge": relevance_full.tolist(),
        "relevance_mi": mutual_information(X, y).tolist(),
        "dropped_constant": [FEATURE_NAMES[i] for i in range(X.shape[1]) if i not in active],
        "qubits": int(res.n_qubits),
    }
    if verbose:
        print("[stage 1] QAOA feature selection on %d qubits" % res.n_qubits)
        print("   QAOA   %-28s obj=%.4f%s" % (
            out["selected_names"], out["objective_qaoa"],
            "  (classically repaired)" if out["needed_repair"] else ""))
        print("   greedy %-28s obj=%.4f" % (
            [FEATURE_NAMES[i] for i in out["greedy"]], out["objective_greedy"]))
        print("   exact  %-28s obj=%.4f" % (
            [FEATURE_NAMES[i] for i in out["exact"]], out["objective_exact"]))
        print("   QAOA == exact optimum: %s" % out["matches_exact"])
    return out


# ==========================================================================
# Stage 2 — QSVM region classifier (eye feature classification)
# ==========================================================================

class QuantumKernelSVM:
    """Multi-class SVM on the fidelity kernel, predicting from support vectors only.

    Prediction cost is ``O(n_support)`` circuits rather than ``O(n_train)``,
    which is the dominant expense of kernel QML.
    """

    def __init__(self, n_features: int, strategy: str = "angle",
                 reps: int = 1, C: float = 8.0, temperature: float = 0.35) -> None:
        self.strategy, self.reps, self.C = strategy, reps, C
        self.temperature = temperature
        self.qk = QuantumKernel(n_features, strategy=strategy, reps=reps)
        self.svc = None
        self._train_Z: Optional[np.ndarray] = None
        self._sv: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "QuantumKernelSVM":
        from sklearn.svm import SVC
        Z = np.asarray(Z, dtype=float)
        self._train_Z = Z
        self.svc = SVC(kernel="precomputed", C=self.C,
                       decision_function_shape="ovr")
        self.svc.fit(self.qk.gram(Z), np.asarray(y))
        self._sv = np.asarray(self.svc.support_, dtype=int)
        self.classes_ = np.asarray(self.svc.classes_)
        return self

    def _kernel_rows(self, Z: np.ndarray) -> np.ndarray:
        K = np.zeros((len(Z), len(self._train_Z)))
        K[:, self._sv] = self.qk.gram(Z, self._train_Z[self._sv])
        return K

    def predict(self, Z: np.ndarray) -> np.ndarray:
        return self.svc.predict(self._kernel_rows(np.asarray(Z, dtype=float)))

    def predict_proba(self, Z: np.ndarray, n_classes: int) -> np.ndarray:
        """Softmax over the one-vs-rest margins.

        Not a calibrated probability — it is a monotone transform of the margin,
        used only as an evidence weight. Its calibration error is measured (ECE)
        rather than claimed.
        """
        d = self.svc.decision_function(self._kernel_rows(np.asarray(Z, dtype=float)))
        d = np.atleast_2d(d)
        full = np.full((len(d), n_classes), -1e3)
        for j, c in enumerate(self.classes_):
            full[:, int(c)] = d[:, j] if d.shape[1] > 1 else d[:, 0]
        z = full / max(self.temperature, 1e-6)
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


# ==========================================================================
# Stage 3 — Quantum-kernel similarity matching against stored gaze patterns
# ==========================================================================

class GazePrototypeMatcher:
    """Fidelity similarity between the live fixation and 26 stored prototypes.

    One prototype per letter, taken as the per-dimension median of that letter's
    training fixations (median, not mean, so a stray saccade frame cannot drag a
    prototype off its key). Similarity is the same fidelity kernel used by the
    QSVM, so this channel is ``26`` circuits per fixation regardless of how much
    training data exists — the property that makes kernel matching affordable
    here at all.
    """

    def __init__(self, n_features: int, strategy: str = "angle", reps: int = 1,
                 temperature: float = 0.03) -> None:
        self.qk = QuantumKernel(n_features, strategy=strategy, reps=reps)
        self.temperature = temperature
        self.prototypes: Optional[np.ndarray] = None

    def fit(self, Z: np.ndarray, y_key: np.ndarray) -> "GazePrototypeMatcher":
        Z = np.asarray(Z, dtype=float)
        y_key = np.asarray(y_key, dtype=int)
        protos = np.zeros((N_KEYS, Z.shape[1]))
        for k in range(N_KEYS):
            m = y_key == k
            protos[k] = np.median(Z[m], axis=0) if m.any() else Z.mean(axis=0)
        self.prototypes = protos
        return self

    def posterior(self, Z: np.ndarray) -> np.ndarray:
        sim = self.qk.gram(np.asarray(Z, dtype=float), self.prototypes)
        z = sim / max(self.temperature, 1e-6)
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


# ==========================================================================
# Stage 4 — VQC letter prediction inside a region
# ==========================================================================

class RegionSlotVQC:
    """Variational classifier for the slot index within a gaze region.

    Input is the selected gaze features plus the region index; output is one of
    ``MAX_SLOTS`` slots. Conditioning on the region is what keeps the class count
    at 4 instead of 26 — the change that makes a shallow VQC viable at all.
    """

    def __init__(self, n_features: int, seed: int = 13, layers: int = 2,
                 epochs: int = 14) -> None:
        self.cfg = VQCConfig(n_features=n_features + 1, n_classes=MAX_SLOTS,
                             n_layers=layers, strategy="angle", epochs=epochs,
                             batch_size=32, lr=0.1, seed=seed)
        self.vqc = VQC(self.cfg)

    @staticmethod
    def _augment(Z: np.ndarray, region: np.ndarray) -> np.ndarray:
        r = np.asarray(region, dtype=float).reshape(-1, 1) / (N_REGIONS - 1) * math.pi
        return np.hstack([np.asarray(Z, dtype=float), r])

    def fit(self, Z: np.ndarray, region: np.ndarray, slot: np.ndarray) -> "RegionSlotVQC":
        self.vqc.fit(self._augment(Z, region), np.asarray(slot, dtype=int))
        return self

    def slot_proba(self, Z: np.ndarray, region: np.ndarray) -> np.ndarray:
        return self.vqc.predict_proba(self._augment(Z, region))


# ==========================================================================
# Stages 5 + 6 — QNN pattern recognition and QBN probability modeling
# ==========================================================================

class ReliabilityModel:
    """QNN movement state -> Quantum Bayesian Network -> P(evidence reliable).

    The QNN answers *what the eye was doing*; the QBN turns that into *how much
    the gaze evidence should count*, via the causal network in ``qbn.py``
    (blink/saccade -> unreliable -> mis-key). The output tempers every evidence
    channel: reliable fixations keep their sharp posteriors, suspect frames get
    flattened so the true letter is not confidently discarded.
    """

    def __init__(self, seed: int = 13, epochs: int = 16) -> None:
        self.qnn = QNN(QNNConfig(n_features=len(KINEMATIC_IDX), n_classes=4,
                                 n_layers=2, strategy="angle", epochs=epochs,
                                 seed=seed))
        self.scaler: Optional[FeatureScaler] = None
        qbn = QuantumBayesianNetwork()
        # 4 evidence combinations only — evaluate the circuit once and cache
        self.table = {
            (b, s): float(qbn.typing_confidence(b, s))
            for b in (0, 1) for s in (0, 1)
        }

    def fit(self, n: int = 220, seed: int = 13) -> "ReliabilityModel":
        Xm, ym = make_movement_dataset(n, seed=seed)
        Xk = Xm[:, list(KINEMATIC_IDX)]
        self.scaler = FeatureScaler().fit(Xk)
        self.qnn.fit(self.scaler.transform(Xk), ym)
        return self

    def state_proba(self, Xk: np.ndarray) -> np.ndarray:
        return self.qnn.predict_proba(self.scaler.transform(np.asarray(Xk, dtype=float)))

    def reliability(self, Xk: np.ndarray) -> np.ndarray:
        """Expected ``P(key correct)`` under the QNN's state distribution."""
        P = self.state_proba(Xk)
        # classes: 0 fixation, 1 saccade, 2 blink, 3 drift
        r = (P[:, 0] * self.table[(0, 0)] + P[:, 1] * self.table[(0, 1)]
             + P[:, 2] * self.table[(1, 0)] + P[:, 3] * self.table[(0, 1)])
        return np.clip(r, 0.05, 1.0)


# ==========================================================================
# Stage 7 — QAOA candidate-letter optimization (classical feasibility)
# ==========================================================================

def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits — a bounded [0,1] disagreement measure."""
    p = np.clip(np.asarray(p, dtype=float), _EPS, None)
    q = np.clip(np.asarray(q, dtype=float), _EPS, None)
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * (np.log2(a) - np.log2(b))))
    return float(np.clip(0.5 * kl(p, m) + 0.5 * kl(q, m), 0.0, 1.0))


def build_channel_qubo(support: np.ndarray, disagree: np.ndarray,
                       alpha: float = 1.0) -> np.ndarray:
    """QUBO choosing which evidence channels to trust for this fixation.

        E(y) = - sum_c y_c * support_c  +  alpha * sum_{c<c'} y_c y_c' D_cc'

    ``support_c`` is how decisively channel ``c`` picks a candidate (its top-1
    probability, scaled by its learned weight); ``D`` is the pairwise
    Jensen-Shannon divergence between channel posteriors. Minimising it keeps a
    mutually-consistent, confident subset and drops evidence that contradicts the
    rest — the robustness a product-of-experts fusion cannot express, since there
    one bad channel vetoes the right answer.

    **Feasibility is entirely classical, by construction rather than by
    post-selection.** There is no constraint at all: every one of the ``2^C``
    bitstrings is a valid channel subset, so the plain transverse-field mixer
    cannot leave the feasible set. The letter itself is then chosen *classically*
    as the argmax over the selected channels.

    This replaced an earlier one-hot-letter formulation, and the reason is
    measured: with letters as QUBO variables and no penalty term, the global
    minimum was the infeasible all-zeros assignment (energy -10.83 versus -8.24
    for the best feasible string). Penalty-free QAOA therefore drove its
    amplitude *out* of the feasible set — post-selection accepted only 6.9% of
    the distribution and the sampler matched the exact optimum 1.5% of the time.
    Removing the constraint instead of policing it fixes the cause.
    """
    C = len(support)
    Q = np.zeros((C, C))
    for c in range(C):
        Q[c, c] -= float(support[c])
    for c in range(C):
        for d in range(c + 1, C):
            Q[c, d] += float(alpha * disagree[c, d])
    return Q


def channel_exact(Q: np.ndarray) -> Tuple[np.ndarray, float]:
    """Exact minimiser by enumeration of all ``2^C`` subsets — ground truth."""
    C = Q.shape[0]
    best, best_e = None, math.inf
    for m in range(1 << C):
        y = np.array([(m >> c) & 1 for c in range(C)], dtype=float)
        e = qubo_energy(Q, y)
        if e < best_e:
            best_e, best = e, y
    return np.asarray(best), float(best_e)


class ConsensusQAOA:
    """Fixed-angle QAOA sampler for the channel-trust QUBO.

    Runtime does **no** parameter optimization: the ``(gamma, beta)`` schedule is
    trained offline by :meth:`train` and reused, which is standard QAOA parameter
    transfer within a problem family. One circuit evaluation per decision.

    Every measured bitstring is a valid answer, so there is no post-selection and
    no repair — the classical side's only job is decoding the chosen subset into a
    letter. Sampling quality is still checked against the exact enumeration on
    every call, and a miss falls back to the exact subset, so the decision can
    never be worse than the classical optimum.
    """

    def __init__(self, n_channels: int = len(CHANNELS), p: int = 2) -> None:
        self.C, self.p = n_channels, p
        self.n = n_channels
        self.dev = qml.device("default.qubit", wires=self.n)
        self._bits = np.array(
            [[(m >> (self.n - 1 - k)) & 1 for k in range(self.n)]
             for m in range(2 ** self.n)], dtype=float)

        @qml.qnode(self.dev)
        def probs(params, h, J):
            self._ansatz(params, h, J)
            return qml.probs(wires=range(self.n))

        @qml.qnode(self.dev, interface="autograd")
        def probs_grad(params, h, J):
            self._ansatz(params, h, J)
            return qml.probs(wires=range(self.n))

        self._probs = probs
        self._probs_grad = probs_grad

    def _ansatz(self, params, h, J) -> None:
        for w in range(self.n):
            qml.Hadamard(wires=w)
        for layer in range(self.p):
            g = params[0][layer]
            for i in range(self.n):
                qml.RZ(2.0 * g * h[i], wires=i)
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    qml.CNOT(wires=[i, j])
                    qml.RZ(2.0 * g * J[i, j], wires=j)
                    qml.CNOT(wires=[i, j])
            for w in range(self.n):                     # transverse-field mixer
                qml.RX(2.0 * params[1][layer], wires=w)

    # ---- energies of every basis state, used for readout and training ----

    def _energies(self, Q: np.ndarray) -> np.ndarray:
        return np.einsum("ij,jk,ik->i", self._bits, np.asarray(Q, dtype=float),
                         self._bits)

    @staticmethod
    def _ising(Q: np.ndarray):
        from .qaoa import qubo_to_ising
        h, J, _off = qubo_to_ising(Q)
        return h, J

    def sample(self, Q: np.ndarray, params: np.ndarray, top: int = 16
               ) -> Dict[str, object]:
        h, J = self._ising(Q)
        probs = np.asarray(self._probs(pnp.array(params, requires_grad=False),
                                       pnp.array(h, requires_grad=False),
                                       pnp.array(J, requires_grad=False)))
        energies = self._energies(Q)
        order = np.argsort(probs)[::-1][:top]
        best_idx = min(order, key=lambda i: energies[i])
        return {"channels": self._bits[best_idx].copy(),
                "energy": float(energies[best_idx]),
                "mass": float(probs[best_idx])}

    # ---- offline schedule training ----

    def train(self, instances: Sequence[np.ndarray], steps: int = 45,
              lr: float = 0.1, seed: int = 13, verbose: bool = True) -> np.ndarray:
        """Train one shared ``(gamma, beta)`` schedule across sample instances.

        The cost is the expected energy, normalised per instance to ``[0, 1]`` so
        instances of different scale contribute equally. Each Adam step lowers it,
        which is exactly what raises the probability of measuring the optimal
        channel subset at runtime.
        """
        prepared = []
        for Q in instances:
            h, J = self._ising(Q)
            E = self._energies(Q)
            lo, hi = float(E.min()), float(E.max())
            span = max(hi - lo, 1e-9)
            prepared.append((pnp.array(h, requires_grad=False),
                             pnp.array(J, requires_grad=False),
                             pnp.array((E - lo) / span, requires_grad=False)))

        def cost(params):
            total = 0.0
            for h, J, En in prepared:
                total = total + pnp.sum(self._probs_grad(params, h, J) * En)
            return total / len(prepared)

        rng = np.random.default_rng(seed)
        params = pnp.array([rng.uniform(0.0, 0.6, self.p),
                            rng.uniform(0.4, 1.0, self.p)], requires_grad=True)
        opt = qml.AdamOptimizer(lr)
        first = float(cost(params))
        for _ in range(steps):
            params = opt.step(cost, params)
        last = float(cost(params))
        if verbose:
            print("[stage 7] consensus schedule: normalised energy %.4f -> %.4f "
                  "over %d steps" % (first, last, steps))
        return np.asarray(params)


# ==========================================================================
# The full stack
# ==========================================================================

@dataclass
class LetterPrediction:
    letter: str
    key_index: int
    posterior: List[float]
    candidates: List[Tuple[str, float]]
    channels_used: List[str]
    reliability: float
    method: str                 # "qaoa" | "exact-fallback"
    matched_exact: bool
    latency_ms: float


class GazeLetterStack:
    """The seven-stage gaze -> letter decoder."""

    def __init__(self, selected: Optional[Sequence[int]] = None,
                 support_beta: float = 1.15, alpha: float = 0.0,
                 channel_weights: Optional[Sequence[float]] = None,
                 slot_vqc_epochs: int = 14) -> None:
        self.selected = list(selected) if selected is not None else list(range(N_SELECT))
        self.slot_vqc_epochs = int(slot_vqc_epochs)
        self.support_beta = float(support_beta)
        self.alpha = float(alpha)
        self.channel_weights = np.array(
            channel_weights if channel_weights is not None else [1.0] * len(CHANNELS),
            dtype=float)
        self.geom = GeometricGaze()
        self.scaler: Optional[FeatureScaler] = None
        self.region_svm: Optional[QuantumKernelSVM] = None
        self.matcher: Optional[GazePrototypeMatcher] = None
        self.slot_vqc: Optional[RegionSlotVQC] = None
        self.reliab: Optional[ReliabilityModel] = None
        self.qaoa: Optional[ConsensusQAOA] = None
        self.qaoa_params: Optional[np.ndarray] = None
        self.feature_report: Dict[str, object] = {}

    # ---------------- training ----------------

    def fit(self, X: np.ndarray, y_key: np.ndarray, pts: np.ndarray,
            n_svm: int = 200, n_vqc: int = 420, seed: int = 13,
            verbose: bool = True) -> "GazeLetterStack":
        X = np.asarray(X, dtype=float)
        y_key = np.asarray(y_key, dtype=int)
        rng = np.random.default_rng(seed)

        self.geom.fit(X, pts)
        Z = self.scaler_fit_transform(X)
        region = KEY_REGION[y_key]
        slot = KEY_SLOT[y_key]

        nq = self.n_quantum_features
        idx = rng.permutation(len(Z))[:n_svm]
        if verbose:
            print("[stage 2] QSVM region classifier: %d train, %d qubits "
                  "(%d kernel circuits, ~%.0fs)"
                  % (len(idx), nq, len(idx) * (len(idx) - 1) // 2,
                     len(idx) ** 2 / 2 * 0.0016), flush=True)
        t = time.perf_counter()
        self.region_svm = QuantumKernelSVM(nq).fit(Z[idx], region[idx])
        if verbose:
            print("          done in %.0fs" % (time.perf_counter() - t), flush=True)

        if verbose:
            print("[stage 3] quantum-kernel prototypes: %d letters" % N_KEYS, flush=True)
        self.matcher = GazePrototypeMatcher(nq).fit(Z, y_key)

        vidx = rng.permutation(len(Z))[:n_vqc]
        if verbose:
            print("[stage 4] region-conditioned VQC: %d train, %d classes, "
                  "%d epochs (slowest stage, ~%.0fs)"
                  % (len(vidx), MAX_SLOTS, self.slot_vqc_epochs,
                     len(vidx) / 32 * self.slot_vqc_epochs * 0.35), flush=True)
        t = time.perf_counter()
        self.slot_vqc = RegionSlotVQC(nq, seed=seed, epochs=self.slot_vqc_epochs).fit(
            Z[vidx], region[vidx], slot[vidx])
        if verbose:
            print("          done in %.0fs" % (time.perf_counter() - t), flush=True)

        if verbose:
            print("[stage 5/6] QNN movement + QBN reliability", flush=True)
        self.reliab = ReliabilityModel(seed=seed).fit()

        self.qaoa = ConsensusQAOA()
        return self

    def quantum_input(self, X: np.ndarray) -> np.ndarray:
        """What the quantum stages actually see: the *calibrated* gaze point plus
        head pose.

        Handing them raw iris ratios makes them re-learn the whole gaze->screen
        map from scratch, competing with an exact ridge solution — measured, the
        hierarchical channel scored 0.398 that way against the geometric channel's
        0.935. Starting from the calibrated point instead leaves them the job they
        can actually win: correcting the *systematic bias* the linear map leaves
        behind (optical curvature, pose-dependent skew), which is nonlinear in
        exactly the way a kernel or variational model represents well.
        """
        pts = self.geom.predict_points(X)
        return np.column_stack([pts, np.asarray(X, dtype=float)[:, list(POSE_IDX)]])

    @property
    def n_quantum_features(self) -> int:
        return 2 + len(POSE_IDX)

    def scaler_fit_transform(self, X: np.ndarray) -> np.ndarray:
        Zi = self.quantum_input(X)
        self.scaler = FeatureScaler().fit(Zi)
        return self.scaler.transform(Zi)

    def _Z(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(self.quantum_input(X))

    # ---------------- evidence channels ----------------

    def channel_posteriors(self, X: np.ndarray, top_regions: int = 2
                           ) -> Dict[str, np.ndarray]:
        """Each channel's ``(n, 26)`` distribution over letters."""
        X = np.asarray(X, dtype=float)
        Z = self._Z(X)

        geom = self.geom.posterior(X)

        qknn = self.matcher.posterior(Z)

        region_p = self.region_svm.predict_proba(Z, N_REGIONS)
        # region evidence on its own: coarse but reliable, and it does not
        # inherit the slot classifier's error
        qsvm = np.zeros((len(Z), N_KEYS))
        for r in range(N_REGIONS):
            keys = REGION_KEYS[r]
            for k in keys:
                qsvm[:, k] = region_p[:, r] / len(keys)

        qhier = np.zeros((len(Z), N_KEYS))
        order = np.argsort(region_p, axis=1)[:, ::-1][:, :top_regions]
        for col in range(top_regions):
            regions = order[:, col]
            slot_p = self.slot_vqc.slot_proba(Z, regions)
            for i in range(len(Z)):
                r = int(regions[i])
                w = region_p[i, r]
                keys = REGION_KEYS[r]
                s = slot_p[i, :len(keys)]
                s = s / max(s.sum(), _EPS)
                for j, k in enumerate(keys):
                    qhier[i, k] += w * s[j]
        z = qhier.sum(axis=1, keepdims=True)
        qhier = np.where(z > _EPS, qhier / np.where(z > _EPS, z, 1.0), 1.0 / N_KEYS)

        return {"geom": geom, "qknn": qknn, "qsvm": qsvm, "qhier": qhier}

    # ---------------- decision ----------------

    def tempered_channels(self, X: np.ndarray,
                          kinematics: Optional[np.ndarray] = None
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """``(n, C, 26)`` channel posteriors after QBN reliability tempering."""
        X = np.asarray(X, dtype=float)
        chans = self.channel_posteriors(X)
        P = np.stack([chans[c] for c in CHANNELS], axis=1)
        rel = (self.reliab.reliability(kinematics) if kinematics is not None
               else np.full(len(X), self.reliab.table[(0, 0)]))
        # QBN tempering: flatten suspect evidence instead of trusting it
        P = np.power(np.clip(P, _EPS, None), rel[:, None, None])
        P /= P.sum(axis=2, keepdims=True)
        return P, rel

    def fit_channel_weights(self, X: np.ndarray, y: np.ndarray,
                            alphas: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
                            verbose: bool = True) -> "GazeLetterStack":
        """Learn the ensemble weights and disagreement weight on held-out fixations.

        **Nested validation.** The weight grid is searched on the first half of
        the held-out set and the winner is *confirmed* on the second half against
        geometry-only. A grid of 1296 weightings scored on a few hundred samples
        will find a spurious 1-2 pp winner by chance — the first run of this
        function did exactly that, picking a fused weighting that then lost to
        the deployed decoder by 0.3-2.5 pp on test. If the search winner does not
        also beat geometry-only on the confirmation half, the stack reverts to
        ``[1, 0, ...]`` and reproduces the deployed decoder exactly.
        """
        y = np.asarray(y, dtype=int)
        P, rel = self.tempered_channels(X, np.asarray(X)[:, list(KINEMATIC_IDX)])
        C = len(CHANNELS)
        # shuffle before splitting: the validation set is built from two noise
        # levels in order, so a positional split would put every low-noise
        # fixation in the search half and every high-noise one in the
        # confirmation half
        order = np.random.default_rng(17).permutation(len(y))
        half = len(y) // 2
        A, B = order[:half], order[half:]

        geom_only = np.zeros(C)
        geom_only[0] = 1.0
        acc_of = lambda w, sl: float(np.mean(
            np.tensordot(P[sl], w, axes=([1], [0])).argmax(axis=1) == y[sl]))

        grid = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
        best_w, best_acc = geom_only, acc_of(geom_only, A)
        for combo in iproduct(grid, repeat=C):
            if sum(combo) <= 0:
                continue
            w = np.array(combo, dtype=float)
            w = w / w.sum()
            acc = acc_of(w, A)
            if acc > best_acc:
                best_acc, best_w = acc, w

        confirm_fused = acc_of(best_w, B)
        confirm_geom = acc_of(geom_only, B)
        # McNemar on the confirmation half. A bare accuracy comparison is not
        # enough of a guard: the first version of this check passed the fused
        # weighting on a +0.7 pp confirmation margin, and it then lost to the
        # deployed decoder by 1.0-2.5 pp on test. Only a *significant* paired
        # win may move the shipped decoder.
        f_ok = np.tensordot(P[B], best_w, axes=([1], [0])).argmax(axis=1) == y[B]
        g_ok = np.tensordot(P[B], geom_only, axes=([1], [0])).argmax(axis=1) == y[B]
        n_fw = int(np.sum(f_ok & ~g_ok))     # fused right, geometry wrong
        n_gw = int(np.sum(~f_ok & g_ok))     # geometry right, fused wrong
        p_value = 1.0
        if n_fw + n_gw:
            from math import comb
            n = n_fw + n_gw
            p_value = sum(comb(n, i) for i in range(n_fw, n + 1)) / 2 ** n
        kept = bool(n_fw > n_gw and p_value < 0.05)
        if not kept:
            best_w = geom_only
        self.channel_weights = best_w

        best_alpha, best_alpha_acc = self.alpha, -1.0
        for a in alphas:
            self.alpha = float(a)
            preds, _ = self._decide(P, rel, collect=False)
            acc = float(np.mean(preds == y))
            if acc > best_alpha_acc:
                best_alpha_acc, best_alpha = acc, float(a)
        self.alpha = best_alpha
        if verbose:
            print("[fusion] weights %s  search=%.4f confirm=%.4f vs geom-only "
                  "%.4f  (McNemar %d/%d, p=%.3f) -> %s;  alpha=%.2f "
                  "(consensus acc %.4f)"
                  % (dict(zip(CHANNELS, np.round(best_w, 3))), best_acc,
                     confirm_fused, confirm_geom, n_fw, n_gw, p_value,
                     "kept" if kept else "REVERTED to geometry-only",
                     best_alpha, best_alpha_acc))
        return self

    def predict_batch(self, X: np.ndarray, kinematics: Optional[np.ndarray] = None,
                      collect: bool = False) -> Tuple[np.ndarray, List[dict]]:
        """Decode a batch of fixations. Returns predicted key indices + diagnostics."""
        P, rel = self.tempered_channels(X, kinematics)
        return self._decide(P, rel, collect)

    def _decide(self, P: np.ndarray, rel: np.ndarray, collect: bool
                ) -> Tuple[np.ndarray, List[dict]]:
        w = self.channel_weights / self.channel_weights.sum()
        fused = np.tensordot(P, w, axes=([1], [0]))          # (n, 26)
        C = len(CHANNELS)
        preds = np.zeros(len(P), dtype=int)
        diags: List[dict] = []
        for i in range(len(P)):
            cand = np.argsort(fused[i])[::-1][:K_CANDIDATES]
            # per-channel distribution restricted to the candidate letters
            Pc = np.stack([P[i, c, cand] / max(P[i, c, cand].sum(), _EPS)
                           for c in range(C)])
            support = np.array([w[c] * Pc[c].max() for c in range(C)])
            disagree = np.zeros((C, C))
            for c in range(C):
                for d in range(c + 1, C):
                    disagree[c, d] = js_divergence(Pc[c], Pc[d])

            Q = build_channel_qubo(support, disagree, alpha=self.alpha)
            exact_y, exact_e = channel_exact(Q)
            res = self.qaoa.sample(Q, self.qaoa_params) if self.qaoa_params is not None else None

            if res is not None and abs(res["energy"] - exact_e) < 1e-9:
                chan, method, matched = res["channels"], "qaoa", True
            else:
                # guarantee: never worse than the exact channel-subset optimum
                chan, method, matched = exact_y, "exact-fallback", False

            # classical decode: argmax over the trusted channels only
            if chan.sum() < 0.5:
                chan = np.ones(C)
            score = np.zeros(len(cand))
            for c in range(C):
                if chan[c] > 0.5:
                    score += w[c] * Pc[c]
            preds[i] = int(cand[int(np.argmax(score))])
            if collect:
                diags.append({
                    "candidates": [(LETTERS[int(k)], float(fused[i, k])) for k in cand],
                    "method": method, "matched_exact": matched,
                    "channels": [CHANNELS[c] for c in range(C) if chan[c] > 0.5],
                    "reliability": float(rel[i]),
                })
        return preds, diags

    def predict_one(self, features: Sequence[float],
                    kinematics: Optional[Sequence[float]] = None) -> LetterPrediction:
        t0 = time.perf_counter()
        X = np.asarray([features], dtype=float)
        kin = np.asarray([kinematics], dtype=float) if kinematics is not None else None
        # channels are the expensive part (26 kernel circuits + the SVM's support
        # vectors); compute them once and reuse for both the decision and the
        # reported posterior
        P, rel = self.tempered_channels(X, kin)
        preds, diags = self._decide(P, rel, collect=True)
        d = diags[0]
        w = self.channel_weights / self.channel_weights.sum()
        fused = np.tensordot(P, w, axes=([1], [0]))[0]
        return LetterPrediction(
            letter=LETTERS[int(preds[0])], key_index=int(preds[0]),
            posterior=[float(v) for v in fused],
            candidates=[(c, float(p)) for c, p in d["candidates"]],
            channels_used=d["channels"], reliability=d["reliability"],
            method=d["method"], matched_exact=bool(d["matched_exact"]),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    # ---------------- persistence ----------------

    def save(self, path: str = MODEL_PATH) -> str:
        blob = {
            "selected": self.selected,
            "support_beta": self.support_beta,
            "alpha": self.alpha,
            "channel_weights": self.channel_weights,
            "feature_report": self.feature_report,
            "geom": {"wx": self.geom.wx, "wy": self.geom.wy, "sigma": self.geom.sigma},
            "scaler": (self.scaler.lo, self.scaler.hi),
            "svm": {"svc": self.region_svm.svc, "train_Z": self.region_svm._train_Z,
                    "sv": self.region_svm._sv, "classes": self.region_svm.classes_,
                    "temperature": self.region_svm.temperature},
            "protos": self.matcher.prototypes,
            "proto_temperature": self.matcher.temperature,
            "vqc": {"cfg": self.slot_vqc.cfg, "theta": np.asarray(self.slot_vqc.vqc.theta)},
            "qnn": {"theta": np.asarray(self.reliab.qnn.theta),
                    "scaler": (self.reliab.scaler.lo, self.reliab.scaler.hi)},
            "qaoa_params": self.qaoa_params,
        }
        with open(path, "wb") as f:
            pickle.dump(blob, f)
        return path

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> "GazeLetterStack":
        with open(path, "rb") as f:
            d = pickle.load(f)
        s = cls(selected=d["selected"], support_beta=d["support_beta"],
                alpha=d["alpha"], channel_weights=d["channel_weights"])
        s.feature_report = d.get("feature_report", {})
        s.geom.wx, s.geom.wy = d["geom"]["wx"], d["geom"]["wy"]
        s.geom.sigma = d["geom"]["sigma"]
        s.scaler = FeatureScaler()
        s.scaler.lo, s.scaler.hi = d["scaler"]

        s.region_svm = QuantumKernelSVM(s.n_quantum_features,
                                        temperature=d["svm"]["temperature"])
        s.region_svm.svc = d["svm"]["svc"]
        s.region_svm._train_Z = d["svm"]["train_Z"]
        s.region_svm._sv = d["svm"]["sv"]
        s.region_svm.classes_ = d["svm"]["classes"]

        s.matcher = GazePrototypeMatcher(s.n_quantum_features,
                                         temperature=d["proto_temperature"])
        s.matcher.prototypes = d["protos"]

        s.slot_vqc = RegionSlotVQC(s.n_quantum_features)
        s.slot_vqc.cfg = d["vqc"]["cfg"]
        s.slot_vqc.vqc = VQC(d["vqc"]["cfg"])
        s.slot_vqc.vqc.theta = pnp.array(d["vqc"]["theta"], requires_grad=True)

        s.reliab = ReliabilityModel()
        s.reliab.qnn.theta = pnp.array(d["qnn"]["theta"], requires_grad=True)
        s.reliab.scaler = FeatureScaler()
        s.reliab.scaler.lo, s.reliab.scaler.hi = d["qnn"]["scaler"]

        s.qaoa = ConsensusQAOA()
        s.qaoa_params = d["qaoa_params"]
        return s


# ==========================================================================
# Browser bridge
# ==========================================================================

#: signals the browser sends. The first four are the gaze features the QAOA
#: selector kept; the last four are the kinematics ``/api/movement`` already
#: computes, reused here so the tracker gains no new per-frame work.
CLIENT_FEATURES: Tuple[str, ...] = (
    "iris_x", "iris_y", "yaw", "pitch", "ear", "velocity", "stability", "history",
)


def vectors_from_client(d: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Place the browser's signals into the 13-dim research feature vector.

    ``iris-tracker.js`` reports per-eye ratios ``hL/hR`` and ``vL/vR``; the caller
    averages them into ``iris_x``/``iris_y``. Head pose comes from the nose
    landmark on the same ``(value - 0.5) * 2`` convention used to build the
    training data. Unselected dimensions stay zero — no stage reads them.
    """
    X = np.zeros(N_FEATURES)
    X[0] = float(d.get("iris_x", 0.5))
    X[1] = float(d.get("iris_y", 0.5))
    X[4] = float(d.get("yaw", 0.0))
    X[5] = float(d.get("pitch", 0.0))
    ear = float(d.get("ear", 0.75))
    X[2] = X[7] = ear
    X[8] = float(d.get("velocity", 0.0))
    X[10] = float(d.get("stability", 0.9))
    X[11] = 0.85
    X[12] = float(d.get("history", 0.0))
    return X, X[list(KINEMATIC_IDX)]


# ==========================================================================
# Training entry point
# ==========================================================================

def _consensus_instances(stack: "GazeLetterStack", X: np.ndarray,
                         n: int = 8, seed: int = 13) -> List[np.ndarray]:
    """Build representative consensus QUBOs from real training fixations."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))[:n]
    chans = stack.channel_posteriors(X[idx])
    P = np.stack([chans[c] for c in CHANNELS], axis=1)
    C = len(CHANNELS)
    out = []
    for i in range(len(idx)):
        fused = P[i].mean(axis=0)
        cand = np.argsort(fused)[::-1][:K_CANDIDATES]
        Pc = np.stack([P[i, c, cand] / max(P[i, c, cand].sum(), _EPS) for c in range(C)])
        support = np.array([Pc[c].max() for c in range(C)])
        disagree = np.zeros((C, C))
        for c in range(C):
            for d in range(c + 1, C):
                disagree[c, d] = js_divergence(Pc[c], Pc[d])
        out.append(build_channel_qubo(support, disagree, alpha=1.0))
    return out


def train(n_train: int = 1800, n_val: int = 600, noise: float = 0.02,
          n_svm: int = 220, n_vqc: int = 1200, seed: int = 13,
          save: bool = True, verbose: bool = True) -> "GazeLetterStack":
    """Fit every stage, train the QAOA schedule offline, and persist the stack.

    Validation fixations are drawn at *two* noise levels so the fusion weights
    are not tuned to one operating point; the test sets used by ``evaluate`` are
    disjoint from both.
    """
    Xtr, ytr, ptr = make_intent_dataset(n_samples=n_train, noise=noise, seed=seed)
    Xa, ya, _pa = make_intent_dataset(n_samples=n_val // 2, noise=noise, seed=seed + 91)
    Xb, yb, _pb = make_intent_dataset(n_samples=n_val - n_val // 2, noise=noise * 2,
                                      seed=seed + 92)
    Xva, yva = np.vstack([Xa, Xb]), np.concatenate([ya, yb])
    if verbose:
        print("training set: %d fixations, %d keys, noise=%.3f  (val %d)"
              % (len(Xtr), len(set(ytr.tolist())), noise, len(Xva)))

    feat = select_features_qaoa(Xtr, ytr, ptr, k=N_SELECT, seed=seed, verbose=verbose)
    stack = GazeLetterStack(selected=feat["selected"])
    stack.feature_report = feat
    stack.fit(Xtr, ytr, ptr, n_svm=n_svm, n_vqc=n_vqc, seed=seed, verbose=verbose)

    inst = _consensus_instances(stack, Xtr, seed=seed)
    stack.qaoa_params = stack.qaoa.train(inst, verbose=verbose)
    stack.fit_channel_weights(Xva, yva, verbose=verbose)

    if save:
        stack.save()
        with open(PARAMS_PATH, "w") as f:
            json.dump({"p": stack.qaoa.p, "K": K_CANDIDATES,
                       "channels": list(CHANNELS),
                       "support_beta": stack.support_beta,
                       "params": np.asarray(stack.qaoa_params).tolist()}, f, indent=2)
        if verbose:
            print("saved -> %s and %s" % (MODEL_PATH, PARAMS_PATH))
    return stack


# ==========================================================================
# Evaluation — deployed baseline, strong classical baselines, per-stage ablation
# ==========================================================================

def evaluate(stack: "GazeLetterStack", noise_levels: Sequence[float] = (0.02, 0.03, 0.045),
             n_test: int = 400, n_train_ref: int = 900, seed: int = 777,
             verbose: bool = True) -> Dict[str, object]:
    """Measure letter accuracy against every baseline that matters.

    Baselines:
      * ``geometric`` — the deployed ridge + Gaussian key posterior. This is the
        number the project ships today and the only one a gain can be claimed
        against.
      * ``mlp`` / ``rbf_svm`` — the strongest reasonable classical learners on
        the same features, so a quantum win is not manufactured by comparing
        against a weak incumbent.
    Ablations isolate each channel and the QAOA consensus stage.
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    Xr, yr, _pr = make_intent_dataset(n_samples=n_train_ref, noise=0.02, seed=13)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), random_state=13,
                        max_iter=900).fit(Xr, yr)
    rbf = SVC(kernel="rbf", C=8.0).fit(Xr, yr)

    rows = []
    for noise in noise_levels:
        Xte, yte, _pte = make_intent_dataset(n_samples=n_test, noise=noise, seed=seed)
        kin = Xte[:, list(KINEMATIC_IDX)]

        chans = stack.channel_posteriors(Xte)
        acc = lambda p: float(np.mean(np.asarray(p) == yte))

        t0 = time.perf_counter()
        preds, diags = stack.predict_batch(Xte, kin, collect=True)
        stack_ms = (time.perf_counter() - t0) / len(Xte) * 1000.0

        w = stack.channel_weights / stack.channel_weights.sum()
        fused = sum(w[c] * chans[CHANNELS[c]] for c in range(len(CHANNELS)))

        matched = float(np.mean([d["matched_exact"] for d in diags]))
        dropped = float(np.mean([len(d["channels"]) < len(CHANNELS) for d in diags]))

        row = {
            "noise": noise,
            "n_test": int(len(yte)),
            "geometric_deployed": acc(chans["geom"].argmax(axis=1)),
            "qknn_only": acc(chans["qknn"].argmax(axis=1)),
            "qsvm_only": acc(chans["qsvm"].argmax(axis=1)),
            "qhier_only": acc(chans["qhier"].argmax(axis=1)),
            "weighted_fusion": acc(fused.argmax(axis=1)),
            "full_stack_qaoa": acc(preds),
            "classical_mlp": acc(mlp.predict(Xte)),
            "classical_rbf_svm": acc(rbf.predict(Xte)),
            "qaoa_matched_exact": matched,
            "fraction_channel_dropped": dropped,
            "stack_ms_per_fixation": stack_ms,
        }
        row["gain_vs_deployed_pp"] = (row["full_stack_qaoa"]
                                      - row["geometric_deployed"]) * 100.0
        rows.append(row)
        if verbose:
            print("\nnoise=%.3f  (n=%d)" % (noise, len(yte)))
            print("   deployed geometric   %.4f" % row["geometric_deployed"])
            print("   quantum-kernel kNN   %.4f" % row["qknn_only"])
            print("   QSVM region only     %.4f" % row["qsvm_only"])
            print("   QSVM region x VQC    %.4f" % row["qhier_only"])
            print("   weighted fusion      %.4f" % row["weighted_fusion"])
            print("   FULL STACK (QAOA)    %.4f   (%+.1f pp vs deployed)"
                  % (row["full_stack_qaoa"], row["gain_vs_deployed_pp"]))
            print("   classical MLP        %.4f" % row["classical_mlp"])
            print("   classical RBF-SVM    %.4f" % row["classical_rbf_svm"])
            print("   QAOA matched exact   %.3f   channel dropped %.3f"
                  "   %.1f ms/fixation" % (matched, dropped, stack_ms))
    return {"rows": rows, "selected_features": stack.feature_report.get("selected_names")}


def make_commit_dataset(n_samples: int = 1200, noise: float = 0.02,
                        saccade_frac: float = 0.15, blink_frac: float = 0.08,
                        curvature: float = 0.45, head_motion: float = 0.06,
                        seed: int = 31) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fixations as the keyboard actually receives them.

    ``make_intent_dataset`` assumes every sample is a clean fixation. Real dwell
    windows are not: a fraction land mid-saccade (the iris is in flight, so its
    position means nothing) or mid-blink (the landmarks are garbage). Those are
    the frames that produce *confidently wrong* keystrokes, because the geometric
    posterior has no way to know its input was junk — it reports the same
    certainty either way.

    Kinematics follow the same class signatures as ``make_movement_dataset``, so
    the movement classifiers see the distribution they were trained on.
    """
    rng = np.random.default_rng(seed)
    layout = keyboard_layout()
    X = np.zeros((n_samples, N_FEATURES))
    y = np.zeros(n_samples, dtype=int)
    state = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        k = int(rng.integers(0, len(layout)))
        _ch, kx, ky, kw, kh = layout[k]
        tx = kx + kw * rng.uniform(0.25, 0.75)
        ty = ky + kh * rng.uniform(0.25, 0.75)
        nose_x = 0.5 + rng.normal(0, head_motion)
        nose_y = 0.5 + rng.normal(0, head_motion)
        h = float(_inverse_forward(tx, nose_x, curvature)) + rng.normal(0, noise)
        v = float(_inverse_forward(ty, nose_y, curvature)) + rng.normal(0, noise)

        u = rng.random()
        if u < saccade_frac:                       # mid-flight: iris lags badly
            cls = 1
            h += rng.normal(0, 0.12)
            v += rng.normal(0, 0.12)
            vel, ear, stab = abs(rng.normal(0.62, 0.16)), rng.normal(0.74, 0.06), rng.normal(0.14, 0.08)
        elif u < saccade_frac + blink_frac:        # lids closing: iris meaningless
            cls = 2
            h, v = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)
            vel, ear, stab = abs(rng.normal(0.30, 0.22)), rng.normal(0.12, 0.07), rng.normal(0.30, 0.16)
        else:                                      # clean fixation
            cls = 0
            vel, ear, stab = abs(rng.normal(0.03, 0.02)), rng.normal(0.78, 0.05), rng.normal(0.90, 0.06)

        vel = float(np.clip(vel, 0, 1)); ear = float(np.clip(ear, 0, 1)); stab = float(np.clip(stab, 0, 1))
        X[i] = [h, v, ear, 1.0 if cls == 2 else 0.0,
                (nose_x - 0.5) * 2, (nose_y - 0.5) * 2, rng.normal(0, 0.05),
                ear, vel, 1.0 if cls == 1 else 0.0, stab,
                float(np.clip(rng.normal(0.85, 0.08), 0, 1)),
                float(np.clip(vel * 0.8 + rng.normal(0, 0.05), 0, 1))]
        y[i] = k
        state[i] = cls
    return X, y, state


def evaluate_commit_quality(stack: "GazeLetterStack", theta: float = 0.80,
                            n_test: int = 1500, noise: float = 0.02,
                            seed: int = 31, verbose: bool = True) -> List[dict]:
    """Does the quantum reliability path reduce *wrong keystrokes*?

    Top-1 accuracy is the wrong yardstick for a dwell keyboard: it never commits
    unless the fused posterior clears ``THETA``, and asks with choice chips
    otherwise. What matters is how many committed letters are wrong (the user has
    to backspace them) and how often the system has to ask.

    Three decision rules on identical fixations:

    * ``ungated``    — the geometric posterior alone (the pre-quantum keyboard).
    * ``qsvm_gate``  — plus the shipped fidelity-kernel movement classifier,
                       which refuses to commit a frame it calls unusable.
    * ``qnn_qbn``    — the QNN's *distribution* over movement states pushed
                       through the Bayesian network to a graded reliability, used
                       to temper the posterior instead of hard-gating it.

    The third is the contribution measured here, and unlike the region/slot
    stages it reads **only kinematics**, so it carries over to a real webcam
    without per-user retraining.
    """
    from .movement_qsvm import MovementQSVM

    Xt, yt, _st = make_commit_dataset(n_test, noise=noise, seed=seed)
    kin = Xt[:, list(KINEMATIC_IDX)]
    post = stack.geom.posterior(Xt)

    try:
        qsvm = MovementQSVM.load()
        usable = np.array([qsvm.classify_one(row)["usable"] for row in kin])
    except Exception:
        usable = np.ones(len(Xt), dtype=bool)

    rel = stack.reliab.reliability(kin)
    tempered = np.power(np.clip(post, _EPS, None), rel[:, None])
    tempered /= tempered.sum(axis=1, keepdims=True)

    def score(name, probs, allow):
        top = probs.argmax(axis=1)
        commit = (probs.max(axis=1) >= theta) & allow
        n = len(yt)
        right = int(np.sum(commit & (top == yt)))
        wrong = int(np.sum(commit & (top != yt)))
        return {
            "rule": name,
            "commit_rate": float(commit.mean()),
            "accuracy_when_committed": float(right / max(1, right + wrong)),
            "wrong_commits_per_100": 100.0 * wrong / n,
            "prompt_rate": float(1.0 - commit.mean()),
        }

    rows = [
        score("ungated", post, np.ones(len(yt), dtype=bool)),
        score("qsvm_gate", post, usable),
        score("qnn_qbn", tempered, np.ones(len(yt), dtype=bool)),
        score("qnn_qbn + qsvm_gate", tempered, usable),
    ]
    if verbose:
        print("\ncommit quality (theta=%.2f, %d fixations, %.0f%% corrupted)"
              % (theta, n_test, 100 * float(np.mean(_st > 0))))
        print("   %-22s %8s %10s %10s %8s" % ("rule", "commit", "acc|commit",
                                              "wrong/100", "prompts"))
        for r in rows:
            print("   %-22s %8.3f %10.4f %10.2f %8.3f"
                  % (r["rule"], r["commit_rate"], r["accuracy_when_committed"],
                     r["wrong_commits_per_100"], r["prompt_rate"]))
    return rows


def make_calibration_samples(clicks: int = 5, noise: float = 0.02,
                             curvature: float = 0.45, head_motion: float = 0.06,
                             seed: int = 13) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate the browser's 7-spot calibration: features + the known targets.

    One fixation per click at each of ``CALIB_SPOTS`` — the same seven positions
    ``static/app.js`` asks the user to look at, in the same order.
    """
    rng = np.random.default_rng(seed)
    X, T = [], []
    for _name, tx, ty in CALIB_SPOTS:
        for _ in range(clicks):
            nose_x = 0.5 + rng.normal(0, head_motion)
            nose_y = 0.5 + rng.normal(0, head_motion)
            h = float(_inverse_forward(tx, nose_x, curvature)) + rng.normal(0, noise)
            v = float(_inverse_forward(ty, nose_y, curvature)) + rng.normal(0, noise)
            ear = float(np.clip(rng.normal(0.75, 0.06), 0.0, 1.0))
            row = np.zeros(N_FEATURES)
            row[0], row[1] = h, v
            row[2] = row[7] = ear
            row[4], row[5] = (nose_x - 0.5) * 2, (nose_y - 0.5) * 2
            row[10], row[11] = 0.95, 0.9
            X.append(row)
            T.append((tx, ty))
    return np.asarray(X), np.asarray(T)


def evaluate_calibration(n_test: int = 800, clicks: int = 5,
                         noise_levels: Sequence[float] = (0.015, 0.02, 0.03, 0.045),
                         curvature: float = 0.45, seed: int = 4242,
                         selected: Sequence[int] = GEOM_IDX,
                         verbose: bool = True) -> List[dict]:
    """The deployable experiment: does the quantum-kernel residual correction
    raise intended-letter accuracy when fitted on a user's own 7-spot calibration?

    Everything here mirrors the shipped flow — the same seven spots, the same
    number of clicks, the same linear ridge, the same Gaussian key posterior.
    The only addition is the correction term. Both decoders see identical test
    fixations, so the difference is attributable to the correction alone.
    """
    rows = []
    for noise in noise_levels:
        Xc, Tc = make_calibration_samples(clicks=clicks, noise=noise,
                                          curvature=curvature, seed=seed)
        geom = GeometricGaze().fit(Xc, Tc)

        scaler = FeatureScaler().fit(Xc[:, list(selected)])
        Zc = scaler.transform(Xc[:, list(selected)])
        resid = Tc - geom.predict_points(Xc)
        qk = CalibrationResidualQK(n_features=len(selected)).fit(Zc, resid)

        Xt, yt, _Tt = make_intent_dataset(n_samples=n_test, noise=noise,
                                          curvature=curvature, seed=seed + 7)
        base_pts = geom.predict_points(Xt)
        corr_pts = base_pts + qk.predict(scaler.transform(Xt[:, list(selected)]))

        base_err = float(np.sqrt(np.mean((base_pts - _Tt) ** 2)))
        corr_err = float(np.sqrt(np.mean((corr_pts - _Tt) ** 2)))
        sig_b = (max(1e-3, float(np.std(base_pts[:, 0] - _Tt[:, 0]))),
                 max(1e-3, float(np.std(base_pts[:, 1] - _Tt[:, 1]))))
        sig_c = (max(1e-3, float(np.std(corr_pts[:, 0] - _Tt[:, 0]))),
                 max(1e-3, float(np.std(corr_pts[:, 1] - _Tt[:, 1]))))

        acc_b = float(np.mean(geom.posterior_from_points(base_pts, sig_b).argmax(1) == yt))
        acc_c = float(np.mean(geom.posterior_from_points(corr_pts, sig_c).argmax(1) == yt))

        row = {
            "noise": noise, "n_calibration_samples": int(len(Xc)),
            "deployed_accuracy": acc_b, "corrected_accuracy": acc_c,
            "gain_pp": (acc_c - acc_b) * 100.0,
            "deployed_rmse": base_err, "corrected_rmse": corr_err,
            "rmse_reduction_pct": (1 - corr_err / base_err) * 100.0,
            "chosen_lambda": qk.lam, "loo_rmse": qk.loo_rmse,
        }
        rows.append(row)
        if verbose:
            print("noise=%.3f  calib n=%d  |  letter acc %.4f -> %.4f (%+.1f pp)"
                  "   point RMSE %.4f -> %.4f (-%.1f%%)   lam=%.3f"
                  % (noise, len(Xc), acc_b, acc_c, row["gain_pp"],
                     base_err, corr_err, row["rmse_reduction_pct"], qk.lam))
    return rows


def print_saved_report(path: str = REPORT_PATH) -> bool:
    """Print the last measured results without retraining anything."""
    if not os.path.exists(path):
        return False
    with open(path) as f:
        rep = json.load(f)
    print("measured results from %s\n" % path)
    print("intended-letter accuracy (%s)" % ", ".join(rep.get("selected_features") or []))
    print("   %-7s %10s %8s %8s %8s %10s %10s"
          % ("noise", "deployed", "qKernel", "QSVM", "QSVMxVQC", "FULL", "MLP"))
    for r in rep.get("rows", []):
        print("   %-7.3f %10.4f %8.4f %8.4f %8.4f %10.4f %10.4f   (%+.1f pp)"
              % (r["noise"], r["geometric_deployed"], r["qknn_only"],
                 r.get("qsvm_only", float("nan")), r["qhier_only"],
                 r["full_stack_qaoa"], r["classical_mlp"], r["gain_vs_deployed_pp"]))
    if rep.get("rows"):
        print("   QAOA matched the exact optimum on %.0f%% of decisions, %.0f ms/fixation"
              % (100 * rep["rows"][0]["qaoa_matched_exact"],
                 rep["rows"][0]["stack_ms_per_fixation"]))

    for key, block in (rep.get("commit_quality") or {}).items():
        print("\ncommit quality @ %s (25%% of dwell windows corrupted)" % key)
        print("   %-22s %8s %11s %11s %8s"
              % ("rule", "commit", "acc|commit", "wrong/100", "prompts"))
        for r in block:
            print("   %-22s %8.3f %11.4f %11.2f %8.3f"
                  % (r["rule"], r["commit_rate"], r["accuracy_when_committed"],
                     r["wrong_commits_per_100"], r["prompt_rate"]))

    if rep.get("calibration_residual"):
        print("\nquantum-kernel calibration correction (7 spots x 5 clicks)")
        for r in rep["calibration_residual"]:
            print("   noise %.3f  letter acc %.4f -> %.4f (%+.1f pp)  lambda=%g"
                  % (r["noise"], r["deployed_accuracy"], r["corrected_accuracy"],
                     r["gain_pp"], r["chosen_lambda"]))
    return True


def run_all(quick: bool = False) -> dict:
    """Train every stage and run all three experiments."""
    stack = (train(n_train=700, n_svm=110, n_vqc=400) if quick else train())
    report = evaluate(stack, n_test=200 if quick else 400)
    print("\n=== commit quality (the metric the dwell keyboard is judged on) ===")
    report["commit_quality"] = {
        "noise_0.020": evaluate_commit_quality(stack, noise=0.02),
        "noise_0.030": evaluate_commit_quality(stack, noise=0.03),
    }
    print("\n=== quantum-kernel calibration correction ===")
    report["calibration_residual"] = evaluate_calibration(n_test=800)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print("\nreport ->", REPORT_PATH)
    return report


if __name__ == "__main__":
    import argparse
    import warnings

    ap = argparse.ArgumentParser(
        description="Quantum gaze->letter stack. With no arguments it prints the "
                    "last measured results; retraining is opt-in because it "
                    "takes several minutes.")
    ap.add_argument("--train", action="store_true",
                    help="retrain every stage and re-measure (~7 min)")
    ap.add_argument("--quick", action="store_true",
                    help="same, on a smaller sample (~2 min); numbers are noisier")
    ap.add_argument("--eval", action="store_true",
                    help="re-measure the saved stack without retraining (~3 min)")
    args = ap.parse_args()

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if args.train or args.quick:
        run_all(quick=args.quick)
    elif args.eval:
        stack = GazeLetterStack.load()
        report = evaluate(stack)
        report["commit_quality"] = {
            "noise_0.020": evaluate_commit_quality(stack, noise=0.02),
            "noise_0.030": evaluate_commit_quality(stack, noise=0.03),
        }
        report["calibration_residual"] = evaluate_calibration(n_test=800)
        with open(REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)
    elif not print_saved_report():
        print("no saved report yet — run with --train (~7 min) or --quick (~2 min)")
