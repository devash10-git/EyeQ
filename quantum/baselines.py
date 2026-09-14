"""Matched classical baselines.

Every quantum module is compared against the strongest *reasonable* classical
counterpart, not a strawman. Where a classical method is provably exact
(ridge regression, Viterbi, BN enumeration, permutation search), that exactness
is stated — an approximate quantum method cannot beat an exact classical one,
and pretending otherwise would invalidate the study.
"""

from __future__ import annotations

import time
from itertools import permutations
from typing import Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------
# Classification baselines
# --------------------------------------------------------------------------

def rbf_svm(C: float = 1.0, gamma: str | float = "scale"):
    """Classical RBF-kernel SVM — the counterpart to the QSVM."""
    from sklearn.svm import SVC
    return SVC(kernel="rbf", C=C, gamma=gamma)


def linear_svm(C: float = 1.0):
    from sklearn.svm import SVC
    return SVC(kernel="linear", C=C)


def mlp(hidden: Tuple[int, ...] = (32, 16), seed: int = 13, max_iter: int = 600):
    """Small MLP — the counterpart to the VQC / QNN.

    Sized to a comparable parameter budget so the comparison is about model
    *class*, not capacity: a 3-layer VQC on 8 qubits has 72 parameters, and a
    (32,16) MLP on 8 inputs has roughly 800 — i.e. the classical model is given
    a generous advantage, which strengthens any quantum win and weakens any
    quantum loss. Both are reported.
    """
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(hidden_layer_sizes=hidden, random_state=seed,
                         max_iter=max_iter, early_stopping=False)


def logistic(seed: int = 13):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=1000, random_state=seed)


def ivt_movement_classifier(X: np.ndarray,
                            vel_idx: int = 8, ear_idx: int = 2,
                            vel_thresh: float = 0.30,
                            ear_thresh: float = 0.35,
                            stab_idx: int = 10,
                            stab_thresh: float = 0.45) -> np.ndarray:
    """I-VT style rule baseline for eye-movement state.

    This is the algorithm actually deployed in the classical system (velocity
    thresholding with an eye-closure guard), so it is the honest incumbent for
    the QNN to beat — not a learned model.
    """
    X = np.asarray(X, dtype=float)
    out = np.zeros(len(X), dtype=int)
    for i, row in enumerate(X):
        if row[ear_idx] < ear_thresh:
            out[i] = 2                                    # blink
        elif row[vel_idx] > vel_thresh:
            out[i] = 1                                    # saccade
        elif row[stab_idx] < stab_thresh:
            out[i] = 3                                    # drift
        else:
            out[i] = 0                                    # fixation
    return out


# --------------------------------------------------------------------------
# Optimisation baselines
# --------------------------------------------------------------------------

def exact_qap(flow: np.ndarray, dist: np.ndarray) -> Tuple[list, float, float]:
    """Exhaustive permutation search. Exact but ``O(n!)``.

    Returns ``(permutation, cost, seconds)``. Tractable only to n≈10; this is
    precisely why the QAP is a legitimate quantum target at larger n.
    """
    n = flow.shape[0]
    t0 = time.perf_counter()
    best, best_c = None, np.inf
    for perm in permutations(range(n)):
        c = float(sum(flow[i, j] * dist[perm[i], perm[j]]
                      for i in range(n) for j in range(n) if i != j))
        if c < best_c:
            best_c, best = c, list(perm)
    return best, float(best_c), time.perf_counter() - t0


def simulated_annealing_qap(flow: np.ndarray, dist: np.ndarray,
                            iters: int = 4000, seed: int = 13
                            ) -> Tuple[list, float, float]:
    """Simulated annealing — the standard classical heuristic for QAP.

    This is the baseline QAOA must actually beat to matter in practice, since
    exact search is infeasible at the scale where quantum would be interesting.
    """
    rng = np.random.default_rng(seed)
    n = flow.shape[0]
    t0 = time.perf_counter()

    def cost(p):
        return float(sum(flow[i, j] * dist[p[i], p[j]]
                         for i in range(n) for j in range(n) if i != j))

    cur = list(rng.permutation(n))
    cur_c = cost(cur)
    best, best_c = list(cur), cur_c
    for k in range(iters):
        T = max(1e-6, 1.0 * (1.0 - k / iters))
        i, j = rng.integers(0, n, 2)
        if i == j:
            continue
        cand = list(cur); cand[i], cand[j] = cand[j], cand[i]
        c = cost(cand)
        if c < cur_c or rng.random() < np.exp(-(c - cur_c) / T):
            cur, cur_c = cand, c
            if c < best_c:
                best, best_c = list(cand), c
    return best, float(best_c), time.perf_counter() - t0


def greedy_feature_selection(relevance: np.ndarray, redundancy: np.ndarray,
                             k: int) -> list:
    """Greedy mRMR — the standard classical heuristic for feature subsets."""
    d = len(relevance)
    chosen: list[int] = []
    remaining = set(range(d))
    while len(chosen) < k and remaining:
        best_i, best_score = None, -np.inf
        for i in remaining:
            red = np.mean([redundancy[i, j] for j in chosen]) if chosen else 0.0
            score = relevance[i] - red
            if score > best_score:
                best_score, best_i = score, i
        chosen.append(int(best_i)); remaining.discard(best_i)
    return sorted(chosen)


def exhaustive_feature_selection(relevance: np.ndarray, redundancy: np.ndarray,
                                 k: int) -> Tuple[list, float]:
    """Exact k-subset search — ground truth for small d."""
    from itertools import combinations
    d = len(relevance)
    best, best_v = None, np.inf
    for combo in combinations(range(d), k):
        v = -sum(relevance[i] for i in combo) + sum(
            redundancy[i, j] for a, i in enumerate(combo) for j in combo[a + 1:])
        if v < best_v:
            best_v, best = v, list(combo)
    return best, float(best_v)


# --------------------------------------------------------------------------
# Calibration baseline — the deployed algorithm
# --------------------------------------------------------------------------

def ridge_gaze_map(F: np.ndarray, t: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    """Closed-form ridge regression, matching ``iris-tracker.js::_fit``.

    ``w = (F^T F + lam I)^-1 F^T t`` — the *global optimum* of the regularised
    least-squares objective. No approximate optimiser can improve on it, which
    is why calibration fitting is deliberately excluded from the QAOA modules.
    """
    F = np.asarray(F, dtype=float)
    A = F.T @ F + lam * np.eye(F.shape[1])
    return np.linalg.solve(A, F.T @ np.asarray(t, dtype=float))


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray,
                               n_bins: int = 10) -> float:
    """ECE — how well confidence matches accuracy.

    Central to this project: the confidence gate only works if the reported
    confidence is *calibrated*. A model with high accuracy but poor calibration
    is worse for gated typing than a slightly less accurate, well-calibrated one.
    """
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            ece += (m.sum() / len(conf)) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)
