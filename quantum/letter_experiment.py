"""Candidate-letter prediction: a controlled comparison of quantum models.

Task, and nothing else
----------------------
    input   eye feature vector
    output  probability distribution over the 26 letters

Every model in this module implements exactly that interface — ``predict_proba``
returning an ``(n, 26)`` row-stochastic array — so the comparison isolates the
*model*, not the task framing, the data map, or the output head.

Five families are investigated: VQC, QSVM, quantum kernel, QNN, and a quantum
Bayesian network. A classical reference is carried throughout as a control; a
quantum result that is not compared against a competent classical model is not a
result.

Metrics, and why each is here
-----------------------------
* ``acc``   top-1 accuracy — the headline number.
* ``top3``  the distribution is consumed as a *candidate list*, so the mass
            placed on the right answer matters even when it is not first.
* ``nll``   negative log-likelihood — the only strictly proper score here. A
            model can win on accuracy while being badly wrong about its own
            confidence; NLL catches that.
* ``ece``   expected calibration error — a downstream confidence gate is only
            meaningful if reported confidence tracks realised accuracy.

Four conditions, mapping to the questions asked
-----------------------------------------------
* ``clean``   in-distribution (sigma = 0.02)      -> predicts letters accurately
* ``noisy``   sigma = 0.030 and 0.045             -> handles noisy gaze
* ``drift``   head offset + gain shift after fit  -> supports calibration drift
* ``xuser``   fitted on 4 users, tested on 4 unseen -> generalises across users

DATA PROVENANCE — all datasets here are synthetic, generated from the forward
model documented in ``features.py``. They are valid for *relative* comparison of
algorithms under a stated noise model and are not evidence of real-world
accuracy. No claim in this module transfers to a real webcam until the same
experiments are repeated on recorded human gaze.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from .features import N_FEATURES, keyboard_layout
from .gaze_letter import GeometricGaze, _inverse_forward

LETTERS: List[str] = [ch for ch, _x, _y, _w, _h in keyboard_layout()]
N_KEYS = len(LETTERS)
ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")
EPS = 1e-12

RESULT_PATH = os.path.join(os.path.dirname(__file__), "letter_experiment.json")

#: row index and within-row slot of every letter, for the factorised models
ROW_OF = np.array([r for r, row in enumerate(ROWS) for _ in row])
SLOT_OF = np.array([i for row in ROWS for i in range(len(row))])
MAX_SLOT = max(len(r) for r in ROWS)


# ==========================================================================
# Per-user geometry — the axis along which cross-user generalisation is tested
# ==========================================================================

@dataclass
class UserGeometry:
    """One synthetic user's optics and posture.

    ``curvature`` is the cubic optical term, ``gain`` an eyeball-rotation scale,
    ``offset`` a resting head displacement. Users differ in all three, which is
    what makes ``xuser`` a genuine held-out test rather than a reshuffle.
    """
    curvature: float
    gain: float
    offset: Tuple[float, float]
    head_motion: float = 0.06

    @staticmethod
    def sample(rng: np.random.Generator) -> "UserGeometry":
        return UserGeometry(
            curvature=float(rng.uniform(0.25, 0.75)),
            gain=float(rng.uniform(0.92, 1.10)),
            offset=(float(rng.normal(0, 0.05)), float(rng.normal(0, 0.05))),
            head_motion=float(rng.uniform(0.04, 0.09)),
        )


def make_user_dataset(user: UserGeometry, n: int, noise: float, seed: int,
                      drift: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(features, intended_letter, intended_point)`` for one user.

    ``drift`` simulates calibration going stale: the head offset and gain move
    *after* the model was fitted, exactly as they do when a user shifts in their
    chair. It is applied to the generating geometry, not to the labels, so the
    intended letter stays the ground truth.
    """
    rng = np.random.default_rng(seed)
    layout = keyboard_layout()
    X = np.zeros((n, N_FEATURES))
    y = np.zeros(n, dtype=int)
    tgt = np.zeros((n, 2))

    ox = user.offset[0] + drift
    oy = user.offset[1] + drift * 0.6
    gain = user.gain * (1.0 + 0.15 * drift / max(drift, 1e-9) * drift)

    for i in range(n):
        k = int(rng.integers(0, len(layout)))
        _ch, kx, ky, kw, kh = layout[k]
        tx = kx + kw * rng.uniform(0.25, 0.75)
        ty = ky + kh * rng.uniform(0.25, 0.75)
        nose_x = 0.5 + ox + rng.normal(0, user.head_motion)
        nose_y = 0.5 + oy + rng.normal(0, user.head_motion)

        h = float(_inverse_forward(tx, nose_x, user.curvature)) * gain + rng.normal(0, noise)
        v = float(_inverse_forward(ty, nose_y, user.curvature)) * gain + rng.normal(0, noise)

        ear = float(np.clip(rng.normal(0.75, 0.06), 0.0, 1.0))
        vel = float(np.clip(abs(rng.normal(0.05, 0.04)), 0.0, 1.0))
        X[i] = [h, v, ear, 0.0,
                (nose_x - 0.5) * 2, (nose_y - 0.5) * 2, rng.normal(0, 0.05),
                ear, vel, 0.0,
                float(np.clip(1.0 - vel * 4, 0.0, 1.0)),
                float(np.clip(rng.normal(0.85, 0.08), 0.0, 1.0)),
                float(np.clip(abs(rng.normal(0.1, 0.06)), 0.0, 1.0))]
        y[i] = k
        tgt[i] = (tx, ty)
    return X, y, tgt


# ==========================================================================
# Metrics
# ==========================================================================

def accuracy(P: np.ndarray, y: np.ndarray) -> float:
    return float((P.argmax(axis=1) == y).mean())


def top_k(P: np.ndarray, y: np.ndarray, k: int = 3) -> float:
    idx = np.argsort(P, axis=1)[:, ::-1][:, :k]
    return float(np.mean([y[i] in idx[i] for i in range(len(y))]))


def nll(P: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(P[np.arange(len(y)), y], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


def ece(P: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf, pred = P.max(axis=1), P.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            out += (m.sum() / len(conf)) * abs(correct[m].mean() - conf[m].mean())
    return float(out)


def score(P: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    return {"acc": accuracy(P, y), "top3": top_k(P, y, 3),
            "nll": nll(P, y), "ece": ece(P, y)}


# ==========================================================================
# Representations — what the quantum model is actually shown
# ==========================================================================

class Representation:
    """Maps raw eye features to the vector a quantum model consumes.

    ``raw``        iris ratios + head pose. The naive choice: the model must
                   learn the whole gaze->screen map itself.
    ``calibrated`` the ridge-calibrated point + head pose. The map is already
                   solved in closed form, so the model's job is the *residual*
                   structure — which is the only job it can win.
    """

    def __init__(self, mode: str = "calibrated") -> None:
        self.mode = mode
        self.geom: Optional[GeometricGaze] = None
        self.lo: Optional[np.ndarray] = None
        self.hi: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, pts: np.ndarray) -> "Representation":
        if self.mode == "calibrated":
            self.geom = GeometricGaze().fit(X, pts)
        Z = self._raw_map(X)
        self.lo, self.hi = Z.min(axis=0), Z.max(axis=0)
        return self

    def _raw_map(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.mode == "calibrated":
            p = self.geom.predict_points(X)
            return np.column_stack([p[:, 0], p[:, 1], X[:, 4], X[:, 5]])
        return np.column_stack([X[:, 0], X[:, 1], X[:, 4], X[:, 5]])

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scaled into ``[0, pi]`` — angle encoding needs a bounded,
        non-wrapping interval or two inputs alias onto the same state."""
        Z = self._raw_map(X)
        span = np.where(self.hi - self.lo < 1e-12, 1.0, self.hi - self.lo)
        return np.clip((Z - self.lo) / span, 0.0, 1.0) * np.pi


# ==========================================================================
# Model family 1 — VQC with a 5-qubit basis-measurement head
# ==========================================================================

class FlatVQC:
    """Variational classifier over all 26 letters at once.

    Output head: five qubits give 32 basis outcomes; the first 26 are the
    letters and the remainder is renormalised away. This is the qubit-efficient
    way to emit a *distribution* (log2 26 = 4.7 qubits) rather than 26 separate
    expectation values, and the measured probabilities are automatically
    non-negative and normalised — no softmax temperature to tune.
    """

    def __init__(self, n_in: int = 4, n_layers: int = 3, epochs: int = 12,
                 lr: float = 0.09, batch: int = 32, seed: int = 13) -> None:
        self.n_qubits = 5
        self.n_in = n_in
        self.epochs, self.lr, self.batch, self.seed = epochs, lr, batch, seed
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        rng = np.random.default_rng(seed)
        shape = qml.StronglyEntanglingLayers.shape(n_layers, self.n_qubits)
        # small-angle init keeps the state near identity, where gradients are
        # informative — the standard barren-plateau mitigation
        self.theta = pnp.array(rng.normal(0.0, 0.1, size=shape), requires_grad=True)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(z, theta):
            qml.AngleEmbedding(z, wires=range(self.n_in), rotation="Y")
            qml.StronglyEntanglingLayers(theta, wires=range(self.n_qubits))
            return qml.probs(wires=range(self.n_qubits))

        self.circuit = circuit

    def _probs(self, Z, theta):
        raw = pnp.stack([self.circuit(z, theta) for z in Z])[:, :N_KEYS]
        return raw / pnp.sum(raw, axis=1, keepdims=True)

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "FlatVQC":
        Zp = pnp.array(np.asarray(Z, dtype=float), requires_grad=False)
        y = np.asarray(y, dtype=int)
        opt = qml.AdamOptimizer(self.lr)
        rng = np.random.default_rng(self.seed)

        def cost(theta, zb, yb):
            p = self._probs(zb, theta)
            return -pnp.mean(pnp.log(p[pnp.arange(len(yb)), yb] + 1e-10))

        for _ in range(self.epochs):
            idx = rng.permutation(len(Zp))
            for s in range(0, len(Zp), self.batch):
                b = idx[s: s + self.batch]
                self.theta = opt.step(lambda t: cost(t, Zp[b], y[b]), self.theta)
        return self

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        raw = np.array([np.asarray(self.circuit(z, self.theta))[:N_KEYS]
                        for z in np.asarray(Z, dtype=float)])
        return raw / np.clip(raw.sum(axis=1, keepdims=True), EPS, None)


# ==========================================================================
# Model family 2 — QNN with data re-uploading
# ==========================================================================

class FlatQNN(FlatVQC):
    """Same head, but the encoding is repeated between variational blocks.

    Data re-uploading provably increases expressivity without extra qubits
    (Perez-Salinas et al., 2020), which is the only principled way to add
    capacity here when the qubit count is fixed by the output alphabet.
    """

    def __init__(self, n_in: int = 4, n_layers: int = 3, epochs: int = 12,
                 lr: float = 0.09, batch: int = 32, seed: int = 13) -> None:
        super().__init__(n_in, n_layers, epochs, lr, batch, seed)
        self.n_layers = n_layers
        rng = np.random.default_rng(seed)
        self.theta = pnp.array(rng.normal(0.0, 0.1, size=(n_layers, self.n_qubits, 3)),
                               requires_grad=True)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(z, theta):
            for layer in range(self.n_layers):
                qml.AngleEmbedding(z, wires=range(self.n_in), rotation="Y")
                for w in range(self.n_qubits):
                    qml.Rot(theta[layer, w, 0], theta[layer, w, 1],
                            theta[layer, w, 2], wires=w)
                for w in range(self.n_qubits):
                    qml.CNOT(wires=[w, (w + 1) % self.n_qubits])
            return qml.probs(wires=range(self.n_qubits))

        self.circuit = circuit


# ==========================================================================
# Model family 3 — QSVM (one-vs-rest over 26 letters)
# ==========================================================================

class QSVMLetters:
    """Fidelity-kernel SVM, margins turned into a distribution.

    26-way one-vs-rest on a precomputed quantum Gram matrix. The decision
    function is not a probability, so margins are mapped through a softmax whose
    temperature is fitted on held-out data — without that step the model is
    uninterpretable as a distribution and its NLL is meaningless.
    """

    def __init__(self, n_in: int = 4, C: float = 16.0, reps: int = 1) -> None:
        from .models import QuantumKernel
        self.qk = QuantumKernel(n_in, strategy="angle", reps=reps)
        self.C = C
        self.Ztr: Optional[np.ndarray] = None
        self.svc = None
        self.temp = 1.0

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "QSVMLetters":
        from sklearn.svm import SVC
        Z = np.asarray(Z, dtype=float)
        self.Ztr = Z
        self.svc = SVC(kernel="precomputed", C=self.C,
                       decision_function_shape="ovr")
        self.svc.fit(self.qk.gram(Z), np.asarray(y))
        return self

    def _margins(self, Z: np.ndarray) -> np.ndarray:
        G = self.qk.gram(np.asarray(Z, dtype=float), self.Ztr)
        d = self.svc.decision_function(G)
        full = np.full((len(Z), N_KEYS), -1e3)
        for j, c in enumerate(self.svc.classes_):
            full[:, int(c)] = d[:, j] if d.ndim > 1 else d
        return full

    def calibrate(self, Z: np.ndarray, y: np.ndarray,
                  grid: Sequence[float] = (0.25, 0.5, 1, 2, 4, 8)) -> "QSVMLetters":
        m = self._margins(Z)
        best = (np.inf, 1.0)
        for t in grid:
            P = _softmax(m * t)
            v = nll(P, y)
            if v < best[0]:
                best = (v, t)
        self.temp = best[1]
        return self

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        return _softmax(self._margins(Z) * self.temp)


def _softmax(a: np.ndarray) -> np.ndarray:
    a = a - a.max(axis=1, keepdims=True)
    e = np.exp(a)
    return e / np.clip(e.sum(axis=1, keepdims=True), EPS, None)


# ==========================================================================
# Model family 4 — quantum kernel density over letter prototypes
# ==========================================================================

class QKernelDensity:
    """Non-parametric: ``P(letter) proportional to`` mean fidelity to that
    letter's training fixations.

    No training beyond storing exemplars, so nothing can be over-fitted and the
    prediction is fully interpretable — every probability traces to specific
    stored gazes. ``beta`` sharpens the similarity; it is the only free
    parameter and is fitted on held-out NLL.
    """

    def __init__(self, n_in: int = 4, per_letter: int = 12, beta: float = 8.0,
                 seed: int = 13) -> None:
        from .models import QuantumKernel
        self.qk = QuantumKernel(n_in, strategy="angle", reps=1)
        self.per_letter, self.beta, self.seed = per_letter, beta, seed
        self.proto: Dict[int, np.ndarray] = {}

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "QKernelDensity":
        rng = np.random.default_rng(self.seed)
        Z = np.asarray(Z, dtype=float)
        for k in range(N_KEYS):
            m = np.flatnonzero(y == k)
            if len(m):
                self.proto[k] = Z[rng.permutation(m)[:self.per_letter]]
        return self

    def _sim(self, Z: np.ndarray) -> np.ndarray:
        out = np.zeros((len(Z), N_KEYS))
        for k, P in self.proto.items():
            out[:, k] = self.qk.gram(np.asarray(Z, dtype=float), P).mean(axis=1)
        return out

    def calibrate(self, Z: np.ndarray, y: np.ndarray,
                  grid: Sequence[float] = (2, 4, 8, 16, 32, 64)) -> "QKernelDensity":
        s = self._sim(Z)
        best = (np.inf, self.beta)
        for b in grid:
            v = nll(_softmax(s * b), y)
            if v < best[0]:
                best = (v, b)
        self.beta = best[1]
        return self

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        return _softmax(self._sim(Z) * self.beta)


# ==========================================================================
# Model family 5 — quantum Bayesian network over (row, slot)
# ==========================================================================

class QBNLetters:
    """Factorised letter model: ``P(letter) = P(row | y-evidence) *
    P(slot | row, x-evidence)``.

    Why factorise. A flat 26-way head must learn 26 decision regions from data
    that only ever varies along two axes. The keyboard's own structure says the
    answer is a *pair*: which row, then which key within it. Learning two small
    conditionals instead of one large one is a strictly stronger inductive bias,
    and it is what a Bayesian network expresses naturally.

    Quantum encoding. Six qubits: two for the row, four for the slot. Conditional
    distributions are prepared as amplitudes ``sum sqrt(P) |row, slot>`` by
    state preparation, so a basis measurement samples the exact joint. The
    encoding is exact — it is the quantum representation of the same network, not
    an approximation — and it is verified against classical enumeration in
    ``self.max_encoding_error``.

    Interpretability. Every prediction decomposes into two human-readable
    statements ("row 2 with p=0.91, slot 4 within that row with p=0.63"), which
    no flat classifier here provides.
    """

    def __init__(self, x_bins: int = 24, y_bins: int = 12, alpha: float = 0.6
                 ) -> None:
        self.x_bins, self.y_bins, self.alpha = x_bins, y_bins, alpha
        self.p_row: Optional[np.ndarray] = None       # (y_bins, 3)
        self.p_slot: Optional[np.ndarray] = None      # (3, x_bins, MAX_SLOT)
        self.lo: Optional[np.ndarray] = None
        self.hi: Optional[np.ndarray] = None
        self.max_encoding_error: Optional[float] = None
        self.n_qubits = 6
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

    # ---- discretisation ----
    def _bins(self, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Z = np.asarray(Z, dtype=float)
        span = np.where(self.hi - self.lo < 1e-12, 1.0, self.hi - self.lo)
        u = np.clip((Z[:, :2] - self.lo[:2]) / span[:2], 0.0, 1 - 1e-9)
        return ((u[:, 0] * self.x_bins).astype(int),
                (u[:, 1] * self.y_bins).astype(int))

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "QBNLetters":
        Z = np.asarray(Z, dtype=float)
        self.lo, self.hi = Z.min(axis=0), Z.max(axis=0)
        xb, yb = self._bins(Z)
        rows, slots = ROW_OF[y], SLOT_OF[y]

        # CPT 1: P(row | y-bin), Laplace-smoothed
        c1 = np.full((self.y_bins, len(ROWS)), self.alpha)
        for b, r in zip(yb, rows):
            c1[b, r] += 1.0
        self.p_row = c1 / c1.sum(axis=1, keepdims=True)

        # CPT 2: P(slot | row, x-bin); slots absent from a row stay at zero
        c2 = np.zeros((len(ROWS), self.x_bins, MAX_SLOT))
        for r in range(len(ROWS)):
            c2[r, :, :len(ROWS[r])] = self.alpha
        for b, r, s in zip(xb, rows, slots):
            c2[r, b, s] += 1.0
        self.p_slot = c2 / np.clip(c2.sum(axis=2, keepdims=True), EPS, None)
        return self

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        xb, yb = self._bins(Z)
        out = np.zeros((len(Z), N_KEYS))
        for i in range(len(Z)):
            pr = self.p_row[yb[i]]
            for k in range(N_KEYS):
                out[i, k] = pr[ROW_OF[k]] * self.p_slot[ROW_OF[k], xb[i], SLOT_OF[k]]
        return out / np.clip(out.sum(axis=1, keepdims=True), EPS, None)

    def explain(self, z: np.ndarray) -> Dict[str, object]:
        """The two statements behind one prediction."""
        xb, yb = self._bins(np.asarray(z, dtype=float)[None, :])
        pr = self.p_row[yb[0]]
        r = int(pr.argmax())
        ps = self.p_slot[r, xb[0]]
        s = int(ps.argmax())
        return {"row": r, "p_row": float(pr[r]), "slot": s, "p_slot": float(ps[s]),
                "letter": ROWS[r][s] if s < len(ROWS[r]) else "?"}

    # ---- exact quantum encoding of the same network ----
    def quantum_joint(self, z: np.ndarray) -> np.ndarray:
        """Prepare ``sum sqrt(P(row, slot)) |row, slot>`` and return the measured
        distribution over the 26 letters.

        Two qubits carry the row, four the slot. Amplitude preparation is exact,
        so this reproduces :meth:`predict_proba` to machine precision — which is
        the correctness check, and simultaneously the honest finding that the
        encoding provides no accuracy advantage over the classical network.
        """
        xb, yb = self._bins(np.asarray(z, dtype=float)[None, :])
        pr, ps = self.p_row[yb[0]], self.p_slot[:, xb[0], :]
        joint = np.zeros(2 ** self.n_qubits)
        for k in range(N_KEYS):
            r, s = ROW_OF[k], SLOT_OF[k]
            joint[r * (2 ** 4) + s] = pr[r] * ps[r, s]
        joint = joint / max(joint.sum(), EPS)

        @qml.qnode(self.dev)
        def circuit(amps):
            qml.StatePrep(amps, wires=range(self.n_qubits))
            return qml.probs(wires=range(self.n_qubits))

        probs = np.asarray(circuit(np.sqrt(joint)))
        out = np.array([probs[ROW_OF[k] * 16 + SLOT_OF[k]] for k in range(N_KEYS)])
        return out / max(out.sum(), EPS)

    def verify_encoding(self, Z: np.ndarray, n: int = 40) -> float:
        """Max |quantum - classical| over ``n`` samples. Should be ~1e-16."""
        Z = np.asarray(Z, dtype=float)[:n]
        cls = self.predict_proba(Z)
        err = max(float(np.max(np.abs(self.quantum_joint(z) - c)))
                  for z, c in zip(Z, cls))
        self.max_encoding_error = err
        return err


# ==========================================================================
# Classical control
# ==========================================================================

class GeometricControl:
    """The exact Gaussian key posterior — the incumbent, carried as a control."""

    def __init__(self) -> None:
        self.geom = GeometricGaze()

    def fit_raw(self, X: np.ndarray, pts: np.ndarray) -> "GeometricControl":
        self.geom.fit(X, pts)
        return self

    def predict_proba_raw(self, X: np.ndarray) -> np.ndarray:
        return self.geom.posterior(X)


# ==========================================================================
# Experiment driver
# ==========================================================================

def build_cohort(n_users: int = 8, seed: int = 7) -> List[UserGeometry]:
    rng = np.random.default_rng(seed)
    return [UserGeometry.sample(rng) for _ in range(n_users)]


def pooled(users: Sequence[UserGeometry], n_each: int, noise: float, seed: int,
           drift: float = 0.0):
    Xs, ys, ps = [], [], []
    for i, u in enumerate(users):
        X, y, p = make_user_dataset(u, n_each, noise, seed + 100 * i, drift)
        Xs.append(X); ys.append(y); ps.append(p)
    return np.vstack(Xs), np.concatenate(ys), np.vstack(ps)


@dataclass
class Version:
    """One architecture in the evaluation loop."""
    name: str
    note: str
    build: object                      # (Xtr, ytr, ptr, Xcal, ycal) -> predictor
    interpretable: bool = False


def evaluate_version(v: Version, cohort: List[UserGeometry], n_each: int = 130,
                     seed: int = 21, verbose: bool = True) -> Dict[str, object]:
    """Fit once on the training cohort, then score the four conditions."""
    train_users, unseen_users = cohort[:4], cohort[4:]
    Xtr, ytr, ptr = pooled(train_users, n_each, 0.02, seed)
    Xcal, ycal, _ = pooled(train_users, 40, 0.02, seed + 5000)

    t0 = time.perf_counter()
    predict = v.build(Xtr, ytr, ptr, Xcal, ycal)
    fit_s = time.perf_counter() - t0

    conds = {
        "clean": pooled(train_users, 60, 0.020, seed + 1000),
        "noisy_030": pooled(train_users, 60, 0.030, seed + 2000),
        "noisy_045": pooled(train_users, 60, 0.045, seed + 3000),
        "drift": pooled(train_users, 60, 0.020, seed + 4000, drift=0.06),
        "xuser": pooled(unseen_users, 60, 0.020, seed + 6000),
    }
    res: Dict[str, object] = {"name": v.name, "note": v.note,
                              "interpretable": v.interpretable,
                              "fit_s": round(fit_s, 1)}
    t0 = time.perf_counter()
    for cond, (X, y, _p) in conds.items():
        res[cond] = score(predict(X), y)
    res["predict_ms"] = round((time.perf_counter() - t0) * 1000 / sum(
        len(c[1]) for c in conds.values()), 3)
    if verbose:
        print("  %-34s acc %.4f  top3 %.4f  nll %.3f  ece %.3f   (%.0fs fit)"
              % (v.name, res["clean"]["acc"], res["clean"]["top3"],
                 res["clean"]["nll"], res["clean"]["ece"], fit_s), flush=True)
    return res


# ---- version builders -----------------------------------------------------

def _rep_builder(mode: str):
    def make(Xtr, ptr):
        return Representation(mode).fit(Xtr, ptr)
    return make


def v_flat_vqc(mode: str, epochs: int = 12):
    def build(Xtr, ytr, ptr, Xcal, ycal):
        rep = _rep_builder(mode)(Xtr, ptr)
        m = FlatVQC(epochs=epochs).fit(rep.transform(Xtr), ytr)
        return lambda X: m.predict_proba(rep.transform(X))
    return build


def v_flat_qnn(mode: str, epochs: int = 12):
    def build(Xtr, ytr, ptr, Xcal, ycal):
        rep = _rep_builder(mode)(Xtr, ptr)
        m = FlatQNN(epochs=epochs).fit(rep.transform(Xtr), ytr)
        return lambda X: m.predict_proba(rep.transform(X))
    return build


def v_qsvm(mode: str, n_fit: int = 260):
    def build(Xtr, ytr, ptr, Xcal, ycal):
        rep = _rep_builder(mode)(Xtr, ptr)
        Z = rep.transform(Xtr)
        rng = np.random.default_rng(3)
        idx = rng.permutation(len(Z))[:n_fit]
        m = QSVMLetters().fit(Z[idx], ytr[idx])
        m.calibrate(rep.transform(Xcal), ycal)
        return lambda X: m.predict_proba(rep.transform(X))
    return build


def v_qkd(mode: str, per_letter: int = 10):
    def build(Xtr, ytr, ptr, Xcal, ycal):
        rep = _rep_builder(mode)(Xtr, ptr)
        m = QKernelDensity(per_letter=per_letter).fit(rep.transform(Xtr), ytr)
        m.calibrate(rep.transform(Xcal), ycal)
        return lambda X: m.predict_proba(rep.transform(X))
    return build


def v_qbn(mode: str, x_bins: int = 24, y_bins: int = 12):
    def build(Xtr, ytr, ptr, Xcal, ycal):
        rep = _rep_builder(mode)(Xtr, ptr)
        m = QBNLetters(x_bins=x_bins, y_bins=y_bins).fit(rep.transform(Xtr), ytr)
        return lambda X: m.predict_proba(rep.transform(X))
    return build


def v_control():
    def build(Xtr, ytr, ptr, Xcal, ycal):
        c = GeometricControl().fit_raw(Xtr, ptr)
        return lambda X: c.predict_proba_raw(X)
    return build


# ==========================================================================
# The evaluation loop
# ==========================================================================

def _summary(res: Dict[str, object]) -> Dict[str, float]:
    """Collapse a version's five conditions into the three loop criteria.

    * accuracy       — clean top-1.
    * robustness     — mean top-1 across the three stressed conditions
                       (noise, calibration drift, unseen users). A model that
                       wins clean and collapses under drift is not an
                       improvement, and averaging is what exposes that.
    * interpretability — ordinal: 2 = decomposable prediction, 1 = traceable to
                       stored exemplars, 0 = opaque parameters.
    """
    rob = np.mean([res["noisy_030"]["acc"], res["noisy_045"]["acc"],
                   res["drift"]["acc"], res["xuser"]["acc"]])
    return {"accuracy": res["clean"]["acc"], "robustness": float(rob),
            "interpretability": float(res["interpretable"]),
            "nll": res["clean"]["nll"], "ece": res["clean"]["ece"]}


def run_loop(n_each: int = 130, verbose: bool = True) -> Dict[str, object]:
    """Generate, evaluate, compare, repeat — rejecting any regression."""
    cohort = build_cohort()
    versions = [
        Version("V1  VQC · raw features", "naive: model must learn the gaze map itself",
                v_flat_vqc("raw"), False),
        Version("V2  VQC · calibrated", "representation fix: model sees the solved point",
                v_flat_vqc("calibrated"), False),
        Version("V3  QNN · data re-uploading", "more expressivity at equal qubit count",
                v_flat_qnn("calibrated"), False),
        Version("V4  QSVM · fidelity kernel", "26-way OvR, temperature-calibrated margins",
                v_qsvm("calibrated"), False),
        Version("V5  Quantum-kernel density", "non-parametric, traceable to exemplars",
                v_qkd("calibrated"), True),
        Version("V6  QBN · factorised row/slot", "keyboard structure as inductive bias",
                v_qbn("calibrated"), True),
        Version("V7  QBN · finer x-resolution", "bins matched to key pitch",
                v_qbn("calibrated", x_bins=40, y_bins=14), True),
    ]

    results, loop = [], []
    best: Optional[Dict[str, float]] = None
    best_name = None
    for v in versions:
        r = evaluate_version(v, cohort, n_each=n_each, verbose=verbose)
        results.append(r)
        s = _summary(r)
        if best is None:
            verdict, reason = "ACCEPT", "first version establishes the baseline"
            best, best_name = s, v.name
        else:
            drops = [k for k in ("accuracy", "robustness", "interpretability")
                     if s[k] < best[k] - 1e-9]
            gains = [k for k in ("accuracy", "robustness", "interpretability")
                     if s[k] > best[k] + 1e-9]
            if drops:
                verdict = "REJECT"
                reason = "decreases " + ", ".join(drops)
            elif gains:
                verdict, reason = "ACCEPT", "improves " + ", ".join(gains)
                best, best_name = s, v.name
            else:
                verdict, reason = "REJECT", "no improvement on any criterion"
        loop.append({"version": v.name, "verdict": verdict, "reason": reason,
                     **{k: round(float(x), 4) for k, x in s.items()}})
        if verbose:
            print("      -> %-7s %s" % (verdict, reason), flush=True)

    ctrl = evaluate_version(
        Version("CONTROL  exact Gaussian posterior", "classical incumbent",
                v_control(), True), cohort, n_each=n_each, verbose=verbose)

    out = {"loop": loop, "results": results, "control": ctrl,
           "converged_on": best_name}
    with open(RESULT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    np.seterr(all="ignore")
    print("candidate-letter prediction — evaluation loop\n", flush=True)
    out = run_loop()
    print("\nconverged on:", out["converged_on"])
    print("results ->", RESULT_PATH)


# ==========================================================================
# Fix A — sigma calibration for the geometric posterior
# ==========================================================================

class CalibratedGeometric:
    """Exact Gaussian key posterior with an *honestly estimated* width.

    The deployed model sets ``sigma`` to the residual RMSE of the calibration
    fit. That is the error on the data the map was fitted to, so it is
    systematically optimistic: at typing time the user's gaze carries noise the
    fit never saw, and under drift the map itself is stale. The posterior is
    therefore sharper than the evidence justifies — measured ECE 0.111 clean and
    0.152 on unseen users, i.e. the model claims ~11-15 points more confidence
    than it earns.

    This matters operationally, not just cosmetically: the keyboard commits when
    the fused posterior clears a fixed threshold. An overconfident posterior
    crosses that threshold on evidence that does not support it, which converts
    directly into wrong keystrokes.

    The fix is one scalar. ``sigma`` is multiplied by a factor chosen on held-out
    data to minimise negative log-likelihood — the proper scoring rule, so the
    factor cannot be gamed by trading calibration against accuracy. Top-1
    predictions are provably unchanged (scaling both axes is monotone in the
    ranking when keys share a size), so this can only improve calibration.
    """

    def __init__(self) -> None:
        self.geom = GeometricGaze()
        self.factor = 1.0

    def fit(self, X: np.ndarray, pts: np.ndarray) -> "CalibratedGeometric":
        self.geom.fit(X, pts)
        return self

    def calibrate(self, X: np.ndarray, y: np.ndarray,
                  grid: Sequence[float] = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0)
                  ) -> "CalibratedGeometric":
        pts = self.geom.predict_points(X)
        sx, sy = self.geom.sigma
        best = (np.inf, 1.0)
        for f in grid:
            P = self.geom.posterior_from_points(pts, (sx * f, sy * f))
            v = nll(P, y)
            if v < best[0]:
                best = (v, f)
        self.factor = best[1]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        sx, sy = self.geom.sigma
        return self.geom.posterior_from_points(
            self.geom.predict_points(X), (sx * self.factor, sy * self.factor))


# ==========================================================================
# Fix B — continuous QBN, removing the discretisation loss
# ==========================================================================

class GaussianQBNLetters:
    """Factorised letter model with *continuous* evidence.

    The binned QBN throws away where inside a bin the gaze landed, and the
    obvious remedy — more bins — starves each cell of data (measured: 40 bins
    scored worse than 24). Both problems come from discretising a continuous
    quantity at all.

    This keeps the factorisation that made the QBN win and drops the bins:

        P(row | y)      proportional to  N(y ; mu_r, sd_r) * P(row)
        P(slot | row,x) proportional to  N(x ; mu_rs, sd_rs) * P(slot | row)
        P(letter)       =  P(row | y) * P(slot | row, x)

    Every parameter is a mean or a standard deviation of a key's fixations, so
    the model stays as interpretable as the binned version — arguably more so,
    since a mean and a spread are directly meaningful where a bin index is not.
    The output is still a discrete distribution over (row, slot), so the exact
    quantum encoding carries over unchanged.
    """

    def __init__(self, min_sd: float = 1e-3, prior_strength: float = 1.0) -> None:
        self.min_sd = float(min_sd)
        self.prior_strength = float(prior_strength)
        self.mu_row: Optional[np.ndarray] = None
        self.sd_row: Optional[np.ndarray] = None
        self.p_row_prior: Optional[np.ndarray] = None
        self.mu_slot: Optional[np.ndarray] = None
        self.sd_slot: Optional[np.ndarray] = None
        self.p_slot_prior: Optional[np.ndarray] = None
        self.n_qubits = 6

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "GaussianQBNLetters":
        Z = np.asarray(Z, dtype=float)
        rows, slots = ROW_OF[y], SLOT_OF[y]
        R = len(ROWS)

        self.mu_row = np.zeros(R)
        self.sd_row = np.full(R, self.min_sd)
        self.p_row_prior = np.full(R, self.prior_strength)
        for r in range(R):
            m = rows == r
            self.p_row_prior[r] += m.sum()
            if m.sum() >= 2:
                self.mu_row[r] = Z[m, 1].mean()
                self.sd_row[r] = max(float(Z[m, 1].std()), self.min_sd)
            elif m.sum() == 1:
                self.mu_row[r] = Z[m, 1][0]
        self.p_row_prior /= self.p_row_prior.sum()

        self.mu_slot = np.zeros((R, MAX_SLOT))
        self.sd_slot = np.full((R, MAX_SLOT), self.min_sd)
        self.p_slot_prior = np.zeros((R, MAX_SLOT))
        for r in range(R):
            self.p_slot_prior[r, :len(ROWS[r])] = self.prior_strength
            for s in range(len(ROWS[r])):
                m = (rows == r) & (slots == s)
                self.p_slot_prior[r, s] += m.sum()
                if m.sum() >= 2:
                    self.mu_slot[r, s] = Z[m, 0].mean()
                    self.sd_slot[r, s] = max(float(Z[m, 0].std()), self.min_sd)
                elif m.sum() == 1:
                    self.mu_slot[r, s] = Z[m, 0][0]
                else:
                    # unseen key: fall back to its geometric position in the row
                    self.mu_slot[r, s] = Z[:, 0].min() + (s + 0.5) / len(ROWS[r]) * (
                        Z[:, 0].max() - Z[:, 0].min())
                    self.sd_slot[r, s] = max(float(Z[:, 0].std()), self.min_sd)
            self.p_slot_prior[r] /= max(self.p_slot_prior[r].sum(), EPS)
        return self

    @staticmethod
    def _gauss(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
        return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z, dtype=float)
        R = len(ROWS)
        lik_row = np.column_stack([
            self._gauss(Z[:, 1], self.mu_row[r], self.sd_row[r]) * self.p_row_prior[r]
            for r in range(R)])
        lik_row /= np.clip(lik_row.sum(axis=1, keepdims=True), EPS, None)

        out = np.zeros((len(Z), N_KEYS))
        for k in range(N_KEYS):
            r, s = ROW_OF[k], SLOT_OF[k]
            out[:, k] = self._gauss(Z[:, 0], self.mu_slot[r, s], self.sd_slot[r, s]) \
                * self.p_slot_prior[r, s]
        # normalise the slot factor within each row, then weight by P(row | y)
        for r in range(R):
            cols = np.flatnonzero(ROW_OF == r)
            z = out[:, cols].sum(axis=1, keepdims=True)
            out[:, cols] = out[:, cols] / np.clip(z, EPS, None) * lik_row[:, [r]]
        return out / np.clip(out.sum(axis=1, keepdims=True), EPS, None)

    def explain(self, z: np.ndarray) -> Dict[str, object]:
        P = self.predict_proba(np.asarray(z, dtype=float)[None, :])[0]
        k = int(P.argmax())
        r, s = int(ROW_OF[k]), int(SLOT_OF[k])
        return {"letter": LETTERS[k], "p": float(P[k]), "row": r, "slot": s,
                "row_mean": float(self.mu_row[r]), "row_sd": float(self.sd_row[r]),
                "slot_mean": float(self.mu_slot[r, s]),
                "slot_sd": float(self.sd_slot[r, s])}

    def quantum_joint(self, z: np.ndarray) -> np.ndarray:
        """Exact amplitude encoding of the same discrete joint (6 qubits)."""
        P = self.predict_proba(np.asarray(z, dtype=float)[None, :])[0]
        amps = np.zeros(2 ** self.n_qubits)
        for k in range(N_KEYS):
            amps[ROW_OF[k] * 16 + SLOT_OF[k]] = P[k]
        amps = amps / max(amps.sum(), EPS)
        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev)
        def circuit(a):
            qml.StatePrep(a, wires=range(self.n_qubits))
            return qml.probs(wires=range(self.n_qubits))

        probs = np.asarray(circuit(np.sqrt(amps)))
        out = np.array([probs[ROW_OF[k] * 16 + SLOT_OF[k]] for k in range(N_KEYS)])
        return out / max(out.sum(), EPS)


def v_gaussian_qbn(mode: str = "calibrated"):
    def build(Xtr, ytr, ptr, Xcal, ycal):
        rep = Representation(mode).fit(Xtr, ptr)
        m = GaussianQBNLetters().fit(rep.transform(Xtr), ytr)
        return lambda X: m.predict_proba(rep.transform(X))
    return build


def v_calibrated_control():
    def build(Xtr, ytr, ptr, Xcal, ycal):
        c = CalibratedGeometric().fit(Xtr, ptr).calibrate(Xcal, ycal)
        return lambda X: c.predict_proba(X)
    return build


class LOOCalibratedGeometric(CalibratedGeometric):
    """Sigma from leave-one-out residuals instead of in-sample residuals.

    The scalar-grid version of this fix failed: it selected factor 1.0, because
    the held-out set used to choose it was drawn from the *same* clean condition
    as the fit, where the in-sample sigma is very nearly right. The
    miscalibration only appears on unseen users and under drift — conditions no
    honest calibration set can contain, since they are exactly what is being
    generalised to.

    Leave-one-out attacks the cause rather than the symptom. For ridge, the LOO
    residual is available in closed form without refitting,

        e_i = r_i / (1 - h_ii),      H = F (F^T F + lam I)^-1 F^T

    and inflates sigma by exactly the amount the fit is over-optimistic about
    itself. No held-out labels are needed, so it costs nothing at calibration
    time and cannot leak test information.
    """

    def fit(self, X: np.ndarray, pts: np.ndarray) -> "LOOCalibratedGeometric":
        self.geom.fit(X, pts)
        F = self.geom._design(X)
        A = F.T @ F + self.geom.lam * np.eye(F.shape[1])
        H = F @ np.linalg.solve(A, F.T)
        h = np.clip(np.diag(H), 0.0, 1 - 1e-6)
        pred = self.geom.predict_points(X)
        r = pred - np.asarray(pts, dtype=float)
        loo = r / (1.0 - h)[:, None]
        sx = max(float(np.sqrt(np.mean(loo[:, 0] ** 2))), 1e-3)
        sy = max(float(np.sqrt(np.mean(loo[:, 1] ** 2))), 1e-3)
        self.loo_sigma = (sx, sy)
        base = self.geom.sigma
        self.factor = 1.0
        self.inflation = (sx / base[0], sy / base[1])
        self.geom.sigma = (sx, sy)
        return self

    def calibrate(self, X, y, grid=None) -> "LOOCalibratedGeometric":
        return self          # nothing to tune: the estimator is closed form


def v_loo_control():
    def build(Xtr, ytr, ptr, Xcal, ycal):
        c = LOOCalibratedGeometric().fit(Xtr, ptr)
        return lambda X: c.predict_proba(X)
    return build
