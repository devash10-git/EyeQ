"""Quantum learning models: fidelity-kernel QSVM, VQC, and QNN.

All three share the same encoding layer (``encoding.py``) so that comparisons
isolate the *model*, not the data map.

Design notes
------------
**Barren plateaus.** Variational circuits suffer gradients that vanish
exponentially in qubit count and depth (McClean et al., 2018). Two mitigations
are applied throughout:

1. *Shallow, structured ansatz* — hardware-efficient layers with a ring of
   entanglers, kept to <= 4 layers.
2. *Small-angle initialisation* — parameters drawn from N(0, 0.1) rather than
   uniform on [0, 2pi), which keeps the initial state near the identity where
   gradients are informative.

**Why analytic simulation.** ``default.qubit`` evaluates expectation values
exactly (infinite-shot limit). This isolates *algorithmic* capability from shot
noise; the shot-noise and depolarising-noise sweeps in ``benchmark.py`` then
quantify how much of that capability survives on hardware. Reporting only the
analytic number would overstate NISQ performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from .encoding import get_encoder, qubits_required


# ==========================================================================
# 1. Quantum kernel / QSVM  — replaces the classical RBF-SVM
# ==========================================================================

class QuantumKernel:
    """Fidelity kernel ``K(x,z) = |<phi(x)|phi(z)>|^2``.

    Computed by the standard *overlap* (compute-uncompute) circuit: encode
    ``x``, apply the adjoint encoding of ``z``, and measure the probability of
    returning to the all-zero state.

    Complexity: evaluating a full ``n x n`` Gram matrix costs ``O(n^2)`` circuit
    executions. This quadratic cost in *dataset size* is the practical
    bottleneck of kernel QML and is reported in the benchmarks.
    """

    def __init__(self, n_features: int, strategy: str = "zz", reps: int = 2,
                 shots: Optional[int] = None) -> None:
        self.n_features = int(n_features)
        self.strategy = strategy
        self.n_qubits = qubits_required(n_features, strategy)
        self.wires = list(range(self.n_qubits))
        self.encode = get_encoder(strategy, reps=reps)
        self.shots = shots
        self.dev = qml.device("default.qubit", wires=self.n_qubits, shots=shots)
        self._circuit = qml.QNode(self._overlap, self.dev)
        self.n_evaluations = 0

    def _overlap(self, x1, x2):
        self.encode(x1, self.wires)
        qml.adjoint(self.encode)(x2, self.wires)
        return qml.probs(wires=self.wires)

    def kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        self.n_evaluations += 1
        return float(self._circuit(x1, x2)[0])

    def gram(self, A: np.ndarray, B: Optional[np.ndarray] = None) -> np.ndarray:
        """Gram matrix. Exploits symmetry when ``B is None`` (halves the cost)."""
        A = np.asarray(A, dtype=float)
        if B is None:
            n = len(A)
            K = np.eye(n)
            for i in range(n):
                for j in range(i + 1, n):
                    K[i, j] = K[j, i] = self.kernel(A[i], A[j])
            return K
        B = np.asarray(B, dtype=float)
        return np.array([[self.kernel(a, b) for b in B] for a in A])


class QSVM:
    """SVM with a precomputed quantum fidelity kernel.

    Replaces: classical SVM with RBF kernel (feature classification).
    Rationale: if the gaze feature space contains structure whose natural
    similarity measure is hard to compute classically, a quantum kernel can
    separate it with fewer support vectors. Whether such structure *exists* in
    gaze data is precisely the empirical question the benchmark answers.
    """

    def __init__(self, n_features: int, strategy: str = "zz", reps: int = 2,
                 C: float = 1.0, shots: Optional[int] = None) -> None:
        from sklearn.svm import SVC
        self.qk = QuantumKernel(n_features, strategy, reps, shots)
        self.svc = SVC(kernel="precomputed", C=C, probability=False)
        self._train_X: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QSVM":
        X = np.asarray(X, dtype=float)
        self._train_X = X
        self.svc.fit(self.qk.gram(X), np.asarray(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._train_X is None:
            raise RuntimeError("QSVM.fit must be called first")
        return self.svc.predict(self.qk.gram(np.asarray(X, dtype=float), self._train_X))

    def decision_margin(self, X: np.ndarray) -> np.ndarray:
        """Signed distance to the separating hyperplane (a confidence proxy)."""
        return self.svc.decision_function(
            self.qk.gram(np.asarray(X, dtype=float), self._train_X)
        )


# ==========================================================================
# 2. Variational Quantum Classifier — replaces the key-prediction MLP
# ==========================================================================

@dataclass
class VQCConfig:
    n_features: int
    n_classes: int
    n_layers: int = 3
    strategy: str = "angle"
    reps: int = 2
    lr: float = 0.08
    epochs: int = 40
    batch_size: int = 24
    seed: int = 13


class VQC:
    """Variational Quantum Classifier with a softmax head and confidence output.

    Circuit
    -------
    ``|0>^n -> Encode(x) -> [StronglyEntanglingLayers(theta)] -> <Z_j>``

    The per-qubit Pauli-Z expectations form a logit vector; a classical softmax
    turns it into a calibrated-ish distribution over classes. Training minimises
    cross-entropy with Adam via the parameter-shift rule, so the same code path
    runs unchanged on hardware.

    Replaces: a small classical MLP over gaze features.
    Qubits: ``max(n_features, n_classes)`` for angle encoding.
    Depth: ``O(n_layers * n_qubits)`` two-qubit gates.
    """

    def __init__(self, cfg: VQCConfig) -> None:
        self.cfg = cfg
        self.n_qubits = max(qubits_required(cfg.n_features, cfg.strategy), cfg.n_classes)
        self.wires = list(range(self.n_qubits))
        self.encode = get_encoder(cfg.strategy, reps=cfg.reps)
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        rng = np.random.default_rng(cfg.seed)
        shape = qml.StronglyEntanglingLayers.shape(cfg.n_layers, self.n_qubits)
        # small-angle init: stay near identity to avoid the barren plateau
        self.theta = pnp.array(rng.normal(0.0, 0.1, size=shape), requires_grad=True)
        self.qnode = qml.QNode(self._circuit, self.dev, interface="autograd")
        self.history: list[float] = []

    def _circuit(self, x, theta):
        padded = pnp.concatenate([x, pnp.zeros(self.n_qubits - len(x))]) \
            if len(x) < self.n_qubits else x[: self.n_qubits]
        self.encode(padded, self.wires)
        qml.StronglyEntanglingLayers(theta, wires=self.wires)
        return [qml.expval(qml.PauliZ(w)) for w in range(self.cfg.n_classes)]

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    def _logits(self, X):
        return pnp.stack([pnp.stack(self.qnode(x, self.theta)) for x in X])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VQC":
        X = pnp.array(np.asarray(X, dtype=float), requires_grad=False)
        y = np.asarray(y, dtype=int)
        opt = qml.AdamOptimizer(self.cfg.lr)
        rng = np.random.default_rng(self.cfg.seed)
        n = len(X)

        def cost(theta, xb, yb):
            logits = pnp.stack([pnp.stack(self.qnode(x, theta)) for x in xb])
            # temperature 4.0 widens the [-1,1] expectation range into usable logits
            shifted = logits * 4.0
            lse = pnp.log(pnp.sum(pnp.exp(shifted - pnp.max(shifted, axis=1, keepdims=True)), axis=1))
            picked = pnp.stack([shifted[i, yb[i]] - pnp.max(shifted[i]) for i in range(len(yb))])
            return -pnp.mean(picked - lse)

        for epoch in range(self.cfg.epochs):
            idx = rng.permutation(n)
            for s in range(0, n, self.cfg.batch_size):
                b = idx[s: s + self.cfg.batch_size]
                self.theta = opt.step(lambda t: cost(t, X[b], y[b]), self.theta)
            if epoch % 5 == 0:
                self.history.append(float(cost(self.theta, X[:64], y[:64])))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        logits = np.array([[float(v) for v in self.qnode(x, self.theta)] for x in X])
        return self._softmax(logits * 4.0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def confidence(self, X: np.ndarray) -> np.ndarray:
        """Max posterior — feeds the classical confidence gate."""
        return self.predict_proba(X).max(axis=1)


# ==========================================================================
# 3. Quantum Neural Network — eye-movement state recognition
# ==========================================================================

@dataclass
class QNNConfig:
    n_features: int
    n_classes: int = 4
    n_layers: int = 2
    strategy: str = "angle"
    lr: float = 0.1
    epochs: int = 35
    batch_size: int = 32
    seed: int = 13
    #: data re-uploading repeats the encoding between variational blocks,
    #: which provably increases expressivity without extra qubits
    #: (Perez-Salinas et al., 2020).
    reupload: bool = True


class QNN:
    """Parameterized QNN with data re-uploading for eye-movement classification.

    Task: classify {fixation, saccade, blink, drift} from kinematic features.
    Replaces: a classical MLP / velocity-threshold (I-VT) classifier.

    Why this task suits a QNN better than key prediction: it is genuinely
    low-dimensional (4-6 informative features), the classes are nonlinearly
    separable in those features, and the decision is *not* on the 33 ms
    real-time path — it can run at fixation boundaries.
    """

    def __init__(self, cfg: QNNConfig) -> None:
        self.cfg = cfg
        self.n_qubits = max(qubits_required(cfg.n_features, cfg.strategy), cfg.n_classes)
        self.wires = list(range(self.n_qubits))
        self.encode = get_encoder(cfg.strategy)
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        rng = np.random.default_rng(cfg.seed)
        shape = (cfg.n_layers, self.n_qubits, 3)
        self.theta = pnp.array(rng.normal(0.0, 0.1, size=shape), requires_grad=True)
        self.qnode = qml.QNode(self._circuit, self.dev, interface="autograd")

    def _circuit(self, x, theta):
        padded = pnp.concatenate([x, pnp.zeros(self.n_qubits - len(x))]) \
            if len(x) < self.n_qubits else x[: self.n_qubits]
        for layer in range(self.cfg.n_layers):
            if layer == 0 or self.cfg.reupload:
                self.encode(padded, self.wires)
            for w in range(self.n_qubits):
                qml.Rot(theta[layer, w, 0], theta[layer, w, 1], theta[layer, w, 2], wires=w)
            for w in range(self.n_qubits):                      # ring entangler
                qml.CNOT(wires=[w, (w + 1) % self.n_qubits])
        return [qml.expval(qml.PauliZ(w)) for w in range(self.cfg.n_classes)]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QNN":
        X = pnp.array(np.asarray(X, dtype=float), requires_grad=False)
        y = np.asarray(y, dtype=int)
        opt = qml.AdamOptimizer(self.cfg.lr)
        rng = np.random.default_rng(self.cfg.seed)

        def cost(theta, xb, yb):
            logits = pnp.stack([pnp.stack(self.qnode(x, theta)) for x in xb]) * 4.0
            mx = pnp.max(logits, axis=1, keepdims=True)
            lse = pnp.log(pnp.sum(pnp.exp(logits - mx), axis=1))
            picked = pnp.stack([logits[i, yb[i]] - mx[i, 0] for i in range(len(yb))])
            return -pnp.mean(picked - lse)

        for _ in range(self.cfg.epochs):
            idx = rng.permutation(len(X))
            for s in range(0, len(X), self.cfg.batch_size):
                b = idx[s: s + self.cfg.batch_size]
                self.theta = opt.step(lambda t: cost(t, X[b], y[b]), self.theta)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        logits = np.array([[float(v) for v in self.qnode(x, self.theta)] for x in X]) * 4.0
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


def circuit_resources(model) -> dict:
    """Qubit / gate accounting for the results table."""
    if isinstance(model, QSVM):
        n_q = model.qk.n_qubits
        return {"model": "QSVM", "qubits": n_q, "kernel_evals": model.qk.n_evaluations}
    if isinstance(model, VQC):
        n_q = model.n_qubits
        return {"model": "VQC", "qubits": n_q,
                "parameters": int(np.prod(model.theta.shape)),
                "two_qubit_gates": model.cfg.n_layers * n_q}
    if isinstance(model, QNN):
        n_q = model.n_qubits
        return {"model": "QNN", "qubits": n_q,
                "parameters": int(np.prod(model.theta.shape)),
                "two_qubit_gates": model.cfg.n_layers * n_q}
    return {}
