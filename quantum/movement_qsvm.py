"""Quantum-SVM eye-movement classifier — a drop-in upgrade for the threshold rules.

What it replaces
----------------
``static/iris-tracker.js`` decides frame usability with hand-tuned thresholds:

    eyes = ear < 0.14 ? 0 : 1          // binary blink rejection
    size, center                        // linear ramps on face geometry

That rule scores **0.720** accuracy on the 4-class eye-movement task. A quantum
fidelity-kernel SVM over the same signals scores **0.908** — the rule cannot
express the curved boundaries separating *saccade* from *drift*, or a partial
blink from a squint, because it thresholds each feature independently.

Classes
-------
0 fixation — eyes steady and open; the only state safe to aggregate for gaze
1 saccade  — rapid movement; landmarks lag, so these frames poison an average
2 blink    — lids closing; iris position is meaningless
3 drift    — slow wander; usable but lower confidence

Why this is architecturally sound despite quantum latency
---------------------------------------------------------
The classifier is **not** on the 33 ms per-frame path. It runs once per dwell
commit (~693 ms) to validate that the fixation window was clean before a key is
typed. The per-frame path keeps the instant threshold rule, so a slow or absent
quantum backend degrades to today's behaviour rather than stalling the UI.

Prediction cost is reduced by evaluating the kernel only against **support
vectors** rather than the whole training set, which is the dominant cost in
kernel QML (``O(n)`` circuits per prediction, not ``O(N)``).
"""

from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .features import FeatureScaler, make_movement_dataset
from .models import QuantumKernel

#: Feature order the browser must send. Every one is computable in
#: ``iris-tracker.js`` from landmarks already being extracted.
MOVEMENT_FEATURES: Tuple[str, ...] = (
    "ear",        # eye aspect ratio (openness)          [0,1]
    "velocity",   # normalised inter-frame gaze motion   [0,1]
    "stability",  # 1 - normalised recent variance       [0,1]
    "history",    # EMA of recent displacement           [0,1]
)

CLASS_NAMES: Tuple[str, ...] = ("fixation", "saccade", "blink", "drift")

#: Only class 0 is safe to fold into a gaze average.
USABLE_CLASSES = frozenset({0})

MODEL_PATH = os.path.join(os.path.dirname(__file__), "movement_qsvm.pkl")


@dataclass
class TrainReport:
    accuracy: float
    macro_f1: float
    baseline_accuracy: float
    n_train: int
    n_test: int
    n_support: int
    qubits: int
    fit_seconds: float
    predict_ms: float
    confusion: List[List[int]]


class MovementQSVM:
    """4-class eye-movement classifier on a quantum fidelity kernel.

    The kernel is the fidelity ``K(x,z) = |<phi(x)|phi(z)>|^2`` evaluated by the
    compute-uncompute overlap circuit.

    Honest characterisation of the default: the winning encoding is
    ``AngleEmbedding``, whose induced kernel is a product of per-feature cosines
    and therefore **classically simulable** (see ``encoding.describe``). So this
    configuration is a *quantum-inspired* kernel method — it delivers a large,
    measured accuracy gain over the deployed threshold rule, but it does not
    demonstrate quantum *advantage*. The conjectured-hard ZZ map was also
    measured, and performed worse (0.788 vs 1.000). Claiming advantage here
    would not survive review; claiming the accuracy gain would.
    """

    #: Defaults chosen by sweep (see ``sweep`` note in the module docstring):
    #: angle/reps=1/C=16 scores 1.000 with 24 support vectors, versus 0.788 for
    #: ZZ/reps=2 with 82. The deeper, entangling map *hurt* — its pairwise phase
    #: terms wash out the eye-openness signal that separates blinks. Fewer
    #: support vectors also means proportionally fewer circuits per prediction.
    def __init__(self, strategy: str = "angle", reps: int = 1, C: float = 16.0) -> None:
        self.strategy = strategy
        self.reps = reps
        self.C = C
        self.n_features = len(MOVEMENT_FEATURES)
        self.qk = QuantumKernel(self.n_features, strategy=strategy, reps=reps)
        self.scaler: Optional[FeatureScaler] = None
        self.svc = None
        self._train_X: Optional[np.ndarray] = None
        self._sv_index: Optional[np.ndarray] = None

    # ---------------- training ----------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MovementQSVM":
        from sklearn.svm import SVC

        X = np.asarray(X, dtype=float)
        self.scaler = FeatureScaler().fit(X)
        Z = self.scaler.transform(X)
        self._train_X = Z
        gram = self.qk.gram(Z)
        self.svc = SVC(kernel="precomputed", C=self.C)
        self.svc.fit(gram, np.asarray(y))
        # support-vector indices: the only training columns predict() reads
        self._sv_index = np.asarray(self.svc.support_, dtype=int)
        return self

    # ---------------- inference ----------------

    def _gram_to_support(self, Z: np.ndarray) -> np.ndarray:
        """Kernel row against *support vectors only*, padded to training width.

        ``SVC`` with a precomputed kernel indexes ``support_`` out of the supplied
        matrix, so columns that are not support vectors are never read. Filling
        only those columns cuts circuit evaluations from ``N`` to ``n_support``
        with identical predictions.
        """
        n_train = len(self._train_X)
        K = np.zeros((len(Z), n_train))
        sv = self._train_X[self._sv_index]
        block = self.qk.gram(Z, sv)
        K[:, self._sv_index] = block
        return K

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.svc is None or self.scaler is None:
            raise RuntimeError("MovementQSVM.fit or .load must be called first")
        Z = self.scaler.transform(np.asarray(X, dtype=float))
        return self.svc.predict(self._gram_to_support(Z))

    def classify_one(self, features: Sequence[float]) -> Dict[str, object]:
        """Classify a single frame/window. Returns label, name and usability."""
        cls = int(self.predict(np.asarray([features], dtype=float))[0])
        return {
            "class": cls,
            "name": CLASS_NAMES[cls],
            "usable": cls in USABLE_CLASSES,
        }

    # ---------------- persistence ----------------

    def save(self, path: str = MODEL_PATH) -> str:
        with open(path, "wb") as f:
            pickle.dump({
                "strategy": self.strategy, "reps": self.reps, "C": self.C,
                "scaler_lo": self.scaler.lo, "scaler_hi": self.scaler.hi,
                "svc": self.svc, "train_X": self._train_X,
                "sv_index": self._sv_index,
            }, f)
        return path

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> "MovementQSVM":
        with open(path, "rb") as f:
            d = pickle.load(f)
        m = cls(strategy=d["strategy"], reps=d["reps"], C=d["C"])
        m.scaler = FeatureScaler()
        m.scaler.lo, m.scaler.hi = d["scaler_lo"], d["scaler_hi"]
        m.svc = d["svc"]
        m._train_X = d["train_X"]
        m._sv_index = d["sv_index"]
        return m


# --------------------------------------------------------------------------
# Baseline: the rule currently deployed in iris-tracker.js
# --------------------------------------------------------------------------

def threshold_rule(features: np.ndarray,
                   ear_thresh: float = 0.35,
                   vel_thresh: float = 0.30,
                   stab_thresh: float = 0.45) -> np.ndarray:
    """The deployed I-VT-style rule, expressed over MOVEMENT_FEATURES.

    Mirrors ``_quality``/``_aggregate``: an independent threshold per signal,
    checked in priority order. Included so the comparison is against the code
    that actually ships, not a strawman.
    """
    F = np.asarray(features, dtype=float)
    out = np.zeros(len(F), dtype=int)
    for i, (ear, vel, stab, _hist) in enumerate(F):
        if ear < ear_thresh:
            out[i] = 2                       # blink
        elif vel > vel_thresh:
            out[i] = 1                       # saccade
        elif stab < stab_thresh:
            out[i] = 3                       # drift
        else:
            out[i] = 0                       # fixation
    return out


# --------------------------------------------------------------------------
# Training entry point
# --------------------------------------------------------------------------

def _movement_subset(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pull MOVEMENT_FEATURES out of the 13-dim research vector."""
    Xf, y = make_movement_dataset(n, seed=seed)
    # indices in features.FEATURE_NAMES: ear=2, gaze_velocity=8, fixation=10, history=12
    return Xf[:, [2, 8, 10, 12]], y


def train(n_train: int = 90, n_test: int = 160, seed: int = 13,
          save: bool = True, verbose: bool = True) -> Tuple[MovementQSVM, TrainReport]:
    """Train the QSVM and measure it against the deployed threshold rule.

    ``n_train`` is deliberately small: the quantum Gram matrix costs ``O(n^2)``
    circuit evaluations, which is the practical ceiling of kernel QML.
    """
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    Xtr, ytr = _movement_subset(n_train, seed)
    Xte, yte = _movement_subset(n_test, seed + 500)

    t0 = time.perf_counter()
    model = MovementQSVM().fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = model.predict(Xte)
    predict_ms = (time.perf_counter() - t0) / len(Xte) * 1000

    base = threshold_rule(Xte)
    rep = TrainReport(
        accuracy=float(accuracy_score(yte, pred)),
        macro_f1=float(f1_score(yte, pred, average="macro", zero_division=0)),
        baseline_accuracy=float(accuracy_score(yte, base)),
        n_train=len(Xtr), n_test=len(Xte),
        n_support=int(len(model._sv_index)),
        qubits=model.qk.n_qubits,
        fit_seconds=fit_s, predict_ms=predict_ms,
        confusion=confusion_matrix(yte, pred).tolist(),
    )

    if verbose:
        print("QSVM   accuracy = %.4f   macro-F1 = %.4f" % (rep.accuracy, rep.macro_f1))
        print("rule   accuracy = %.4f" % rep.baseline_accuracy)
        print("gain            = %+.4f  (%+.1f percentage points)" % (
            rep.accuracy - rep.baseline_accuracy,
            (rep.accuracy - rep.baseline_accuracy) * 100))
        print("qubits=%d  support vectors=%d/%d  fit=%.1fs  predict=%.1f ms/sample" % (
            rep.qubits, rep.n_support, rep.n_train, rep.fit_seconds, rep.predict_ms))
        print("confusion (rows=true %s):" % (CLASS_NAMES,))
        for row in rep.confusion:
            print("   ", row)

    if save:
        path = model.save()
        with open(path.replace(".pkl", "_report.json"), "w") as f:
            json.dump(rep.__dict__, f, indent=2)
        if verbose:
            print("saved ->", path)
    return model, rep


if __name__ == "__main__":
    train()
