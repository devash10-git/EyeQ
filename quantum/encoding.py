"""Quantum feature-encoding strategies and their resource accounting.

Encoding is the decisive design choice in near-term QML: it fixes the qubit
count, the circuit depth, and — through the induced kernel — what the model can
actually express. Three strategies are provided, each with a documented reason
to prefer it for gaze features.

Summary
-------
=================  =========  ==================  ==========================
Strategy           Qubits     Depth (1 block)     When to prefer
=================  =========  ==================  ==========================
AngleEmbedding     d          O(1)                Low d, shallow, NISQ-safe
ZZFeatureMap       d          O(d^2) per rep      Need classically-hard kernel
AmplitudeEmbedding ceil(lg d) O(2^n) worst case   Very high d, offline only
=================  =========  ==================  ==========================

``d`` is the feature dimension. For this project ``d in {8, 13}``, so angle and
ZZ maps need 8-13 qubits — comfortably simulable — while amplitude encoding
would need only 4 qubits but pays a state-preparation depth that erases any
advantage at this scale. This asymmetry is why the experiments default to angle
and ZZ maps.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
import pennylane as qml


# --------------------------------------------------------------------------
# Resource accounting
# --------------------------------------------------------------------------

def qubits_required(n_features: int, strategy: str) -> int:
    """Qubits needed to encode ``n_features`` under a given strategy."""
    if strategy in ("angle", "zz"):
        return int(n_features)
    if strategy == "amplitude":
        return max(1, math.ceil(math.log2(max(1, n_features))))
    raise ValueError(f"unknown strategy: {strategy}")


def entangling_gate_count(n_qubits: int, strategy: str, reps: int = 1) -> int:
    """Two-qubit gate count — the quantity NISQ error budgets are spent on."""
    if strategy == "angle":
        return 0
    if strategy == "zz":
        # one ZZ interaction per feature pair, per repetition
        return reps * n_qubits * (n_qubits - 1) // 2
    if strategy == "amplitude":
        return 2 ** n_qubits  # worst-case Mottonen state preparation
    raise ValueError(f"unknown strategy: {strategy}")


# --------------------------------------------------------------------------
# Encoders — each writes gates onto the *current* circuit context
# --------------------------------------------------------------------------

def angle_embedding(x: Sequence[float], wires: Sequence[int]) -> None:
    """Single-rotation encoding: ``|x> = ⊗_j RY(x_j)|0>``.

    Depth O(1), no entanglement, hardware-cheapest. The induced kernel is a
    product of per-feature cosines, which is classically simulable — so this
    map alone cannot yield a quantum *advantage*, but it is the right control
    condition and the most NISQ-robust choice.
    """
    qml.AngleEmbedding(np.asarray(x), wires=wires, rotation="Y")


def zz_feature_map(x: Sequence[float], wires: Sequence[int], reps: int = 2) -> None:
    """Havlicek et al. (2019) ZZ feature map.

    Structure per repetition::

        H on all wires
        RZ(2*x_j) on wire j
        for each pair (i,j):  CNOT - RZ(2*(pi - x_i)(pi - x_j)) - CNOT

    The pairwise phase term is what makes the induced kernel conjectured hard to
    evaluate classically. That is the *only* reason to accept its O(d^2) depth,
    and it is why this map is used for the QSVM experiments.
    """
    x = np.asarray(x, dtype=float)
    n = len(wires)
    for _ in range(reps):
        for w in wires:
            qml.Hadamard(wires=w)
        for j, w in enumerate(wires):
            qml.RZ(2.0 * x[j], wires=w)
        for i in range(n):
            for j in range(i + 1, n):
                qml.CNOT(wires=[wires[i], wires[j]])
                qml.RZ(2.0 * (np.pi - x[i]) * (np.pi - x[j]), wires=wires[j])
                qml.CNOT(wires=[wires[i], wires[j]])


def amplitude_embedding(x: Sequence[float], wires: Sequence[int]) -> None:
    """Encode a normalised feature vector in state amplitudes.

    Exponentially qubit-efficient (``ceil(log2 d)`` qubits) but the preparation
    circuit is deep. Included for completeness and for the resource comparison;
    not used in the real-time path.
    """
    qml.AmplitudeEmbedding(np.asarray(x), wires=wires, pad_with=0.0, normalize=True)


def get_encoder(strategy: str, reps: int = 2) -> Callable[[Sequence[float], Sequence[int]], None]:
    """Return the encoder callable for a named strategy."""
    if strategy == "angle":
        return lambda x, w: angle_embedding(x, w)
    if strategy == "zz":
        return lambda x, w: zz_feature_map(x, w, reps=reps)
    if strategy == "amplitude":
        return lambda x, w: amplitude_embedding(x, w)
    raise ValueError(f"unknown strategy: {strategy}")


def describe(strategy: str, n_features: int, reps: int = 2) -> dict:
    """Resource summary for reporting tables."""
    n_q = qubits_required(n_features, strategy)
    return {
        "strategy": strategy,
        "features": n_features,
        "qubits": n_q,
        "two_qubit_gates": entangling_gate_count(n_q, strategy, reps),
        "hilbert_dim": 2 ** n_q,
        "classically_simulable_kernel": strategy == "angle",
    }
