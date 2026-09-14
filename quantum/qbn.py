"""Quantum Bayesian Network for uncertainty estimation in gaze typing.

What a QBN is
-------------
A discrete Bayesian network is encoded by mapping each binary node to a qubit
and each conditional probability table (CPT) entry to a controlled rotation:

    P(X = 1 | parents = s)  ->  multi-controlled RY(2*arcsin(sqrt(p)))

Applying these in topological order prepares

    |psi> = sum_x sqrt(P(x)) |x>

so that measuring in the computational basis samples exactly from the joint
distribution. This construction is exact — it is the quantum representation of
the same network, not an approximation.

Where the quantum part could matter
-----------------------------------
Marginal/conditional inference by *rejection sampling* needs ``O(1/P(e))``
samples classically for evidence ``e``. Amplitude amplification reduces this to
``O(1/sqrt(P(e)))`` (Low, Yoder & Chuang, 2014) — a quadratic speedup that
matters when evidence is rare.

Honest assessment for this project: the gaze network below has 5 binary nodes,
so exact classical inference is a 32-row table enumeration costing microseconds.
The quadratic sampling speedup is real but applies to a bottleneck this system
does not have. The QBN is therefore implemented and validated for correctness,
and reported as *architecturally sound but not accuracy-improving at this
scale* — which is the finding, not a failure.

Network
-------
    Blink -> EyeClosed
    Blink -> Unreliable
    Saccade -> Unreliable
    Unreliable -> MisKey        (probability the committed key is wrong)

This models the causal story the tracker already assumes: blinks and saccades
corrupt the landmark signal, and a corrupted signal raises mis-key risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pennylane as qml


@dataclass
class Node:
    """A binary Bayesian-network node.

    Attributes:
        name: node label.
        parents: parent node names (must precede this node topologically).
        cpt: ``P(node = 1 | parent assignment)``, keyed by the parent tuple.
    """
    name: str
    parents: List[str] = field(default_factory=list)
    cpt: Dict[Tuple[int, ...], float] = field(default_factory=dict)

    def prob(self, parent_state: Tuple[int, ...]) -> float:
        return float(self.cpt[parent_state])


def default_gaze_network() -> List[Node]:
    """The 5-node gaze-reliability network described in the module docstring."""
    return [
        Node("blink", [], {(): 0.06}),
        Node("saccade", [], {(): 0.18}),
        Node("eye_closed", ["blink"], {(0,): 0.02, (1,): 0.93}),
        Node("unreliable", ["blink", "saccade"],
             {(0, 0): 0.05, (0, 1): 0.55, (1, 0): 0.80, (1, 1): 0.92}),
        Node("mis_key", ["unreliable"], {(0,): 0.03, (1,): 0.46}),
    ]


class QuantumBayesianNetwork:
    """Exact quantum encoding of a discrete Bayesian network.

    One qubit per node; CPTs become (multi-)controlled RY rotations. The
    prepared state's basis-measurement distribution equals the network's joint
    distribution, which is asserted against classical enumeration in the tests.
    """

    def __init__(self, nodes: Optional[Sequence[Node]] = None) -> None:
        self.nodes: List[Node] = list(nodes) if nodes else default_gaze_network()
        self.index = {n.name: i for i, n in enumerate(self.nodes)}
        self.n_qubits = len(self.nodes)
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self._probs = qml.QNode(self._circuit, self.dev)

    # ---------------- circuit ----------------

    @staticmethod
    def _theta(p: float) -> float:
        """Rotation angle placing amplitude ``sqrt(p)`` on |1>."""
        p = float(np.clip(p, 0.0, 1.0))
        return 2.0 * float(np.arcsin(np.sqrt(p)))

    def _circuit(self):
        for node in self.nodes:
            w = self.index[node.name]
            if not node.parents:
                qml.RY(self._theta(node.prob(())), wires=w)
                continue
            pwires = [self.index[p] for p in node.parents]
            # one multi-controlled rotation per parent assignment; X-conjugation
            # selects the control pattern for zero-valued parents
            for assignment in product([0, 1], repeat=len(pwires)):
                theta = self._theta(node.prob(assignment))
                if abs(theta) < 1e-12:
                    continue
                flips = [pw for pw, bit in zip(pwires, assignment) if bit == 0]
                for f in flips:
                    qml.PauliX(wires=f)
                qml.ctrl(qml.RY, control=pwires)(theta, wires=w)
                for f in flips:
                    qml.PauliX(wires=f)
        return qml.probs(wires=range(self.n_qubits))

    # ---------------- inference ----------------

    def joint(self) -> np.ndarray:
        """Full joint distribution over the ``2^n`` assignments."""
        return np.asarray(self._probs())

    def _states(self) -> np.ndarray:
        n = self.n_qubits
        return np.array([[(m >> (n - 1 - k)) & 1 for k in range(n)] for m in range(2 ** n)])

    def marginal(self, name: str) -> float:
        """``P(name = 1)``."""
        i = self.index[name]
        return float(self.joint()[self._states()[:, i] == 1].sum())

    def conditional(self, target: str, evidence: Dict[str, int]) -> float:
        """``P(target = 1 | evidence)`` by conditioning the joint distribution."""
        probs, states = self.joint(), self._states()
        mask = np.ones(len(states), dtype=bool)
        for k, v in evidence.items():
            mask &= states[:, self.index[k]] == v
        denom = probs[mask].sum()
        if denom < 1e-15:
            return float("nan")
        return float(probs[mask & (states[:, self.index[target]] == 1)].sum() / denom)

    def typing_confidence(self, blink: int, saccade: int) -> float:
        """``P(key correct | blink, saccade)`` — feeds the classical gate."""
        return 1.0 - self.conditional("mis_key", {"blink": blink, "saccade": saccade})

    # ---------------- resources ----------------

    def resources(self) -> dict:
        n_ctrl = sum(2 ** len(n.parents) for n in self.nodes if n.parents)
        return {
            "qubits": self.n_qubits,
            "nodes": len(self.nodes),
            "controlled_rotations": n_ctrl,
            "joint_states": 2 ** self.n_qubits,
        }


class ClassicalBayesianNetwork:
    """Exact classical enumeration — the correctness oracle and speed baseline."""

    def __init__(self, nodes: Optional[Sequence[Node]] = None) -> None:
        self.nodes: List[Node] = list(nodes) if nodes else default_gaze_network()
        self.index = {n.name: i for i, n in enumerate(self.nodes)}

    def joint(self) -> np.ndarray:
        n = len(self.nodes)
        probs = np.zeros(2 ** n)
        for m in range(2 ** n):
            bits = [(m >> (n - 1 - k)) & 1 for k in range(n)]
            p = 1.0
            for node in self.nodes:
                val = bits[self.index[node.name]]
                parent_state = tuple(bits[self.index[pa]] for pa in node.parents)
                p1 = node.prob(parent_state)
                p *= p1 if val == 1 else (1.0 - p1)
            probs[m] = p
        return probs

    def conditional(self, target: str, evidence: Dict[str, int]) -> float:
        n = len(self.nodes)
        probs = self.joint()
        states = np.array([[(m >> (n - 1 - k)) & 1 for k in range(n)] for m in range(2 ** n)])
        mask = np.ones(len(states), dtype=bool)
        for k, v in evidence.items():
            mask &= states[:, self.index[k]] == v
        denom = probs[mask].sum()
        if denom < 1e-15:
            return float("nan")
        return float(probs[mask & (states[:, self.index[target]] == 1)].sum() / denom)
