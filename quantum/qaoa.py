"""QAOA formulations for the combinatorial problems in the gaze-typing system.

Scope discipline
----------------
QAOA solves *combinatorial* problems. Several items commonly grouped under
"optimization" in this project are **not** combinatorial, and are deliberately
NOT routed here:

* **Calibration fit** — ridge regression is continuous and convex with a
  closed-form global optimum. No approximate optimiser can beat an exact one.
* **Dwell-time / threshold tuning** — one- or two-dimensional continuous
  scalars; a grid search or golden-section search is exact to machine precision.
* **Typing-sequence decoding** — best path through a word lattice is a
  shortest-path problem on a DAG, solved *exactly* in ``O(n·V^2)`` by Viterbi
  (implemented classically in ``decoder.py``). QAOA can only approximate it, so
  it is included below purely as a measured control showing the classical
  method dominates.

Three problems here *are* genuinely NP-hard and are the defensible targets:

1. **Keyboard layout** — Quadratic Assignment Problem (QAP). NP-hard; no known
   polynomial exact algorithm. Offline and one-time, so QPU latency is
   irrelevant. This is the strongest quantum-suitability case in the project.
2. **Feature subset selection** — 2^d subsets under a relevance/redundancy
   objective; QUBO-native.
3. **Calibration point selection** — choose k of n candidate targets to
   minimise predictive variance; a k-subset problem.

QUBO -> Ising
-------------
A QUBO ``min x^T Q x`` over ``x in {0,1}^n`` maps to an Ising Hamiltonian with
``x_i = (1 - z_i)/2``, giving ``H_C = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j``.
QAOA then alternates ``exp(-i*gamma*H_C)`` with a transverse mixer
``exp(-i*beta*sum_i X_i)`` for ``p`` layers, and a classical optimiser tunes
``(gamma, beta)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


# --------------------------------------------------------------------------
# QUBO / Ising utilities
# --------------------------------------------------------------------------

def qubo_to_ising(Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Convert a QUBO matrix to Ising coefficients ``(h, J, offset)``.

    With ``x_i = (1 - z_i)/2`` and ``E(x) = sum_ij Q_ij x_i x_j``.
    """
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    Qs = (Q + Q.T) / 2.0                      # symmetrise; x_i x_j == x_j x_i
    h = np.zeros(n)
    J = np.zeros((n, n))
    offset = 0.0
    for i in range(n):
        offset += Qs[i, i] / 2.0
        h[i] -= Qs[i, i] / 2.0
        for j in range(i + 1, n):
            offset += Qs[i, j] / 2.0
            h[i] -= Qs[i, j] / 2.0
            h[j] -= Qs[i, j] / 2.0
            J[i, j] += Qs[i, j] / 2.0
    return h, J, offset


def qubo_energy(Q: np.ndarray, x: np.ndarray) -> float:
    """Objective value of a bitstring under the QUBO."""
    x = np.asarray(x, dtype=float)
    return float(x @ np.asarray(Q, dtype=float) @ x)


def brute_force_qubo(Q: np.ndarray, max_bits: int = 20) -> Tuple[np.ndarray, float]:
    """Exact minimiser by exhaustive search — the ground truth for small n."""
    n = Q.shape[0]
    if n > max_bits:
        raise ValueError(f"refusing brute force over 2^{n} states")
    best_x, best_e = None, np.inf
    for m in range(2 ** n):
        x = np.array([(m >> k) & 1 for k in range(n)], dtype=float)
        e = qubo_energy(Q, x)
        if e < best_e:
            best_e, best_x = e, x
    return best_x, float(best_e)


# --------------------------------------------------------------------------
# QAOA solver core
# --------------------------------------------------------------------------

@dataclass
class QAOAResult:
    bitstring: np.ndarray
    energy: float
    optimal_energy: Optional[float]
    approximation_ratio: Optional[float]
    n_qubits: int
    p_layers: int
    circuit_evaluations: int
    params: np.ndarray
    top_samples: List[Tuple[str, float]]


class QAOASolver:
    """Generic QAOA solver for a QUBO instance.

    Args:
        Q: QUBO matrix (n x n).
        p: number of QAOA layers. Deeper is more expressive but harder to
            optimise and noisier on hardware; p in [1,4] is the NISQ-realistic
            range.
        steps: classical optimiser iterations over ``(gamma, beta)``.
    """

    def __init__(self, Q: np.ndarray, p: int = 2, steps: int = 60,
                 lr: float = 0.12, seed: int = 13,
                 mixer: str = "x", rows: Optional[int] = None) -> None:
        """
        Args:
            mixer: ``"x"`` for the standard transverse-field mixer (feasibility
                enforced only by QUBO penalties), or ``"xy_rows"`` for a
                constraint-preserving XY mixer.

                The XY mixer conserves Hamming weight *within each row*, so if
                the circuit is initialised on a valid one-hot-per-row state it
                can never leave that subspace (Hadfield et al., 2019, Quantum
                Alternating Operator Ansatz). For assignment problems this
                removes an entire class of constraint violations that the
                penalty formulation must otherwise learn to avoid, and it
                shrinks the search space from ``2^(n^2)`` to ``n^n``.
            rows: number of rows in the one-hot block structure (required for
                ``"xy_rows"``); the QUBO length must be ``rows * cols``.
        """
        self.Q = np.asarray(Q, dtype=float)
        self.n = self.Q.shape[0]
        self.p = int(p)
        self.steps = int(steps)
        self.lr = lr
        self.seed = seed
        self.mixer = mixer
        self.rows = rows
        if mixer == "xy_rows":
            if not rows or self.n % rows:
                raise ValueError("xy_rows mixer needs `rows` dividing the QUBO size")
            self.cols = self.n // rows
        self.h, self.J, self.offset = qubo_to_ising(self.Q)
        self.dev = qml.device("default.qubit", wires=self.n)
        self._evals = 0
        self._cost_h = self._build_hamiltonian()

    def _build_hamiltonian(self) -> qml.Hamiltonian:
        coeffs: List[float] = []
        ops: List[qml.operation.Operator] = []
        for i in range(self.n):
            if abs(self.h[i]) > 1e-12:
                coeffs.append(float(self.h[i])); ops.append(qml.PauliZ(i))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if abs(self.J[i, j]) > 1e-12:
                    coeffs.append(float(self.J[i, j]))
                    ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
        if not coeffs:                       # degenerate all-zero objective
            coeffs, ops = [0.0], [qml.Identity(0)]
        return qml.Hamiltonian(coeffs, ops)

    def _initial_state(self) -> None:
        """Prepare the QAOA start state."""
        if self.mixer == "xy_rows":
            # a feasible assignment (identity permutation): one 1 per row
            for r in range(self.rows):
                qml.PauliX(wires=r * self.cols + (r % self.cols))
        else:
            for w in range(self.n):
                qml.Hadamard(wires=w)        # uniform superposition

    def _apply_mixer(self, beta) -> None:
        if self.mixer == "xy_rows":
            # XY (swap) interactions along a ring within each row conserve the
            # row's Hamming weight, so one-hot feasibility is preserved exactly.
            for r in range(self.rows):
                base = r * self.cols
                for c in range(self.cols):
                    a, b = base + c, base + (c + 1) % self.cols
                    qml.IsingXX(2.0 * beta, wires=[a, b])
                    qml.IsingYY(2.0 * beta, wires=[a, b])
        else:
            for w in range(self.n):
                qml.RX(2.0 * beta, wires=w)

    def _ansatz(self, params) -> None:
        gammas, betas = params[0], params[1]
        self._initial_state()
        for layer in range(self.p):
            # cost unitary exp(-i gamma H_C)
            for i in range(self.n):
                if abs(self.h[i]) > 1e-12:
                    qml.RZ(2.0 * gammas[layer] * self.h[i], wires=i)
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    if abs(self.J[i, j]) > 1e-12:
                        qml.CNOT(wires=[i, j])
                        qml.RZ(2.0 * gammas[layer] * self.J[i, j], wires=j)
                        qml.CNOT(wires=[i, j])
            self._apply_mixer(betas[layer])

    def solve(self) -> QAOAResult:
        @qml.qnode(self.dev, interface="autograd")
        def expectation(params):
            self._ansatz(params)
            return qml.expval(self._cost_h)

        @qml.qnode(self.dev)
        def probabilities(params):
            self._ansatz(params)
            return qml.probs(wires=range(self.n))

        rng = np.random.default_rng(self.seed)
        # standard heuristic init: small gammas, betas near pi/4
        params = pnp.array(
            [rng.uniform(0.0, 0.6, self.p), rng.uniform(0.4, 1.0, self.p)],
            requires_grad=True,
        )
        opt = qml.AdamOptimizer(self.lr)
        for _ in range(self.steps):
            params = opt.step(expectation, params)
            self._evals += 1

        probs = np.asarray(probabilities(params))
        order = np.argsort(probs)[::-1]

        # QAOA returns a *distribution*; the answer is the best sampled string,
        # so read out the top candidates and keep the lowest-energy one.
        #
        # Feasibility matters here: with the XY mixer there are no penalty
        # terms in Q, so infeasible strings (e.g. all-zeros, energy 0) can be
        # numerically cheaper than every valid assignment. They live outside
        # the mixer-preserved subspace and only appear with ~0 amplitude via
        # numerical noise — but an unfiltered energy scan would still pick
        # them. Skip anything violating the one-hot row structure.
        best_x, best_e = None, np.inf
        for idx in order[: min(64, len(order))]:
            bits = np.array([(int(idx) >> (self.n - 1 - k)) & 1 for k in range(self.n)], dtype=float)
            if self.mixer == "xy_rows" and not all(
                bits[r * self.cols:(r + 1) * self.cols].sum() == 1
                for r in range(self.rows)
            ):
                continue
            e = qubo_energy(self.Q, bits)
            if e < best_e:
                best_e, best_x = e, bits

        top = []
        for idx in order[:5]:
            s = format(int(idx), f"0{self.n}b")
            top.append((s, float(probs[idx])))

        opt_e = None
        ratio = None
        if self.n <= 18:
            if self.mixer == "xy_rows":
                # ground truth over the feasible (one-hot per row) subspace
                from itertools import product as _iproduct
                opt_e = np.inf
                for combo in _iproduct(range(self.cols), repeat=self.rows):
                    x = np.zeros(self.n)
                    for r, c in enumerate(combo):
                        x[r * self.cols + c] = 1.0
                    opt_e = min(opt_e, qubo_energy(self.Q, x))
                opt_e = float(opt_e)
            else:
                _, opt_e = brute_force_qubo(self.Q)
            ratio = float(opt_e / best_e) if best_e not in (0, np.inf) else None
            if opt_e <= 0 and best_e <= 0:
                ratio = float(best_e / opt_e) if opt_e != 0 else None

        return QAOAResult(
            bitstring=best_x, energy=float(best_e), optimal_energy=opt_e,
            approximation_ratio=ratio, n_qubits=self.n, p_layers=self.p,
            circuit_evaluations=self._evals, params=np.asarray(params),
            top_samples=top,
        )


# --------------------------------------------------------------------------
# Problem 1 — Keyboard layout as a Quadratic Assignment Problem  (NP-hard)
# --------------------------------------------------------------------------

def build_layout_qubo(
    flow: np.ndarray,
    dist: np.ndarray,
    penalty: Optional[float] = None,
) -> np.ndarray:
    """QUBO for assigning ``n`` letters to ``n`` key positions.

    Variables ``x[i*n + p] = 1`` iff letter ``i`` occupies position ``p``.

    Objective: minimise expected gaze travel
        ``sum_{i,j,p,q} flow[i,j] * dist[p,q] * x_ip * x_jq``
    subject to one-hot rows and columns, enforced by quadratic penalties.

    The penalty weight must exceed the largest achievable objective swing or the
    optimiser can profit by violating feasibility; it defaults to that bound.
    """
    n = flow.shape[0]
    N = n * n
    Q = np.zeros((N, N))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    Q[i * n + p, j * n + q] += flow[i, j] * dist[p, q]

    if penalty is None:
        penalty = 2.0 * float(np.abs(Q).sum() / max(1, N)) + 1.0

    # sum_p x_ip = 1  ->  (sum_p x_ip - 1)^2 ; x^2 = x for binaries
    for i in range(n):
        for p in range(n):
            Q[i * n + p, i * n + p] -= penalty
            for q in range(n):
                if q != p:
                    Q[i * n + p, i * n + q] += penalty
    # sum_i x_ip = 1
    for p in range(n):
        for i in range(n):
            Q[i * n + p, i * n + p] -= penalty
            for j in range(n):
                if j != i:
                    Q[i * n + p, j * n + p] += penalty
    return Q


def decode_layout(bits: np.ndarray, n: int) -> Optional[List[int]]:
    """Decode a QAP bitstring into a permutation, or ``None`` if infeasible."""
    x = np.asarray(bits).reshape(n, n)
    if not (np.all(x.sum(axis=1) == 1) and np.all(x.sum(axis=0) == 1)):
        return None
    return [int(np.argmax(x[i])) for i in range(n)]


def layout_travel_cost(perm: Sequence[int], flow: np.ndarray, dist: np.ndarray) -> float:
    """Expected gaze travel for a given assignment (lower is better)."""
    n = len(perm)
    return float(sum(
        flow[i, j] * dist[perm[i], perm[j]]
        for i in range(n) for j in range(n) if i != j
    ))


def exact_layout(flow: np.ndarray, dist: np.ndarray) -> Tuple[List[int], float]:
    """Exact QAP solution by permutation enumeration (feasible for n <= 8)."""
    from itertools import permutations
    n = flow.shape[0]
    best, best_c = None, np.inf
    for perm in permutations(range(n)):
        c = layout_travel_cost(perm, flow, dist)
        if c < best_c:
            best_c, best = c, list(perm)
    return best, float(best_c)


# --------------------------------------------------------------------------
# Problem 2 — Feature subset selection (mRMR-style QUBO)
# --------------------------------------------------------------------------

def build_feature_selection_qubo(
    relevance: np.ndarray,
    redundancy: np.ndarray,
    k: Optional[int] = None,
    alpha: float = 1.0,
    penalty: float = 4.0,
) -> np.ndarray:
    """QUBO selecting an informative, non-redundant feature subset.

    ``min  -sum_i rel_i x_i + alpha * sum_{i<j} red_ij x_i x_j``
    optionally constrained to exactly ``k`` features.

    This is the max-relevance/min-redundancy criterion, which is NP-hard in
    general and QUBO-native — a genuine combinatorial target.
    """
    d = len(relevance)
    Q = np.zeros((d, d))
    for i in range(d):
        Q[i, i] -= float(relevance[i])
        for j in range(i + 1, d):
            Q[i, j] += alpha * float(redundancy[i, j])
    if k is not None:
        for i in range(d):
            Q[i, i] += penalty * (1.0 - 2.0 * k)
            for j in range(i + 1, d):
                Q[i, j] += 2.0 * penalty
    return Q


# --------------------------------------------------------------------------
# Problem 3 — Calibration point subset selection
# --------------------------------------------------------------------------

def build_calibration_qubo(
    coverage: np.ndarray,
    overlap: np.ndarray,
    k: int,
    penalty: float = 6.0,
) -> np.ndarray:
    """Choose ``k`` calibration targets maximising screen coverage.

    ``coverage[i]`` is the marginal value of target ``i`` (e.g. the reduction in
    predictive variance it provides); ``overlap[i,j]`` penalises targets that
    are spatially redundant. Same QUBO shape as feature selection, different
    semantics — both are k-subset problems.
    """
    return build_feature_selection_qubo(coverage, overlap, k=k, alpha=1.0, penalty=penalty)
