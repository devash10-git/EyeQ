"""Quantum Residual Gaze Correction (QRGC): a second stage after the canthus ridge.

The canthus-normalised ridge answers "where is the user looking?" This stage
answers a narrower question: "given the ridge's estimate and the current state
of the eye, what error is the ridge making right now, and should I trust my
correction enough to apply it?"

    g_c   = ridge(12 canthus features)                   coarse estimate
    E_t   = error-state vector (below)
    p_x, p_y = Born distributions over a correction-hypothesis register
    dx    = sum_i p_x(i) c_i,   C_x = 1 - H(p_x) / log K      (same for y)
    g*    = g_c + [C_x dx, C_y dy]                        confidence-gated

Error-state vector E_t (14 dims, all computable live, frame by frame):
    canthus offsets (4) | ridge estimate x, y (2) | its velocity (2) and
    acceleration (2) | eye aspect ratio L, R (2) | binocular disagreement (1) |
    time into the current fixation (1)

Register: two 3-qubit registers (dx, dy), 8 hypotheses each, 1.5 deg apart over
+-5.25 deg. The 6 qubits are entangled, so the two axes are not independent.
Features are re-uploaded in every layer. Trained by cross-entropy against soft
labels that interpolate the true residual between its two nearest hypotheses,
so the register's expected value is unbiased inside the range.

Everything is exact state-vector simulation on a CPU (64 amplitudes); nothing
here runs on quantum hardware or claims a speed-up.
"""

from __future__ import annotations

import math
from typing import Callable, Tuple

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

N_REG = 3                                   # qubits per axis register
K = 2 ** N_REG                              # hypotheses per axis
SPACING_DEG = 1.5
CENTRES_DEG = (np.arange(K) - (K - 1) / 2) * SPACING_DEG        # -5.25 ... +5.25

ERROR_STATE = ["can0", "can1", "can2", "can3", "x_hat", "y_hat", "vx", "vy", "ax", "ay",
               "ear_l", "ear_r", "binoc", "dwell_s"]


# ====================================================================== simulator

class Sim:
    """Batched state-vector simulator, big-endian (qubit 0 = most significant)."""

    def __init__(self, n: int):
        self.n, self.N = n, 2 ** n
        idx = torch.arange(self.N)
        self.bit = [((idx >> (n - 1 - k)) & 1) for k in range(n)]
        self.flip = [idx ^ (1 << (n - 1 - k)) for k in range(n)]
        self.sgn = [(1 - 2 * b).to(torch.float64) for b in self.bit]       # +1 for |0>, -1 for |1>

    def zero(self, B: int) -> torch.Tensor:
        psi = torch.zeros(B, self.N, dtype=torch.complex128)
        psi[:, 0] = 1
        return psi

    # Single-qubit gates act on a (B, 2^k, 2, 2^(n-k-1)) view, so a gate is two
    # slices and a stack rather than a gather over all amplitudes.
    def _split(self, psi, k):
        v = psi.reshape(psi.shape[0], 2 ** k, 2, 2 ** (self.n - k - 1))
        return v[:, :, 0, :], v[:, :, 1, :]

    def _join(self, a0, a1):
        return torch.stack([a0, a1], 2).reshape(a0.shape[0], self.N)

    @staticmethod
    def _angle(th):
        th = torch.as_tensor(th, dtype=torch.float64)
        return th.reshape(-1, 1, 1) if th.dim() else th

    def ry(self, psi, k, th):
        th = self._angle(th)
        c, s = torch.cos(th / 2), torch.sin(th / 2)
        a0, a1 = self._split(psi, k)
        return self._join(c * a0 - s * a1, s * a0 + c * a1)

    def rz(self, psi, k, th):
        th = self._angle(th)
        a0, a1 = self._split(psi, k)
        return self._join(a0 * torch.exp(-0.5j * th), a1 * torch.exp(0.5j * th))

    def h(self, psi, k):
        a0, a1 = self._split(psi, k)
        return self._join((a0 + a1) / math.sqrt(2), (a0 - a1) / math.sqrt(2))

    def cnot_perm(self, c, t):
        return torch.where(self.bit[c] == 1, self.flip[t], torch.arange(self.N))

    def cnot(self, psi, c, t):
        return psi[:, self.cnot_perm(c, t)]

    def pauli_expectations(self, psi) -> torch.Tensor:
        """<X_k>, <Y_k>, <Z_k> for every qubit -> (B, 3n). Used by the projected kernel."""
        p = psi.abs() ** 2
        out = []
        for k in range(self.n):
            pk = psi[:, self.flip[k]]
            ex = (psi.conj() * pk).sum(1).real
            ey = (psi.conj() * pk * (-1j) * self.sgn[k]).sum(1).real
            ez = (p * self.sgn[k]).sum(1)
            out += [ex, ey, ez]
        return torch.stack(out, 1)


# ====================================================================== the register

class CorrectionRegister(torch.nn.Module):
    """Data re-uploading circuit whose measurement is a distribution over corrections."""

    def __init__(self, d: int, layers: int = 4, seed: int = 0):
        super().__init__()
        n = 2 * N_REG
        self.sim, self.n, self.layers = Sim(n), n, layers
        g = torch.Generator().manual_seed(seed)
        self.w = torch.nn.Parameter(torch.ones(layers, n))
        self.b = torch.nn.Parameter(torch.zeros(layers, n))
        self.rot = torch.nn.Parameter(0.1 * torch.randn(layers, n, 2, generator=g))
        self.fin = torch.nn.Parameter(0.1 * torch.randn(n, generator=g))
        # which feature each (layer, qubit) slot re-uploads: all d features are
        # visited, cycling, so every feature reaches both registers
        self.register_buffer("slot", torch.tensor([[(l * n + q) % d for q in range(n)]
                                                   for l in range(layers)]))

    def _fixed_layer(self, l: int) -> torch.Tensor:
        """The data-independent part of layer l as one matrix M (psi_out = psi_in @ M).

        Built by running the identity through the exact gate sequence, so it is
        the same circuit as ``forward_gates`` by construction (and tested).
        """
        s, psi = self.sim, torch.eye(self.sim.N, dtype=torch.complex128)
        for q in range(self.n):
            psi = s.ry(psi, q, self.rot[l, q, 0])
            psi = s.rz(psi, q, self.rot[l, q, 1])
        for q in range(self.n):
            psi = s.cnot(psi, q, (q + 1) % self.n)
        return psi

    def _final(self) -> torch.Tensor:
        s, psi = self.sim, torch.eye(self.sim.N, dtype=torch.complex128)
        for q in range(self.n):
            psi = s.ry(psi, q, self.fin[q])
        return psi

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s, psi = self.sim, self.sim.zero(x.shape[0])
        for l in range(self.layers):
            for q in range(self.n):
                psi = s.ry(psi, q, self.w[l, q] * x[:, self.slot[l, q]] + self.b[l, q])
            psi = psi @ self._fixed_layer(l)
        psi = psi @ self._final()
        p = (psi.abs() ** 2).reshape(-1, K, K)            # Born rule; qubits 0-2 = dx, 3-5 = dy
        return p.sum(2), p.sum(1)

    def forward_gates(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reference: the same circuit applied one gate at a time."""
        s, psi = self.sim, self.sim.zero(x.shape[0])
        for l in range(self.layers):
            for q in range(self.n):
                psi = s.ry(psi, q, self.w[l, q] * x[:, self.slot[l, q]] + self.b[l, q])
                psi = s.ry(psi, q, self.rot[l, q, 0])
                psi = s.rz(psi, q, self.rot[l, q, 1])
            for q in range(self.n):
                psi = s.cnot(psi, q, (q + 1) % self.n)
        for q in range(self.n):
            psi = s.ry(psi, q, self.fin[q])
        p = (psi.abs() ** 2).reshape(-1, K, K)
        return p.sum(2), p.sum(1)


class ClassicalTwin(torch.nn.Module):
    """Same readout, same loss, same gate; a tiny MLP in place of the circuit.

    Hidden width 3 gives 109 parameters against the register's 102, so any gap
    between the two is not a parameter-count effect.
    """

    def __init__(self, d: int, hidden: int = 3, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.Tanh(),
                                       torch.nn.Linear(hidden, 2 * K))

    def forward(self, x):
        z = self.net(x)
        return torch.softmax(z[:, :K], 1), torch.softmax(z[:, K:], 1)


def n_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def soft_labels(r_deg: np.ndarray) -> np.ndarray:
    """Split each residual between its two nearest hypotheses so E[c] = r (inside range)."""
    r = np.clip(r_deg, CENTRES_DEG[0], CENTRES_DEG[-1])
    pos = (r - CENTRES_DEG[0]) / SPACING_DEG
    lo = np.clip(np.floor(pos).astype(int), 0, K - 2)
    frac = pos - lo
    out = np.zeros((len(r), K))
    out[np.arange(len(r)), lo] = 1 - frac
    out[np.arange(len(r)), lo + 1] = frac
    return out


def readout(px: np.ndarray, py: np.ndarray, gate: bool = True) -> np.ndarray:
    """Expected correction in degrees, optionally scaled by the register's own confidence."""
    c = CENTRES_DEG
    d = np.column_stack([px @ c, py @ c])
    if not gate:
        return d
    ent = lambda p: -(p * np.log(np.clip(p, 1e-12, 1))).sum(1) / math.log(K)
    conf = np.column_stack([1 - ent(px), 1 - ent(py)])
    return conf * d


def fit_register(model: torch.nn.Module, Z: np.ndarray, R_deg: np.ndarray,
                 steps: int = 250, lr: float = 0.03) -> Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    X = torch.tensor(Z)
    Lx, Ly = torch.tensor(soft_labels(R_deg[:, 0])), torch.tensor(soft_labels(R_deg[:, 1]))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        px, py = model(X)
        loss = -(Lx * torch.log(px + 1e-12)).sum(1).mean() - (Ly * torch.log(py + 1e-12)).sum(1).mean()
        loss.backward()
        opt.step()

    def predict(Q):
        with torch.no_grad():
            px, py = model(torch.tensor(Q))
        return px.numpy(), py.numpy()
    return predict


# ====================================================================== quantum kernels

KERNEL_QUBITS = 7                       # 14 features, two per qubit


def kernel_states(Z: np.ndarray, scale: float, reps: int = 2) -> torch.Tensor:
    """Re-uploading feature map: H, RZ(s x_2q), RY(s x_2q+1) per qubit, CNOT ring."""
    n = KERNEL_QUBITS
    s, X = Sim(n), torch.tensor(scale * Z)
    psi = s.zero(len(Z))
    for _ in range(reps):
        for q in range(n):
            psi = s.h(psi, q)
            psi = s.rz(psi, q, X[:, 2 * q])
            psi = s.ry(psi, q, X[:, 2 * q + 1])
        for q in range(n):
            psi = s.cnot(psi, q, (q + 1) % n)
    return psi


def fidelity_kernel(A: torch.Tensor, B: torch.Tensor) -> np.ndarray:
    return (torch.abs(A.conj() @ B.T) ** 2).numpy()


def projected_features(psi: torch.Tensor) -> np.ndarray:
    return Sim(KERNEL_QUBITS).pauli_expectations(psi).numpy()


# ====================================================================== error state

def error_state(C: np.ndarray, P_hat: np.ndarray, t_ms: np.ndarray, stream: np.ndarray,
                ear: np.ndarray, binoc: np.ndarray, screen: Tuple[float, float]) -> np.ndarray:
    """Build E_t frame by frame.

    ``stream`` labels contiguous runs of frames (one fixation); velocity,
    acceleration and dwell restart at the start of each run, exactly as they
    would when the live dwell timer restarts on a new key.
    """
    W, H = screen
    g = P_hat / np.array([W, H])
    v = np.zeros_like(g)
    a = np.zeros_like(g)
    dwell = np.zeros(len(g))
    order = np.lexsort((t_ms, stream))
    prev = None
    for i in order:
        if prev is not None and stream[i] == stream[prev]:
            dt = max(1e-3, (t_ms[i] - t_ms[prev]) / 1000.0)
            v[i] = (g[i] - g[prev]) / dt
            a[i] = (v[i] - v[prev]) / dt
            dwell[i] = dwell[prev] + dt
        prev = i
    return np.column_stack([C, g, v, a, ear, binoc, dwell])
