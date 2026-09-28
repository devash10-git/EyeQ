"""Checks for quantum/residual_correction.py.

1. The simulator agrees with PennyLane on random circuits (state and Pauli
   expectations), so the register and kernels are the circuits they claim to be.
2. The register's output is a normalised distribution on each axis.
3. Soft labels reproduce the residual exactly as an expectation inside the range.
4. The confidence gate never enlarges a correction.

    python3 quantum/test_residual_correction.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quantum.residual_correction import (CENTRES_DEG, K, CorrectionRegister, ClassicalTwin, Sim,
                                         n_params, readout, soft_labels)


def test_against_pennylane():
    import pennylane as qml
    rng = np.random.default_rng(0)
    n = 4
    ops = [("h", 0), ("ry", 1, 0.7), ("rz", 2, -1.1), ("cnot", 0, 3), ("ry", 3, 2.3),
           ("h", 2), ("cnot", 2, 1), ("rz", 0, 0.4), ("ry", 2, -0.9), ("cnot", 3, 0)]
    for trial in range(5):
        ops_t = [(o[0], o[1], o[2] + rng.normal() if o[0] in ("ry", "rz") else o[2]) if len(o) == 3 else o
                 for o in ops]
        dev = qml.device("default.qubit", wires=n)

        @qml.qnode(dev)
        def circ():
            for o in ops_t:
                if o[0] == "h": qml.Hadamard(o[1])
                elif o[0] == "ry": qml.RY(o[2], o[1])
                elif o[0] == "rz": qml.RZ(o[2], o[1])
                else: qml.CNOT([o[1], o[2]])
            return qml.state()

        @qml.qnode(dev)
        def paulis():
            for o in ops_t:
                if o[0] == "h": qml.Hadamard(o[1])
                elif o[0] == "ry": qml.RY(o[2], o[1])
                elif o[0] == "rz": qml.RZ(o[2], o[1])
                else: qml.CNOT([o[1], o[2]])
            return [qml.expval(P(k)) for k in range(n) for P in (qml.PauliX, qml.PauliY, qml.PauliZ)]

        s = Sim(n)
        psi = s.zero(1)
        for o in ops_t:
            if o[0] == "h": psi = s.h(psi, o[1])
            elif o[0] == "ry": psi = s.ry(psi, o[1], o[2])
            elif o[0] == "rz": psi = s.rz(psi, o[1], o[2])
            else: psi = s.cnot(psi, o[1], o[2])
        ref = np.asarray(circ())
        assert np.allclose(psi.numpy()[0], ref, atol=1e-10), "state mismatch"
        assert np.allclose(s.pauli_expectations(psi).numpy()[0], np.array(paulis(), float), atol=1e-10), \
            "pauli expectation mismatch"
    print("  simulator == PennyLane on 5 random circuits (state and X/Y/Z expectations)")


def test_register_normalised():
    m = CorrectionRegister(d=14)
    x = torch.tensor(np.random.default_rng(1).normal(size=(50, 14)))
    with torch.no_grad():
        m.rot.normal_(); m.fin.normal_(); m.w.normal_(); m.b.normal_()
    px, py = m(x)
    assert torch.allclose(px.sum(1), torch.ones(50)) and torch.allclose(py.sum(1), torch.ones(50))
    gx, gy = m.forward_gates(x)
    assert torch.allclose(px, gx, atol=1e-12) and torch.allclose(py, gy, atol=1e-12)
    print("  fused-layer register == gate-by-gate register (random parameters)")
    print(f"  register distributions sum to 1  ({n_params(m)} parameters; twin has "
          f"{n_params(ClassicalTwin(14))})")


def test_soft_labels_unbiased():
    r = np.linspace(CENTRES_DEG[0], CENTRES_DEG[-1], 101)
    assert np.allclose(soft_labels(r) @ CENTRES_DEG, r)
    print("  soft labels: expectation reproduces every in-range residual exactly")


def test_gate_shrinks():
    rng = np.random.default_rng(2)
    px, py = rng.dirichlet(np.ones(K), 100), rng.dirichlet(np.ones(K), 100)
    assert np.all(np.abs(readout(px, py, gate=True)) <= np.abs(readout(px, py, gate=False)) + 1e-12)
    print("  confidence gate never enlarges a correction")


if __name__ == "__main__":
    test_against_pennylane()
    test_register_normalised()
    test_soft_labels_unbiased()
    test_gate_shrinks()
    print("all checks passed")
