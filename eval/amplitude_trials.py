"""Amplitude estimators and Grover, mapped onto this project's actual bottlenecks.

Two places in this pipeline have the shape these algorithms need.

**C1 — amplitude estimation for the reliability network.** ``quantum/qbn.py``
encodes a five-node Bayesian network as multi-controlled rotations, so that
measuring the register samples the joint distribution. Conditional inference by
rejection sampling costs $O(1/P(e))$ samples; amplitude estimation costs
$O(1/\\sqrt{P(e)})$. The four estimators in ``qiskit_algorithms`` are exactly the
tools for this, and the network is small enough that exact enumeration supplies
ground truth. The question is not whether they are correct but whether the
speedup is reachable at this problem size.

**C2 — Grover on the consistency oracle.** ``clusters.py::is_consistent`` is a
predicate over candidate letters, and a predicate is an oracle. We put
``qiskit_algorithms.Grover`` on it and compare against exhaustive argmax and
against the hand-rolled amplification in ``quantum/conditional_selection.py``.

Neither touches the gaze coordinate. Gaze is a regression and these are search
and estimation routines; we report what they do to inference cost and to the
downstream commit, and say plainly that the degrees figure is untouched.

Run: ``python3 eval/amplitude_trials.py [c1|c2|all]``
"""

from __future__ import annotations

import itertools
import json
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qiskit import QuantumCircuit                                   # noqa: E402
from qiskit.circuit.library import MCXGate, RYGate                  # noqa: E402
from qiskit.primitives import StatevectorSampler                    # noqa: E402
from qiskit_algorithms import (                                     # noqa: E402
    AmplificationProblem, AmplitudeEstimation, EstimationProblem,
    FasterAmplitudeEstimation, Grover, IterativeAmplitudeEstimation,
    MaximumLikelihoodAmplitudeEstimation,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "amplitude_trials_results.json"


def _save(path, new: dict) -> None:
    """Merge into the results file rather than overwriting it.

    Each part of this script can be run on its own, and a plain write means the
    last part run is the only one on disk. Merging keeps every part's results
    together however they were produced.
    """
    old = {}
    if path.exists():
        try:
            old = json.loads(path.read_text())
        except Exception:
            old = {}
    old.update(new)
    path.write_text(json.dumps(old, indent=2, default=float))


# The deployed five-node reliability network (quantum/qbn.py).
NETWORK = [
    ("blink",       [],                       {(): 0.06}),
    ("saccade",     [],                       {(): 0.18}),
    ("eye_closed",  ["blink"],                {(0,): 0.02, (1,): 0.93}),
    ("unreliable",  ["blink", "saccade"],     {(0, 0): 0.05, (0, 1): 0.55,
                                               (1, 0): 0.80, (1, 1): 0.92}),
    ("mis_key",     ["unreliable"],           {(0,): 0.03, (1,): 0.46}),
]
NAMES = [n for n, _, _ in NETWORK]
IDX = {n: i for i, n in enumerate(NAMES)}


# ==========================================================================
# Exact ground truth
# ==========================================================================

def exact_joint() -> Dict[Tuple[int, ...], float]:
    """Enumerate all 32 assignments — the correctness oracle and speed baseline."""
    joint = {}
    for assign in itertools.product((0, 1), repeat=len(NETWORK)):
        p = 1.0
        for i, (name, parents, cpt) in enumerate(NETWORK):
            key = tuple(assign[IDX[q]] for q in parents)
            p1 = cpt[key]
            p *= p1 if assign[i] == 1 else (1.0 - p1)
        joint[assign] = p
    return joint


def exact_marginal(evidence: Dict[str, int]) -> float:
    """P(evidence) by enumeration."""
    return sum(p for a, p in exact_joint().items()
               if all(a[IDX[k]] == v for k, v in evidence.items()))


# ==========================================================================
# The network as a circuit
# ==========================================================================

def network_circuit() -> QuantumCircuit:
    """Prepare sum_x sqrt(P(x)) |x> with multi-controlled RY, in topological order."""
    n = len(NETWORK)
    qc = QuantumCircuit(n)
    for i, (name, parents, cpt) in enumerate(NETWORK):
        if not parents:
            qc.ry(2 * math.asin(math.sqrt(cpt[()])), i)
            continue
        pq = [IDX[q] for q in parents]
        for assignment, p1 in sorted(cpt.items()):
            # open-control on the zeros
            zeros = [pq[k] for k, b in enumerate(assignment) if b == 0]
            for z in zeros:
                qc.x(z)
            theta = 2 * math.asin(math.sqrt(p1))
            if len(pq) == 1:
                qc.cry(theta, pq[0], i)
            else:
                qc.append(_mcry(theta, len(pq)), pq + [i])
            for z in zeros:
                qc.x(z)
    return qc


def _mcry(theta: float, n_ctrl: int):
    """Multi-controlled RY. RYGate carries its angle, so .control() works on it
    directly; wrapping it in a sub-circuit first loses the parameter."""
    return RYGate(theta).control(n_ctrl)


def evidence_oracle(evidence: Dict[str, int]) -> QuantumCircuit:
    """Phase-flip the basis states consistent with the evidence."""
    n = len(NETWORK)
    qc = QuantumCircuit(n)
    qs = [IDX[k] for k in evidence]
    zeros = [IDX[k] for k, v in evidence.items() if v == 0]
    for z in zeros:
        qc.x(z)
    if len(qs) == 1:
        qc.z(qs[0])
    else:
        qc.h(qs[-1])
        qc.append(MCXGate(len(qs) - 1), qs)
        qc.h(qs[-1])
    for z in zeros:
        qc.x(z)
    return qc


def good_states(evidence: Dict[str, int]) -> List[str]:
    """Bitstrings consistent with the evidence, in qiskit's little-endian order."""
    out = []
    n = len(NETWORK)
    for assign in itertools.product((0, 1), repeat=n):
        if all(assign[IDX[k]] == v for k, v in evidence.items()):
            out.append("".join(str(assign[n - 1 - i]) for i in range(n)))
    return out


# ==========================================================================
# C1 — amplitude estimation
# ==========================================================================

def part_c1() -> List[dict]:
    """Estimate P(evidence) with each estimator, against exact enumeration.

    A note on the setup, because getting it wrong is silent. An
    ``EstimationProblem`` without an explicit ``grover_operator`` builds its
    oracle from ``objective_qubits`` alone: it marks the state where *all* of
    them are 1. ``is_good_state`` only post-processes the sampled counts, so
    naming the whole register and supplying a custom predicate leaves the Grover
    operator marking |11111> and every query silently estimates the same
    quantity. Every case below therefore states its evidence as
    ``qubit = 1`` and passes exactly those qubits as the objective.
    """
    cases = [
        ("P(mis_key=1)", {"mis_key": 1}),
        ("P(blink=1)", {"blink": 1}),
        ("P(eye_closed=1, mis_key=1)", {"eye_closed": 1, "mis_key": 1}),
        ("P(blink=1, saccade=1, mis_key=1)  [rare evidence]",
         {"blink": 1, "saccade": 1, "mis_key": 1}),
    ]
    sampler = StatevectorSampler(default_shots=8192, seed=7)
    rows = []

    for label, ev in cases:
        assert all(v == 1 for v in ev.values()), "default oracle marks all-ones"
        truth = exact_marginal(ev)
        problem = EstimationProblem(
            state_preparation=network_circuit(),
            objective_qubits=[IDX[k] for k in ev])

        t0 = time.time()
        for _ in range(1000):
            exact_marginal(ev)
        exact_sec = time.time() - t0        # 1000 full enumerations

        estimators = [
            ("AmplitudeEstimation (m=6)",
             lambda: AmplitudeEstimation(num_eval_qubits=6, sampler=sampler)),
            ("AmplitudeEstimation (m=8)",
             lambda: AmplitudeEstimation(num_eval_qubits=8, sampler=sampler)),
            ("IterativeAmplitudeEstimation (eps=0.01)",
             lambda: IterativeAmplitudeEstimation(0.01, alpha=0.05, sampler=sampler)),
            ("IterativeAmplitudeEstimation (eps=0.001)",
             lambda: IterativeAmplitudeEstimation(0.001, alpha=0.05, sampler=sampler)),
            ("MaximumLikelihoodAmplitudeEstimation (k=5)",
             lambda: MaximumLikelihoodAmplitudeEstimation(5, sampler=sampler)),
            ("FasterAmplitudeEstimation (delta=0.01)",
             lambda: FasterAmplitudeEstimation(0.01, maxiter=6, sampler=sampler)),
        ]
        for name, make in estimators:
            t = time.time()
            try:
                res = make().estimate(problem)
                est = float(getattr(res, "estimation_processed", res.estimation))
                rows.append({"case": label, "name": name, "truth": truth,
                             "estimate": est, "abs_err": abs(est - truth),
                             "rel_err": abs(est - truth) / max(truth, 1e-12),
                             "sec": time.time() - t, "exact_1000x_sec": exact_sec})
            except Exception as exc:
                rows.append({"case": label, "name": name, "truth": truth,
                             "failed": "%s: %s" % (type(exc).__name__, exc)})
    return rows


# ==========================================================================
# C2 — Grover on the consistency oracle
# ==========================================================================

def part_c2() -> List[dict]:
    """Search the cluster-consistent subspace with Grover.

    The task: among the 32 letters of the candidate register, find one the
    consistency oracle accepts. Classical exhaustive check is 32 predicate
    evaluations and always succeeds; Grover is O(sqrt(32/|S|)) oracle calls and
    succeeds probabilistically. We report both, and the classical cost in the
    only unit that matters here, which is wall-clock microseconds.
    """
    from clusters import build_char_map
    sampler = StatevectorSampler()
    rows = []

    cmap = build_char_map(9)
    alphabet = [chr(o) for o in range(ord("a"), ord("z") + 1)]

    for cluster in ("1", "5", "9"):
        members = [c for c in alphabet if cmap.get(c) == cluster]
        if not members:
            continue
        marked = [alphabet.index(c) for c in members]
        n = 5                       # 32-state register, as deployed
        good = ["".join("1" if (m >> (n - 1 - i)) & 1 else "0" for i in range(n))
                for m in marked]

        oracle = QuantumCircuit(n)
        for m in marked:
            bits = [(m >> i) & 1 for i in range(n)]
            zeros = [i for i, b in enumerate(bits) if b == 0]
            for z in zeros:
                oracle.x(z)
            oracle.h(n - 1)
            oracle.append(MCXGate(n - 1), list(range(n)))
            oracle.h(n - 1)
            for z in zeros:
                oracle.x(z)

        problem = AmplificationProblem(oracle, is_good_state=good)

        t = time.time()
        try:
            res = Grover(sampler=sampler,
                         iterations=list(range(1, 5))).amplify(problem)
            top = res.top_measurement
            hit = top in good
            rows.append({"cluster": cluster, "n_marked": len(marked),
                         "members": "".join(members),
                         "name": "Grover (iterations 1..4)",
                         "found": top, "correct": bool(hit),
                         "sec": time.time() - t})
        except Exception as exc:
            rows.append({"cluster": cluster, "name": "Grover",
                         "failed": "%s: %s" % (type(exc).__name__, exc)})

        # classical control on the identical predicate
        t = time.time()
        reps = 10000
        for _ in range(reps):
            _ = [c for c in alphabet if cmap.get(c) == cluster]
        classical_us = 1e6 * (time.time() - t) / reps
        rows.append({"cluster": cluster, "n_marked": len(marked),
                     "name": "classical exhaustive scan",
                     "found": members[0], "correct": True,
                     "sec": classical_us / 1e6,
                     "microseconds": classical_us})
    return rows


# ==========================================================================
# Driver
# ==========================================================================

def main() -> None:
    part = (sys.argv[1] if len(sys.argv) > 1 else "all").lower()
    out = {}
    print("=" * 86)
    print("AMPLITUDE ESTIMATION AND GROVER — qiskit_algorithms on this pipeline")
    print("=" * 86)

    if part in ("c1", "all"):
        print("\nC1. AMPLITUDE ESTIMATION FOR THE RELIABILITY NETWORK")
        print("-" * 86)
        rows = part_c1()
        cur = None
        for r in rows:
            if r["case"] != cur:
                cur = r["case"]
                print("\n  %s   exact = %.6f" % (cur, r["truth"]))
                print("  %-46s %11s %11s %9s"
                      % ("estimator", "estimate", "abs err", "seconds"))
                print("  " + "-" * 80)
            if r.get("failed"):
                print("  %-46s %11s   %s" % (r["name"], "FAILED", r["failed"][:30]))
            else:
                print("  %-46s %11.6f %11.2e %9.2f"
                      % (r["name"], r["estimate"], r["abs_err"], r["sec"]))
        ex = [r for r in rows if not r.get("failed")]
        if ex:
            print("\n  exact enumeration of all 32 rows: %.1f microseconds"
                  % (1e6 * ex[0]["exact_1000x_sec"] / 1000))
        out["c1"] = rows

    if part in ("c2", "all"):
        print("\n\nC2. GROVER ON THE CONSISTENCY ORACLE")
        print("-" * 86)
        rows = part_c2()
        print("  %-28s %-10s %-9s %-9s %10s"
              % ("method", "cluster", "marked", "correct", "seconds"))
        print("  " + "-" * 80)
        for r in rows:
            if r.get("failed"):
                print("  %-28s %-10s %s" % (r["name"], r["cluster"], r["failed"][:36]))
            else:
                print("  %-28s %-10s %-9d %-9s %10.5f"
                      % (r["name"], r["cluster"], r["n_marked"],
                         "yes" if r["correct"] else "NO", r["sec"]))
        out["c2"] = rows

    _save(RESULTS, out)
    print("\nwrote %s" % RESULTS)


if __name__ == "__main__":
    main()
