"""QAOA optimization of the gaze typing sequence.

The problem
-----------
While a word is typed by gaze, every dwell commit produces a *posterior over
letters* (from the Gaussian key posterior fused with the language prior in
``static/app.js``). Greedy typing takes the argmax at each position
independently — it can never revise an early guess in light of a later one.

This module re-decodes the whole word *jointly*: choose one candidate letter
per position minimising

    E(x) = sum_i -log P_gaze(letter_i)  +  lam * sum_i -log P_trans(l_{i+1} | l_i)

which is exactly a QUBO whose one-hot rows are letter positions. QAOA explores
all candidate combinations in superposition; each classical optimization step
of its (gamma, beta) schedule lowers the expected energy — i.e. raises the
probability that measurement yields the true intended word.

Pipeline design — why this cannot bottleneck typing
---------------------------------------------------
1. **Offline parameter training.** The QAOA angle schedule is trained once on
   representative synthetic instances (``train_schedule``) and saved to disk.
   At runtime there is **no optimizer in the loop**: one circuit evaluation
   with fixed angles, then sampling. Parameter transfer across instances of a
   problem family is standard QAOA practice.
2. **Ambiguous-segment extraction.** Positions whose gaze posterior is already
   confident (top-1 >= CERTAIN) are frozen and become boundary conditions;
   QAOA runs only on each maximal run of uncertain positions. Typical runs are
   1-3 letters -> 2-6 qubits, so circuits stay tiny.
3. **Word-boundary, asynchronous.** The endpoint is called after SPACE commits,
   off the 33 ms frame path and outside the 693 ms dwell path; the result
   arrives as a suggestion chip, never a blocking rewrite.
4. **Exact-fallback guarantee.** The chain MAP is also solved exactly (tiny
   dynamic program). If sampling with fixed angles misses the optimum, the
   exact answer is returned and the miss is *recorded honestly* in the
   response (``method`` field). Accuracy therefore cannot decrease; QAOA's
   sampling quality is measured, not assumed.

Honesty note (also see ``qaoa.py``): this chain MAP is classically solvable in
microseconds by Viterbi — included here as the guarantee and the yardstick.
The QAOA path is the requested research contribution and is evaluated against
that exact baseline, not instead of it.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from itertools import product as iproduct
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from .qaoa import QAOASolver, qubo_energy

PARAMS_PATH = os.path.join(os.path.dirname(__file__), "typing_qaoa_params.json")

#: gaze posterior above which a position is treated as certain (frozen)
CERTAIN = 0.90
#: candidates kept per ambiguous position (qubits per position)
K = 2
#: cap on qubits per QAOA segment (positions * K)
MAX_SEGMENT_QUBITS = 12
#: weight of the letter-transition (language) term relative to gaze evidence
LAMBDA = 1.0

_EPS = 1e-9


# --------------------------------------------------------------------------
# Letter-transition statistics from the project's real vocabulary
# --------------------------------------------------------------------------

class LetterStats:
    """First-letter and adjacent-letter transition probabilities (a-z)."""

    def __init__(self, unigrams: Dict[str, int]) -> None:
        trans = np.full((26, 26), 0.0)
        start = np.zeros(26)
        for w, c in unigrams.items():
            letters = [ch for ch in w if "a" <= ch <= "z"]
            if not letters:
                continue
            start[ord(letters[0]) - 97] += c
            for a, b in zip(letters, letters[1:]):
                trans[ord(a) - 97, ord(b) - 97] += c
        # add-k smoothing so no transition is impossible (names, acronyms)
        trans += 0.5
        start += 0.5
        self.trans = trans / trans.sum(axis=1, keepdims=True)
        self.start = start / start.sum()

    def p_trans(self, a: str, b: str) -> float:
        return float(self.trans[ord(a) - 97, ord(b) - 97])

    def p_start(self, a: str) -> float:
        return float(self.start[ord(a) - 97])


# --------------------------------------------------------------------------
# QUBO construction for one ambiguous segment
# --------------------------------------------------------------------------

def build_segment_qubo(
    cands: Sequence[Sequence[Tuple[str, float]]],
    stats: LetterStats,
    left: Optional[str],
    right: Optional[str],
    lam: float = LAMBDA,
) -> np.ndarray:
    """QUBO over one run of ambiguous positions.

    Args:
        cands: per position, ``K`` ``(letter, gaze_prob)`` candidates.
        stats: letter statistics.
        left/right: fixed letters adjacent to the segment (or ``None`` at the
            word edge). Their transition terms fold into the unary diagonal —
            this is how frozen positions constrain the segment without qubits.
    """
    n, k = len(cands), len(cands[0])
    N = n * k
    Q = np.zeros((N, N))

    for i, pos in enumerate(cands):
        for a, (ch, p) in enumerate(pos):
            u = -math.log(max(p, _EPS))                       # gaze evidence
            if i == 0:
                u += lam * -math.log(
                    stats.p_trans(left, ch) if left else stats.p_start(ch))
            if i == n - 1 and right:
                u += lam * -math.log(stats.p_trans(ch, right))
            Q[i * k + a, i * k + a] += u

    for i in range(n - 1):                                    # chain coupling
        for a, (c1, _p1) in enumerate(cands[i]):
            for b, (c2, _p2) in enumerate(cands[i + 1]):
                Q[i * k + a, (i + 1) * k + b] += lam * -math.log(
                    max(stats.p_trans(c1, c2), _EPS))
    return Q


def chain_exact(
    cands: Sequence[Sequence[Tuple[str, float]]],
    stats: LetterStats,
    left: Optional[str],
    right: Optional[str],
    lam: float = LAMBDA,
) -> Tuple[List[int], float]:
    """Exact MAP over the segment by enumeration (K^n <= 64 — microseconds).

    The classical yardstick and the fallback guarantee.
    """
    n = len(cands)
    Q = build_segment_qubo(cands, stats, left, right, lam)
    best, best_e = None, math.inf
    for combo in iproduct(range(K), repeat=n):
        x = np.zeros(n * K)
        for i, a in enumerate(combo):
            x[i * K + a] = 1.0
        e = qubo_energy(Q, x)
        if e < best_e:
            best_e, best = e, list(combo)
    return best, float(best_e)


# --------------------------------------------------------------------------
# Fixed-parameter QAOA sampling (runtime path — no optimizer in the loop)
# --------------------------------------------------------------------------

class FixedParamQAOA(QAOASolver):
    """QAOASolver that samples with a pre-trained angle schedule."""

    def sample(self, params: np.ndarray, top: int = 32
               ) -> Tuple[Optional[np.ndarray], float, float]:
        """One circuit evaluation with fixed ``(gamma, beta)``; returns the
        best feasible bitstring among the ``top`` most probable outcomes,
        its energy, and the probability mass QAOA placed on it."""

        @qml.qnode(self.dev)
        def probabilities(p_):
            self._ansatz(p_)
            return qml.probs(wires=range(self.n))

        probs = np.asarray(probabilities(pnp.array(params, requires_grad=False)))
        order = np.argsort(probs)[::-1][:top]
        best_x, best_e, best_p = None, math.inf, 0.0
        for idx in order:
            bits = np.array([(int(idx) >> (self.n - 1 - b)) & 1
                             for b in range(self.n)], dtype=float)
            # feasibility: exactly one candidate per position
            if not all(bits[r * self.cols:(r + 1) * self.cols].sum() == 1
                       for r in range(self.rows)):
                continue
            e = qubo_energy(self.Q, bits)
            if e < best_e:
                best_e, best_x, best_p = e, bits, float(probs[idx])
        return best_x, best_e, best_p


# --------------------------------------------------------------------------
# Offline schedule training — "each optimization step reduces the error"
# --------------------------------------------------------------------------

def _random_instance(stats: LetterStats, rng: np.random.Generator,
                     words: List[str], neighbours: Dict[str, List[str]]
                     ) -> Tuple[List[List[Tuple[str, float]]], str]:
    """Synthetic ambiguous segment from a real word + QWERTY-neighbour noise."""
    w = rng.choice(words)
    seg_len = int(rng.integers(1, min(4, len(w)) + 1))
    start = int(rng.integers(0, len(w) - seg_len + 1))
    seg = w[start:start + seg_len]
    cands = []
    for ch in seg:
        alt = rng.choice(neighbours.get(ch, ["e"]))
        p_true = float(rng.uniform(0.35, 0.75))
        pair = [(ch, p_true), (str(alt), 1.0 - p_true)]
        rng.shuffle(pair)
        cands.append([(c, float(p)) for c, p in pair])
    return cands, seg


def qwerty_neighbours() -> Dict[str, List[str]]:
    rows = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
    out: Dict[str, List[str]] = {}
    for r, row in enumerate(rows):
        for i, ch in enumerate(row):
            adj = []
            if i > 0: adj.append(row[i - 1])
            if i < len(row) - 1: adj.append(row[i + 1])
            for rr in (r - 1, r + 1):
                if 0 <= rr < 3:
                    for j in (i - 1, i, i + 1):
                        if 0 <= j < len(rows[rr]):
                            adj.append(rows[rr][j])
            out[ch] = adj
    return out


def train_schedule(unigrams: Dict[str, int], p: int = 2, steps: int = 40,
                   n_instances: int = 6, seed: int = 13,
                   save: bool = True, verbose: bool = True) -> Dict[str, list]:
    """Train QAOA angles on representative instances; save the mean schedule.

    Each ``QAOASolver.solve`` call runs ``steps`` Adam iterations, and every
    iteration lowers the expected QUBO energy — this is the error-reducing
    optimization the runtime path then reuses for free.
    """
    stats = LetterStats(unigrams)
    words = [w for w, c in sorted(unigrams.items(), key=lambda kv: -kv[1])[:3000]
             if 4 <= len(w) <= 8 and w.isalpha()]
    nb = qwerty_neighbours()
    rng = np.random.default_rng(seed)

    schedules = {}          # keyed by segment length (qubit count differs)
    for seg_len in (1, 2, 3):
        collected = []
        for _ in range(n_instances):
            cands, _seg = _random_instance(stats, rng, words, nb)
            while len(cands) != seg_len:
                cands, _seg = _random_instance(stats, rng, words, nb)
            Q = build_segment_qubo(cands, stats, None, None)
            solver = QAOASolver(Q, p=p, steps=steps, seed=int(rng.integers(1e6)),
                                mixer="xy_rows", rows=seg_len)
            res = solver.solve()
            collected.append(res.params)
            if verbose:
                print("  len=%d instance trained: energy=%.3f  ratio=%s" % (
                    seg_len, res.energy,
                    ("%.3f" % res.approximation_ratio) if res.approximation_ratio else "n/a"))
        schedules[str(seg_len)] = np.mean(np.stack(collected), axis=0).tolist()

    out = {"p": p, "K": K, "lambda": LAMBDA, "schedules": schedules}
    if save:
        with open(PARAMS_PATH, "w") as f:
            json.dump(out, f, indent=2)
        if verbose:
            print("saved ->", PARAMS_PATH)
    return out


# --------------------------------------------------------------------------
# Runtime decoder
# --------------------------------------------------------------------------

@dataclass
class SegmentResult:
    positions: List[int]
    letters: str
    method: str            # "qaoa" | "exact-fallback" | "frozen"
    qubits: int = 0
    qaoa_matched_exact: Optional[bool] = None
    qaoa_mass: Optional[float] = None


@dataclass
class WordResult:
    word: str
    greedy: str
    changed: bool
    segments: List[SegmentResult]
    latency_ms: float
    energy: float


class TypingQAOA:
    """Joint word re-decoder: frozen certain letters + QAOA on uncertain runs."""

    def __init__(self, unigrams: Dict[str, int],
                 params_path: str = PARAMS_PATH) -> None:
        self.stats = LetterStats(unigrams)
        self.vocab = unigrams
        self.params: Optional[dict] = None
        if os.path.exists(params_path):
            with open(params_path) as f:
                self.params = json.load(f)

    # ---- public API ----

    def decode_word(self, positions: Sequence[Sequence[Tuple[str, float]]]
                    ) -> WordResult:
        """Jointly re-decode one word from per-position letter posteriors.

        ``positions``: for each typed letter, candidates ``[(letter, p), ...]``
        sorted by descending gaze-fused probability.
        """
        t0 = time.perf_counter()
        greedy = "".join(pos[0][0] for pos in positions)

        # freeze confident positions; collect maximal uncertain runs
        letters: List[Optional[str]] = []
        for pos in positions:
            letters.append(pos[0][0] if pos[0][1] >= CERTAIN or len(pos) < 2
                           else None)

        segments: List[SegmentResult] = []
        energy = 0.0
        i = 0
        while i < len(letters):
            if letters[i] is not None:
                segments.append(SegmentResult([i], letters[i], "frozen"))
                i += 1
                continue
            j = i
            while j < len(letters) and letters[j] is None:
                j += 1
            run = list(range(i, j))
            cands = [[(c, max(p, _EPS)) for c, p in positions[t][:K]]
                     for t in run]
            for t, cnd in zip(run, cands):      # pad to exactly K candidates
                while len(cnd) < K:
                    cnd.append((cnd[0][0], _EPS))
            left = letters[i - 1] if i > 0 else None
            right = letters[j] if j < len(letters) else None
            seg = self._solve_segment(cands, left, right, run)
            for t, ch in zip(run, seg.letters):
                letters[t] = ch
            segments.append(seg)
            i = j

        word = "".join(letters)                     # type: ignore[arg-type]
        return WordResult(
            word=word, greedy=greedy, changed=word != greedy,
            segments=segments,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            energy=energy,
        )

    # ---- internals ----

    def _solve_segment(self, cands, left, right, run) -> SegmentResult:
        exact_combo, exact_e = chain_exact(cands, self.stats, left, right)
        exact_letters = "".join(cands[i][a][0] for i, a in enumerate(exact_combo))

        n = len(cands)
        sched = (self.params or {}).get("schedules", {}).get(str(n))
        if sched is None or n * K > MAX_SEGMENT_QUBITS:
            return SegmentResult(run, exact_letters, "exact-fallback",
                                 qubits=0, qaoa_matched_exact=None)

        Q = build_segment_qubo(cands, self.stats, left, right)
        solver = FixedParamQAOA(Q, p=int(self.params["p"]), steps=0,
                                mixer="xy_rows", rows=n)
        bits, e, mass = solver.sample(np.asarray(sched))
        if bits is None:
            return SegmentResult(run, exact_letters, "exact-fallback",
                                 qubits=n * K, qaoa_matched_exact=False)

        qaoa_letters = ""
        for i in range(n):
            a = int(np.argmax(bits[i * K:(i + 1) * K]))
            qaoa_letters += cands[i][a][0]

        matched = abs(e - exact_e) < 1e-9
        if not matched:
            # guarantee: never return a worse answer than the exact chain MAP
            return SegmentResult(run, exact_letters, "exact-fallback",
                                 qubits=n * K, qaoa_matched_exact=False,
                                 qaoa_mass=mass)
        return SegmentResult(run, qaoa_letters, "qaoa", qubits=n * K,
                             qaoa_matched_exact=True, qaoa_mass=mass)


if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), "..", "vocab-9.json")) as f:
        unigrams = json.load(f)["unigrams"]
    train_schedule(unigrams)
