"""Quantum conditional selection rule for gaze typing.

The problem with a marginal
---------------------------
The deployed keyboard commits a key by fusing two independent pieces of
evidence multiplicatively (``static/app.js::smartCommit``)::

    score(k) = P(k | gaze) * max(P(k | prefix), eps) ** LM_BETA

``P(k | prefix)`` is a *marginal*. It is the total frequency mass of every
vocabulary word that continues the current prefix with letter ``k``, and in
forming that sum it throws away the one thing that distinguishes a confident
continuation from a diffuse one: whether the mass came from a single dominant
word or from fifty mutually unrelated ones. After the prefix ``th``, the letter
``e`` and the letter ``r`` may carry comparable marginal mass while meaning
very different things — ``e`` is essentially one enormous hypothesis (*the*),
``r`` is a wide spray of *thr-* words that agree on nothing beyond their third
character. A product of marginals cannot tell these apart, because the
information was destroyed before the product was taken.

The rule
--------
Keep the hypotheses distinct and let them interfere.

Let ``H`` be the vocabulary completions of the current prefix, each with
frequency weight ``w_h``, each proposing the next letter ``k_h``. Prepare the
candidate register in superposition with amplitudes carrying the gaze
likelihood, then apply a language-conditioned diagonal unitary that attaches to
each hypothesis both a magnitude and a **phase** ``phi_h`` reflecting how well
that hypothesis coheres with the context::

    a(k) = sqrt(P(k|gaze)) * P(k|prefix)**(beta/2) * sum_{h: k_h = k} v_h e^{i phi_h}
    P(k) = |a(k)|**2                                        (Born rule)

where ``v_h = w_h / P(k|prefix)`` are the hypothesis weights *within* letter
``k``, so they sum to one. The trailing factor is the **coherence** of letter
``k``::

    c(k) = sum_h v_h e^{i phi_h}
    |c(k)|**2 = sum_h sum_h' v_h v_h' cos(phi_h - phi_h')

The off-diagonal terms are the interference. They are what a classical mixture
cannot produce: a mixture of the same hypotheses gives ``sum_h v_h**2`` alone,
with no cross-terms and no dependence on phase at all. ``|c(k)| <= 1`` with
equality exactly when every hypothesis supporting ``k`` shares a phase, so the
coherence acts as a penalty on letters that are reachable only through mutually
incoherent continuations.

Containment
-----------
At ``phi = 0`` the coherence is identically 1 and the rule collapses to::

    P(k) ∝ P(k|gaze) * P(k|prefix)**beta

which is the deployed log-linear score, recovered exactly rather than
approximately. This matters: a model class that does not strictly contain the
incumbent cannot be fairly said to beat it, and this project has already been
bitten once by comparing against a baseline the alternative could not express.
``test_conditional_selection.py`` asserts the identity to floating point.

Amplitude amplification
-----------------------
The consistency oracle marks the letters that keep the word inside the
vocabulary — the same guarantee ``clusters.py::is_consistent`` provides for the
cluster decoder, applied to the per-letter path. Amplitude amplification
(Brassard, Hoyer, Mosca & Tapp, 2002) rotates amplitude into that marked
subspace in ``O(1/sqrt(p_S))`` rather than ``O(1/p_S)`` steps, reflecting about
the *prepared* state rather than the uniform one::

    Q = -(2|a><a| - I) . O_S

Operationally this lets the top posterior clear the commit threshold sooner,
which is where a dwell reduction would come from. It fires only when the marked
mass is small — a scattered fixation straying onto keys no vocabulary word
allows. When the prefix is comfortably in vocabulary the optimal round count is
zero and the state is returned untouched.

Everything here is classical state-vector simulation: 27 candidates live in a
32-amplitude (5-qubit) register and the whole rule is a handful of NumPy
operations costing microseconds. No speedup over classical inference is claimed
or implied — the claim is that the *selection rule itself* is different, and
strictly more expressive than the product of marginals it generalises.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

END = "$"  # end-of-word symbol; the keyboard maps it to SPACE

#: Candidate register: a-z plus end-of-word, padded to 2**N_QUBITS amplitudes.
ALPHABET: List[str] = [chr(o) for o in range(ord("a"), ord("z") + 1)] + [END]
N_QUBITS = 5
DIM = 2 ** N_QUBITS
INDEX: Dict[str, int] = {ch: i for i, ch in enumerate(ALPHABET)}

assert len(ALPHABET) <= DIM, "candidate set does not fit the register"


# ==========================================================================
# State preparation
# ==========================================================================

def prepare(p_gaze: Dict[str, float]) -> np.ndarray:
    """Amplitude-encode the gaze likelihood: ``|a> = sum_k sqrt(P(k|gaze)) |k>``.

    Args:
        p_gaze: ``{letter: P(letter | gaze)}``. Need not be normalised; the
            returned state always is. Letters outside ``ALPHABET`` are ignored.

    Returns:
        A complex state vector of length ``DIM`` with unit norm. An all-zero or
        empty input returns the uniform state over the alphabet, which is the
        honest representation of "the gaze tells us nothing".
    """
    amp = np.zeros(DIM, dtype=complex)
    for ch, p in p_gaze.items():
        i = INDEX.get(ch)
        if i is not None and p > 0:
            amp[i] = math.sqrt(p)
    n = np.linalg.norm(amp)
    if n < 1e-12:
        amp[:len(ALPHABET)] = 1.0 / math.sqrt(len(ALPHABET))
        return amp
    return amp / n


# ==========================================================================
# Language-conditioned controlled unitary
# ==========================================================================

def coherence(hypotheses: Sequence[Tuple[str, float, float]]) -> Dict[str, complex]:
    """Per-letter coherence ``c(k) = sum_h v_h e^{i phi_h}``.

    Args:
        hypotheses: ``(next_letter, weight, phase)`` triples. Weights are
            arbitrary positive numbers; they are normalised within each letter,
            so ``c(k)`` is a weighted mean phasor with ``|c(k)| <= 1``.

    Returns:
        ``{letter: complex coherence}``. A letter supported by one hypothesis,
        or by several that agree in phase, gets ``|c| = 1`` and is not
        penalised. Disagreement in phase shrinks it.
    """
    acc: Dict[str, complex] = {}
    mass: Dict[str, float] = {}
    for ch, w, phi in hypotheses:
        if w <= 0:
            continue
        acc[ch] = acc.get(ch, 0j) + w * complex(math.cos(phi), math.sin(phi))
        mass[ch] = mass.get(ch, 0.0) + w
    return {ch: acc[ch] / mass[ch] for ch in acc if mass[ch] > 0}


def condition(state: np.ndarray, prior: Dict[str, float],
              hypotheses: Sequence[Tuple[str, float, float]],
              beta: float, floor: float = 1e-4) -> np.ndarray:
    """Apply the language-conditioned diagonal unitary.

    Each candidate amplitude is multiplied by ``P(k|prefix)**(beta/2)`` — the
    magnitude channel, exponent-damped exactly as the deployed rule damps it so
    the language model advises rather than vetoes — and by the coherence
    ``c(k)``, which is the phase channel.

    Args:
        state: prepared candidate register.
        prior: ``P(next letter | prefix)`` over ``ALPHABET``.
        hypotheses: ``(next_letter, weight, phase)`` triples backing that prior.
            Pass an empty sequence for the zero-phase limit.
        beta: language exponent (``LM_BETA`` in the deployed rule).
        floor: minimum prior. Unfloored, a zero prior is an absolute veto and
            no amount of gaze certainty could type an out-of-vocabulary letter.

    Returns:
        The conditioned state, renormalised. Renormalisation is required: the
        diagonal is a contraction, not a unitary, once the coherence damps a
        letter — physically this is a post-selected measurement, which is how
        the classical simulation implements conditioning without the ancillae a
        strictly unitary dilation would need.
    """
    c = coherence(hypotheses)
    out = state.copy()
    for ch, i in INDEX.items():
        if out[i] == 0:
            continue
        mag = max(prior.get(ch, 0.0), floor) ** (beta / 2.0)
        out[i] *= mag * (c.get(ch, 1.0 + 0j))
    n = np.linalg.norm(out)
    return out if n < 1e-12 else out / n


# ==========================================================================
# Amplitude amplification
# ==========================================================================

def optimal_rounds(marked_mass: float, cap: int = 2) -> int:
    """Round count ``floor((pi/4)/asin(sqrt(p)))``, clamped to ``[0, cap]``.

    The textbook count for amplitude amplification. Capped because overshoot is
    real — past the optimum the rotation carries amplitude back *out* of the
    marked subspace — and because the marked mass here is estimated from a
    prior, not measured, so a long schedule would compound that error.
    """
    p = float(np.clip(marked_mass, 1e-12, 1.0))
    if p >= 0.5:
        return 0
    theta = math.asin(math.sqrt(p))
    return int(np.clip(math.floor((math.pi / 4.0) / theta), 0, cap))


def amplify(state: np.ndarray, marked: Sequence[str],
            rounds: Optional[int] = None, cap: int = 2) -> Tuple[np.ndarray, int]:
    """Rotate amplitude into the oracle-marked subspace.

    Implements ``Q = -(2|a><a| - I) . O_S`` with the reflection taken about the
    *prepared* state ``|a>`` rather than the uniform superposition, which is the
    general amplitude-amplification form and the correct one when the initial
    distribution is non-uniform (it is: it carries the gaze likelihood).

    Args:
        state: the conditioned register.
        marked: letters the consistency oracle accepts.
        rounds: explicit round count; ``None`` selects the optimal count from
            the currently marked mass.
        cap: maximum rounds when choosing automatically.

    Returns:
        ``(amplified state, rounds actually applied)``. Zero rounds returns the
        input unchanged, which is the common case: when the prefix is well
        inside the vocabulary the marked mass is already large and there is
        nothing to amplify.
    """
    idx = [INDEX[ch] for ch in marked if ch in INDEX]
    if not idx:
        return state, 0
    mask = np.zeros(DIM, dtype=bool)
    mask[idx] = True

    p_marked = float(np.sum(np.abs(state[mask]) ** 2))
    k = optimal_rounds(p_marked, cap) if rounds is None else int(rounds)
    if k <= 0:
        return state, 0

    ref = state.copy()          # |a>, the reflection axis
    cur = state.copy()
    for _ in range(k):
        cur[mask] = -cur[mask]                       # oracle O_S
        cur = 2.0 * np.vdot(ref, cur) * ref - cur    # reflect about |a>
    n = np.linalg.norm(cur)
    return (cur if n < 1e-12 else cur / n), k


# ==========================================================================
# Measurement
# ==========================================================================

def measure(state: np.ndarray) -> Dict[str, float]:
    """Born-rule readout: ``P(k) = |a(k)|**2``, normalised over the alphabet."""
    p = np.abs(state[:len(ALPHABET)]) ** 2
    z = float(p.sum())
    if z < 1e-12:
        return {ch: 1.0 / len(ALPHABET) for ch in ALPHABET}
    return {ch: float(p[i] / z) for i, ch in enumerate(ALPHABET)}


# ==========================================================================
# Phase assignment
# ==========================================================================

def phases(completions: Sequence[Tuple[str, float]], scale: float) -> List[float]:
    """Assign ``phi_h = scale * pi * (1 - s_h)`` from per-character surprisal.

    ``s_h`` is ``log(w_h) / len(h)`` — the hypothesis's mean per-character
    log-frequency — rescaled to ``[0, 1]`` across the hypothesis set. This is
    the path-action analogue: a hypothesis accumulates phase along its own
    continuation, and two hypotheses are in phase when their continuations are
    equally natural, not merely when they are equally frequent.

    Length has to enter. Phase driven by raw weight alone is degenerate in the
    exact case this mechanism exists to catch: a diffuse spray of comparably
    frequent completions would be assigned one common phase, would interfere
    perfectly constructively, and would be indistinguishable from a single
    confident hypothesis. Dividing by length separates *the* (enormous, short,
    unsurprising) from *threatened* (rare, long, and surprising per character)
    even where their frequency mass is similar.

    ``scale = 0`` puts every hypothesis at phase 0, which is the classical
    zero-phase limit. It is the single knob that interpolates between the
    deployed rule and the interference rule.
    """
    if not len(completions):
        return []
    lg = np.array([math.log(max(w, 1e-300)) / max(len(word), 1)
                   for word, w in completions], dtype=float)
    lo, hi = float(lg.min()), float(lg.max())
    if hi - lo < 1e-12:
        return [0.0] * len(completions)
    s = (lg - lo) / (hi - lo)
    return [float(scale * math.pi * (1.0 - v)) for v in s]


# ==========================================================================
# The whole rule
# ==========================================================================

def select(p_gaze: Dict[str, float], prior: Dict[str, float],
           completions: Sequence[Tuple[str, float]], prefix: str = "",
           *, beta: float = 0.70, floor: float = 1e-4,
           phase_scale: float = 0.0, amplify_rounds: Optional[int] = None,
           amplify_cap: int = 0, amplify_stage: str = "pre",
           oracle: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """Run prepare -> condition -> amplify -> measure and return the posterior.

    Args:
        p_gaze: ``{letter: P(letter | gaze)}`` from the key-rectangle integral.
        prior: ``{letter: P(letter | prefix)}`` from the language model.
        completions: ``(word, weight)`` vocabulary continuations of ``prefix``.
            These are the hypotheses; their next letter is the character at
            position ``len(prefix)``.
        prefix: the word typed so far, used to read each completion's next
            letter. A completion no longer than the prefix proposes ``END``.
        beta: language exponent.
        floor: prior floor keeping the language model advisory.
        phase_scale: 0 reproduces the deployed log-linear rule exactly; larger
            values strengthen interference.
        amplify_rounds: explicit amplification rounds, or ``None`` for optimal.
        amplify_cap: maximum automatic rounds; 0 disables amplification.
        amplify_stage: ``"pre"`` amplifies the raw gaze register before the
            language model weighs in; ``"post"`` amplifies the conditioned
            state. The distinction is not cosmetic. After conditioning, the
            language prior has already placed 94-98% of the amplitude on
            consistent letters, the optimal round count is zero, and
            amplification is a silent no-op. Before conditioning the marked mass
            on a scattered fixation is 0.31-0.78 and there is genuinely
            something to rotate.
        oracle: letters the consistency oracle accepts — the continuations that
            keep the word inside the vocabulary. Defaults to the next letters of
            ``completions``, which is the trie's own answer and the per-letter
            analogue of ``clusters.py::is_consistent``. Passing the letters whose
            *prior* clears the floor instead would mark everything, because the
            smoothed character n-gram gives every letter non-zero mass, and
            amplification would be a silent no-op.

    Returns:
        ``{letter: posterior}`` summing to 1 over ``ALPHABET``.
    """
    hyps: List[Tuple[str, float, float]] = []
    if phase_scale > 0.0 and completions:
        phi = phases(completions, phase_scale)
        n = len(prefix)
        for (word, w), ph in zip(completions, phi):
            hyps.append((word[n] if len(word) > n else END, w, ph))

    amplifying = amplify_cap > 0 or bool(amplify_rounds)
    if amplifying:
        if oracle is not None:
            marked = list(oracle)
        else:
            n = len(prefix)
            marked = list({word[n] if len(word) > n else END
                           for word, _ in completions})

    state = prepare(p_gaze)
    if amplifying and amplify_stage == "pre":
        state, _ = amplify(state, marked, amplify_rounds, amplify_cap)
    state = condition(state, prior, hyps, beta, floor)
    if amplifying and amplify_stage == "post":
        state, _ = amplify(state, marked, amplify_rounds, amplify_cap)
    return measure(state)


def loglinear(p_gaze: Dict[str, float], prior: Dict[str, float],
              beta: float = 0.70, floor: float = 1e-4) -> Dict[str, float]:
    """The deployed incumbent, written out for head-to-head comparison.

    Mirrors ``static/app.js::smartCommit``: gaze posterior times floored,
    exponent-damped language prior, renormalised.
    """
    out: Dict[str, float] = {}
    for ch in ALPHABET:
        g = p_gaze.get(ch, 0.0)
        if g <= 0:
            continue
        out[ch] = g * (max(prior.get(ch, 0.0), floor) ** beta)
    z = sum(out.values())
    if z <= 0:
        return {ch: 1.0 / len(ALPHABET) for ch in ALPHABET}
    return {ch: v / z for ch, v in out.items()}
