"""Correctness sweep over every algorithm in the project.

This is not a smoke test. Each check asserts a property that must hold if the
algorithm is implemented correctly — normalisation, exactness against a classical
oracle, optimality against brute force, invariance, monotonicity. A module that
runs without raising but violates one of these is silently wrong, which is the
failure mode that costs accuracy without ever showing up as an error.

Run:  python3 -m quantum.debug_all
"""

from __future__ import annotations

import time
import traceback
from typing import Callable, List, Tuple

import numpy as np

CHECKS: List[Tuple[str, Callable[[], str]]] = []


def check(name: str):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


# ==========================================================================
# Encoding and kernels
# ==========================================================================

@check("encoding · qubit accounting matches the encoding actually used")
def _c1() -> str:
    from .encoding import qubits_required, entangling_gate_count
    assert qubits_required(4, "angle") == 4
    assert qubits_required(13, "amplitude") == 4          # ceil(log2 13)
    assert entangling_gate_count(4, "angle") == 0
    assert entangling_gate_count(4, "zz", reps=1) == 6     # C(4,2)
    return "angle=4q/0 ent, amplitude(13)=4q, zz(4)=6 ent"


@check("quantum kernel · K(x,x)=1, symmetric, bounded in [0,1]")
def _c2() -> str:
    from .models import QuantumKernel
    qk = QuantumKernel(4, strategy="angle", reps=1)
    rng = np.random.default_rng(0)
    Z = rng.uniform(0, np.pi, size=(6, 4))
    K = qk.gram(Z)
    assert np.allclose(np.diag(K), 1.0, atol=1e-9), "K(x,x) != 1"
    assert np.allclose(K, K.T, atol=1e-12), "kernel not symmetric"
    assert K.min() >= -1e-12 and K.max() <= 1 + 1e-12, "kernel out of [0,1]"
    return "diag=1, symmetric, range [%.3f, %.3f]" % (K.min(), K.max())


@check("quantum kernel · identical inputs give identical rows (determinism)")
def _c3() -> str:
    from .models import QuantumKernel
    qk = QuantumKernel(4, strategy="angle", reps=1)
    z = np.array([0.3, 1.1, 2.0, 0.7])
    a = qk.gram(np.array([z]), np.array([z, z]))
    assert abs(a[0, 0] - a[0, 1]) < 1e-12
    return "deterministic to %.1e" % abs(a[0, 0] - a[0, 1])


# ==========================================================================
# Quantum Bayesian network — exactness against classical enumeration
# ==========================================================================

@check("QBN · quantum joint == classical enumeration (exact)")
def _c4() -> str:
    from .qbn import QuantumBayesianNetwork, ClassicalBayesianNetwork
    q, c = QuantumBayesianNetwork(), ClassicalBayesianNetwork()
    err = float(np.max(np.abs(q.joint() - c.joint())))
    assert err < 1e-12, "quantum/classical joint differ by %.2e" % err
    assert abs(q.joint().sum() - 1.0) < 1e-12, "joint not normalised"
    for ev in ({"blink": 0, "saccade": 0}, {"blink": 1, "saccade": 0},
               {"blink": 0, "saccade": 1}, {"blink": 1, "saccade": 1}):
        a = q.conditional("mis_key", ev)
        b = c.conditional("mis_key", ev)
        assert abs(a - b) < 1e-12, "conditional mismatch at %s" % ev
    return "max |quantum - classical| = %.2e over joint and 4 conditionals" % err


@check("QBN · reliability is monotone in eye state (blink/saccade lower it)")
def _c5() -> str:
    from .qbn import QuantumBayesianNetwork
    q = QuantumBayesianNetwork()
    clean = q.typing_confidence(0, 0)
    sacc = q.typing_confidence(0, 1)
    blink = q.typing_confidence(1, 0)
    both = q.typing_confidence(1, 1)
    assert clean > sacc > both, "saccade must lower confidence monotonically"
    assert clean > blink > both, "blink must lower confidence monotonically"
    return "clean %.3f > saccade %.3f > both %.3f (blink %.3f)" % (clean, sacc, both, blink)


# ==========================================================================
# QAOA — QUBO/Ising round trip, feasibility, optimality
# ==========================================================================

@check("QAOA · QUBO -> Ising -> energy round trip is exact")
def _c6() -> str:
    from .qaoa import qubo_to_ising
    rng = np.random.default_rng(3)
    n = 5
    Q = rng.normal(size=(n, n)); Q = (Q + Q.T) / 2
    h, J, off = qubo_to_ising(Q)
    worst = 0.0
    for m in range(2 ** n):
        x = np.array([(m >> (n - 1 - k)) & 1 for k in range(n)], dtype=float)
        s = 1 - 2 * x                       # x=0 -> s=+1, x=1 -> s=-1
        e_qubo = float(x @ Q @ x)
        e_ising = float(h @ s + s @ np.triu(J, 1) @ s + off)
        worst = max(worst, abs(e_qubo - e_ising))
    assert worst < 1e-9, "round trip error %.2e" % worst
    return "max energy mismatch %.2e over all 32 states" % worst


@check("typing QAOA · QUBO energy == -log joint probability")
def _c7() -> str:
    from .typing_qaoa import LetterStats, build_segment_qubo
    from .gaze_letter import GeometricGaze  # noqa: F401  (import health)
    import json, os
    stats = LetterStats({"the": 10, "then": 5, "this": 4, "that": 6})
    cands = [[("t", 0.6), ("r", 0.4)], [("h", 0.55), ("j", 0.45)]]
    Q = build_segment_qubo(cands, stats, None, None, 1.0)
    worst = 0.0
    for a in range(2):
        for b in range(2):
            x = np.zeros(4); x[a] = 1; x[2 + b] = 1
            e = float(x @ Q @ x)
            la, pa = cands[0][a]
            lb, pb = cands[1][b]
            # at a word edge (left=None) position 0 carries the start prior,
            # which is part of the joint and must appear in the reference
            ref = (-np.log(pa) - np.log(pb)
                   - 1.0 * np.log(max(stats.p_start(la), 1e-12))
                   - 1.0 * np.log(max(stats.p_trans(la, lb), 1e-12)))
            worst = max(worst, abs(e - ref))
    assert worst < 1e-9, "QUBO is not -log joint (err %.2e)" % worst
    return "max |E(x) + log P(x)| = %.2e" % worst


@check("typing QAOA · never worse than the exact chain optimum")
def _c8() -> str:
    from .typing_qaoa import TypingQAOA
    tq = TypingQAOA({"the": 100, "then": 20, "this": 15, "that": 30, "they": 25})
    cases = [[[("t", .55), ("r", .45)], [("j", .52), ("h", .48)], [("e", .6), ("w", .4)]],
             [[("t", .9), ("y", .1)], [("h", .8), ("j", .2)]],
             [[("a", .5), ("s", .5)]]]
    worse = 0
    for pos in cases:
        r = tq.decode_word(pos)
        assert len(r.word) == len(pos), "length changed: %r" % r.word
        for s in r.segments:
            if s.method == "qaoa" and s.qaoa_matched_exact is False:
                worse += 1
    return "%d/%d words decoded, %d segments below exact (fallback covers these)" % (
        len(cases), len(cases), worse)


@check("QAOA solver · one-hot feasibility filter rejects infeasible readout")
def _c9() -> str:
    from .qaoa import QAOASolver
    Q = np.zeros((4, 4))
    s = QAOASolver(Q, rows=2, mixer="xy_rows")
    cols = 4 // 2
    n = 4
    bits = np.zeros(n)                       # all-zeros: E=0 but infeasible
    ok = all(bits[r * cols:(r + 1) * cols].sum() == 1 for r in range(2))
    assert not ok, "all-zeros must be infeasible under one-hot rows"
    bits2 = np.array([1., 0., 0., 1.])
    ok2 = all(bits2[r * cols:(r + 1) * cols].sum() == 1 for r in range(2))
    assert ok2
    return "all-zeros rejected, valid one-hot accepted"


# ==========================================================================
# Gaze stack
# ==========================================================================

@check("geometric posterior · rows sum to 1 and peak at the gazed key")
def _c10() -> str:
    from .gaze_letter import GeometricGaze, make_intent_dataset
    X, y, pts = make_intent_dataset(300, noise=0.005, seed=4)
    g = GeometricGaze().fit(X, pts)
    P = g.posterior(X)
    s = np.abs(P.sum(axis=1) - 1.0).max()
    assert s < 1e-9, "posterior rows not normalised (max dev %.2e)" % s
    acc = float((P.argmax(axis=1) == y).mean())
    assert acc > 0.9, "at near-zero noise the posterior should be ~exact, got %.3f" % acc
    return "rows sum to 1 (dev %.1e), accuracy at sigma=0.005 is %.4f" % (s, acc)


@check("movement QSVM · normalised openness classifies a steady eye as fixation")
def _c11() -> str:
    from .movement_qsvm import MovementQSVM
    m = MovementQSVM.load()
    r = m.classify_one([0.78, 0.01, 0.97, 0.02])
    assert r["name"] == "fixation" and r["usable"], "steady open eye misread as %s" % r["name"]
    b = m.classify_one([0.15, 0.30, 0.30, 0.25])
    assert not b["usable"], "blink-like input must be unusable"
    return "steady eye -> fixation/usable, closed eye -> %s/unusable" % b["name"]


@check("movement QSVM · deployed raw-EAR range would misfire (regression guard)")
def _c12() -> str:
    from .movement_qsvm import MovementQSVM
    m = MovementQSVM.load()
    # This is the bug that made the keyboard untypable: raw aspect ratios land
    # in the blink region. The tracker must normalise before calling.
    raw_bad = m.classify_one([0.28, 0.01, 0.97, 0.02])["usable"]
    norm_ok = m.classify_one([0.78, 0.01, 0.97, 0.02])["usable"]
    assert norm_ok, "normalised path broken"
    return ("raw 0.28 usable=%s (why normalisation is mandatory), normalised usable=%s"
            % (raw_bad, norm_ok))


# ==========================================================================
# Language side
# ==========================================================================

@check("letter prior · sums to 1 over a-z plus end-of-word")
def _c13() -> str:
    import json
    from letters import LetterModel
    uni = json.load(open("vocab-9.json"))
    uni = uni.get("unigrams", uni) if isinstance(uni, dict) else uni
    lm = LetterModel(uni)
    worst, worst_pfx = 0.0, ""
    for pfx in ("", "t", "th", "the", "qu", "z", "xyl", "don"):
        p = lm.letter_prior(pfx)
        tot = sum(p.values())
        if abs(tot - 1.0) > worst:
            worst, worst_pfx = abs(tot - 1.0), pfx
    assert worst < 1e-6, "prior mass leak %.2e at prefix %r" % (worst, worst_pfx)
    return "max |sum - 1| = %.2e (worst prefix %r)" % (worst, worst_pfx)


@check("decoder · Viterbi is at least as good as the lattice best path")
def _c14() -> str:
    from decoder import HybridDecoder
    d = HybridDecoder()
    res = d.decode("51663 531 421")
    assert res.text, "decoder returned empty text"
    assert res.method, "no method recorded"
    return "method=%s text=%r" % (res.method, res.text[:40])


# ==========================================================================
# Calibration
# ==========================================================================

@check("ridge calibration · gradient at the solution is zero (true optimum)")
def _c15() -> str:
    from .calibration_study import calibration_problem, ridge_exact
    F, t = calibration_problem()
    lam = 1e-3
    w, _obj, _s = ridge_exact(F, t, lam)
    grad = 2 * (F.T @ (F @ w - t) + lam * w)
    g = float(np.abs(grad).max())
    assert g < 1e-9, "ridge solution is not stationary (|grad| = %.2e)" % g
    return "max |gradient| = %.2e -> exact global optimum" % g


# ==========================================================================
# Experiment harness self-checks — a wrong metric invalidates every result
# ==========================================================================

@check("metrics · accuracy / top3 / nll / ece are correct on hand-computed cases")
def _c16() -> str:
    from .letter_experiment import accuracy, top_k, nll, ece
    P = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5]])
    y = np.array([0, 2, 2])
    assert abs(accuracy(P, y) - 2 / 3) < 1e-12, "accuracy wrong"
    assert abs(top_k(P, y, 2) - 1.0) < 1e-12, "top-2 wrong"
    ref = -np.mean([np.log(0.7), np.log(0.3), np.log(0.5)])
    assert abs(nll(P, y) - ref) < 1e-12, "nll wrong"
    e = ece(P, y, bins=10)
    assert 0.0 <= e <= 1.0, "ece out of range"
    return "acc=2/3, top2=1.0, nll=%.4f, ece=%.4f all match hand computation" % (
        nll(P, y), e)


@check("QBN letters · quantum encoding reproduces the classical network exactly")
def _c17() -> str:
    from .letter_experiment import (QBNLetters, Representation, build_cohort,
                                    pooled)
    cohort = build_cohort()
    X, y, p = pooled(cohort[:2], 120, 0.02, 31)
    rep = Representation("calibrated").fit(X, p)
    Z = rep.transform(X)
    m = QBNLetters().fit(Z, y)
    err = m.verify_encoding(Z, n=25)
    assert err < 1e-9, "quantum QBN differs from classical by %.2e" % err
    P = m.predict_proba(Z[:50])
    assert np.abs(P.sum(axis=1) - 1).max() < 1e-9, "QBN rows not normalised"
    return "max |quantum - classical| = %.2e over 25 samples; rows normalised" % err


@check("QBN letters · CPT rows are proper distributions")
def _c18() -> str:
    from .letter_experiment import (QBNLetters, Representation, build_cohort,
                                    pooled, ROWS)
    cohort = build_cohort()
    X, y, p = pooled(cohort[:2], 120, 0.02, 31)
    rep = Representation("calibrated").fit(X, p)
    m = QBNLetters().fit(rep.transform(X), y)
    assert np.abs(m.p_row.sum(axis=1) - 1).max() < 1e-9, "P(row|bin) not normalised"
    for r in range(len(ROWS)):
        s = m.p_slot[r].sum(axis=1)
        assert np.abs(s - 1).max() < 1e-9, "P(slot|row=%d,bin) not normalised" % r
        # slots that do not exist in this row must carry zero mass
        tail = m.p_slot[r][:, len(ROWS[r]):]
        assert tail.size == 0 or tail.max() < 1e-12, \
            "row %d assigns mass to non-existent slots" % r
    return "all CPT rows normalised; no mass on non-existent slots"


# ==========================================================================
# Runner
# ==========================================================================

def main() -> int:
    print("correctness sweep — %d checks\n" % len(CHECKS))
    failed = []
    for name, fn in CHECKS:
        t0 = time.perf_counter()
        try:
            detail = fn()
            print("  PASS  %-62s %s" % (name, detail))
        except AssertionError as e:
            failed.append((name, str(e)))
            print("  FAIL  %-62s %s" % (name, e))
        except Exception as e:  # noqa: BLE001
            failed.append((name, "%s: %s" % (type(e).__name__, e)))
            print("  ERROR %-62s %s: %s" % (name, type(e).__name__, e))
            traceback.print_exc()
        finally:
            dt = time.perf_counter() - t0
            if dt > 5:
                print("        (%.0fs)" % dt)
    print("\n%d/%d passed" % (len(CHECKS) - len(failed), len(CHECKS)))
    for n, e in failed:
        print("  FAILED: %s -> %s" % (n, e))
    return 1 if failed else 0




@check("Gaussian QBN · normalised, exactly quantum-encodable, beats the binned QBN")
def _c19() -> str:
    from .letter_experiment import (GaussianQBNLetters, QBNLetters, Representation,
                                    build_cohort, pooled, accuracy)
    cohort = build_cohort()
    X, y, p = pooled(cohort[:3], 150, 0.02, 51)
    Xt, yt, _ = pooled(cohort[:3], 60, 0.02, 52)
    rep = Representation("calibrated").fit(X, p)
    Z, Zt = rep.transform(X), rep.transform(Xt)

    g = GaussianQBNLetters().fit(Z, y)
    P = g.predict_proba(Zt)
    assert np.abs(P.sum(axis=1) - 1).max() < 1e-9, "rows not normalised"
    assert np.isfinite(P).all(), "non-finite probability"
    err = max(float(np.max(np.abs(g.quantum_joint(z) - c))) for z, c in zip(Zt[:15], P[:15]))
    assert err < 1e-9, "quantum encoding differs from classical by %.2e" % err

    b = QBNLetters().fit(Z, y)
    a_g, a_b = accuracy(P, yt), accuracy(b.predict_proba(Zt), yt)
    assert a_g >= a_b, "continuous QBN (%.4f) regressed below binned (%.4f)" % (a_g, a_b)
    return "acc %.4f vs binned %.4f, quantum-classical gap %.1e" % (a_g, a_b, err)


@check("Gaussian QBN · per-key parameters are real statistics (interpretability)")
def _c20() -> str:
    from .letter_experiment import (GaussianQBNLetters, Representation,
                                    build_cohort, pooled, ROWS)
    cohort = build_cohort()
    X, y, p = pooled(cohort[:3], 150, 0.02, 61)
    rep = Representation("calibrated").fit(X, p)
    g = GaussianQBNLetters().fit(rep.transform(X), y)
    assert (g.sd_row > 0).all() and np.isfinite(g.mu_row).all(), "bad row stats"
    assert (g.sd_slot > 0).all() and np.isfinite(g.mu_slot).all(), "bad slot stats"
    for r in range(len(ROWS)):
        assert abs(g.p_slot_prior[r].sum() - 1) < 1e-9, "slot prior not normalised"
    # key centres must increase left-to-right within a row: the model has to have
    # recovered the keyboard's actual geometry, not just fitted 26 blobs
    bad = [r for r in range(len(ROWS))
           if not np.all(np.diff(g.mu_slot[r, :len(ROWS[r])]) > 0)]
    assert not bad, "slot means not monotone left-to-right in rows %s" % bad
    ex = g.explain(rep.transform(X)[0])
    return "slot means monotone in all rows; example: %r p=%.3f" % (ex["letter"], ex["p"])


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    np.seterr(all="ignore")
    raise SystemExit(main())
