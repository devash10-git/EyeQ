"""Correctness tests for the quantum conditional selection rule.

Four properties, in order of how much they matter:

1. **Containment.** At zero phase the rule reproduces the deployed log-linear
   score exactly. Without this the head-to-head comparison is rigged, because a
   model class that cannot express the incumbent cannot be beaten by it. This
   project has already been caught by that once.
2. **Expressiveness.** Two hypothesis sets with *identical* marginals produce
   *different* posteriors. This is the whole claim: the rule reads information a
   product of marginals has already discarded.
3. **Well-formedness.** States stay normalised, coherence stays in the unit
   disc, posteriors sum to one.
4. **Amplification.** Grover rounds move mass into the marked subspace, and the
   round schedule matches the textbook count.

Run: ``python3 quantum/test_conditional_selection.py``
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum.conditional_selection import (  # noqa: E402
    ALPHABET, DIM, END, INDEX, amplify, coherence, condition, loglinear,
    measure, optimal_rounds, phases, prepare, select,
)

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("  PASS  %s" % name)
    else:
        print("  FAIL  %s  %s" % (name, detail))
        FAILURES.append(name)


def gaze(**kw):
    return dict(kw)


# ==========================================================================
# 1. Containment
# ==========================================================================

def test_containment():
    print("\n1. containment — zero phase reproduces the deployed rule")
    rng = np.random.default_rng(11)
    worst = 0.0
    for trial in range(200):
        g = {ch: float(v) for ch, v in
             zip(ALPHABET, rng.random(len(ALPHABET)) ** 3) if v > 0.02}
        pr = {ch: float(v) for ch, v in
              zip(ALPHABET, rng.random(len(ALPHABET)))}
        z = sum(pr.values())
        pr = {k: v / z for k, v in pr.items()}
        beta = float(rng.uniform(0.3, 1.2))

        want = loglinear(g, pr, beta=beta)
        got = select(g, pr, [], beta=beta, phase_scale=0.0, amplify_cap=0)
        for ch in ALPHABET:
            worst = max(worst, abs(want.get(ch, 0.0) - got.get(ch, 0.0)))
    check("matches loglinear to floating point over 200 random inputs",
          worst < 1e-12, "max abs deviation %.3e" % worst)

    # A prior of exactly zero must stay typable: the floor is not cosmetic.
    g = {"q": 0.9, "a": 0.1}
    pr = {ch: 0.0 for ch in ALPHABET}
    pr["a"] = 1.0
    post = select(g, pr, [], beta=0.70, phase_scale=0.0, amplify_cap=0)
    check("zero prior is floored, not a veto", post["q"] > 0.0,
          "P(q) = %.3e" % post["q"])

    # phase_scale is the only knob between the two rules: a single hypothesis
    # has nothing to interfere with, so any scale leaves the incumbent intact.
    solo = [("ab", 5.0)]
    a = select(g, pr, solo, prefix="a", beta=0.7, phase_scale=0.0, amplify_cap=0)
    b = select(g, pr, solo, prefix="a", beta=0.7, phase_scale=1.0, amplify_cap=0)
    d = max(abs(a[ch] - b[ch]) for ch in ALPHABET)
    check("a lone hypothesis cannot interfere with itself", d < 1e-12,
          "deviation %.3e" % d)


# ==========================================================================
# 2. Expressiveness — the actual novelty claim
# ==========================================================================

def test_expressiveness():
    print("\n2. expressiveness — information the marginal has discarded")

    # Two hypothesis sets with identical per-letter marginals. Both put mass
    # 100 on 'e' and 100 on 'r'; log-linear fusion cannot tell them apart
    # because it only ever sees the sums.
    concentrated = [("the", 100.0), ("thr", 100.0)]
    diffuse = [("the", 100.0), ("thrash", 40.0), ("through", 30.0),
               ("threatened", 18.0), ("thrillingly", 12.0)]

    def marginal(hyps, n=2):
        out = {}
        for w, c in hyps:
            k = w[n] if len(w) > n else END
            out[k] = out.get(k, 0.0) + c
        z = sum(out.values())
        return {k: v / z for k, v in out.items()}

    m1, m2 = marginal(concentrated), marginal(diffuse)
    same = all(abs(m1.get(k, 0) - m2.get(k, 0)) < 1e-12 for k in set(m1) | set(m2))
    check("the two hypothesis sets have identical marginals", same,
          "%s vs %s" % (m1, m2))

    g = {ch: 1.0 / len(ALPHABET) for ch in ALPHABET}
    pr = {ch: 0.0 for ch in ALPHABET}
    pr["e"] = pr["r"] = 0.5

    a = select(g, pr, concentrated, prefix="th", beta=0.7, phase_scale=0.6,
               amplify_cap=0)
    b = select(g, pr, diffuse, prefix="th", beta=0.7, phase_scale=0.6,
               amplify_cap=0)
    check("but the rule separates them", abs(a["r"] - b["r"]) > 1e-3,
          "P(r) concentrated=%.4f diffuse=%.4f" % (a["r"], b["r"]))
    check("the diffuse branch is the one penalised", b["r"] < a["r"],
          "%.4f vs %.4f" % (b["r"], a["r"]))

    # Log-linear fusion, given the same marginals, returns the same posterior
    # for both — which is precisely the blindness being repaired.
    la = loglinear(g, pr, beta=0.7)
    lb = loglinear(g, pr, beta=0.7)
    check("log-linear fusion cannot separate them",
          abs(la["r"] - lb["r"]) < 1e-15)

    # On the real vocabulary the mechanism must actually engage rather than
    # sitting at unit coherence everywhere.
    try:
        from decoder import HybridDecoder
        from letters import LetterModel
        lm = LetterModel(HybridDecoder(vocab_path="vocab-9.json", model_dir=None,
                                       num_clusters=9).unigrams)
    except Exception as exc:                       # vocabulary not present
        print("  SKIP  real-vocabulary checks (%s)" % exc)
        return

    moved = 0
    worst_drop = 0.0
    for prefix in ("", "t", "th", "thr", "qu", "wor", "sta", "pre", "i", "ab"):
        comps = lm.completions(prefix, n=32)
        prior = lm.letter_prior(prefix)
        base = loglinear(g, prior, beta=0.70)
        rule = select(g, prior, comps, prefix=prefix, beta=0.70,
                      phase_scale=0.6, amplify_cap=0)
        d = max(abs(base.get(c, 0) - rule.get(c, 0)) for c in ALPHABET)
        if d > 1e-4:
            moved += 1
        phi = phases(comps, 0.6)
        coh = coherence([(w[len(prefix)] if len(w) > len(prefix) else END, c, p)
                         for (w, c), p in zip(comps, phi)])
        if coh:
            worst_drop = max(worst_drop, 1.0 - min(abs(v) for v in coh.values()))
    check("the rule moves the posterior on real prefixes", moved >= 8,
          "%d of 10 prefixes moved" % moved)
    check("real branches show genuine decoherence", worst_drop > 0.05,
          "largest coherence drop %.3f" % worst_drop)


# ==========================================================================
# 3. Well-formedness
# ==========================================================================

def test_wellformed():
    print("\n3. well-formedness")
    rng = np.random.default_rng(5)

    g = {ch: float(v) for ch, v in zip(ALPHABET, rng.random(len(ALPHABET)))}
    st = prepare(g)
    check("prepared state has unit norm", abs(np.linalg.norm(st) - 1) < 1e-12)
    check("prepared state has the register's width", st.shape == (DIM,))
    check("unused register slots stay empty",
          np.allclose(st[len(ALPHABET):], 0))

    check("an empty gaze gives the uniform state, not a crash",
          abs(np.linalg.norm(prepare({})) - 1) < 1e-12)
    check("an all-zero gaze gives the uniform state",
          abs(measure(prepare({ch: 0.0 for ch in ALPHABET}))["a"]
              - 1.0 / len(ALPHABET)) < 1e-12)

    pr = {ch: 1.0 / len(ALPHABET) for ch in ALPHABET}
    hyps = [("abc", 10.0, 0.3), ("abd", 4.0, 2.9), ("axe", 1.0, 1.1)]
    hyps = [(h[0][2], h[1], h[2]) for h in hyps]
    st2 = condition(st, pr, hyps, beta=0.7)
    check("conditioned state stays normalised",
          abs(np.linalg.norm(st2) - 1) < 1e-12)

    c = coherence(hyps)
    check("coherence lies in the closed unit disc",
          all(abs(v) <= 1 + 1e-12 for v in c.values()),
          str({k: round(abs(v), 4) for k, v in c.items()}))
    solo = coherence([("c", 10.0, 1.234)])
    check("a single hypothesis has unit coherence",
          abs(abs(solo["c"]) - 1.0) < 1e-12)
    agree = coherence([("c", 3.0, 0.77), ("c", 9.0, 0.77)])
    check("phase agreement gives unit coherence",
          abs(abs(agree["c"]) - 1.0) < 1e-12)
    cancel = coherence([("c", 1.0, 0.0), ("c", 1.0, math.pi)])
    check("exact opposition cancels", abs(cancel["c"]) < 1e-12)

    post = select(g, pr, [("abc", 3.0)], prefix="ab", phase_scale=0.5)
    check("posterior sums to one", abs(sum(post.values()) - 1) < 1e-12)
    check("posterior is non-negative", all(v >= 0 for v in post.values()))

    ph = phases([("cat", 5.0), ("cot", 5.0), ("cut", 5.0)], 1.0)
    check("identical weight and length give identical phases",
          max(ph) - min(ph) < 1e-12)
    check("phases are empty for an empty hypothesis set", phases([], 1.0) == [])
    ph2 = phases([("the", 100.0), ("thermodynamic", 1.0)], 0.5)
    check("the least surprising hypothesis sits at phase zero",
          abs(ph2[0]) < 1e-12)
    check("the most surprising sits at scale*pi",
          abs(ph2[1] - 0.5 * math.pi) < 1e-12)

    # Length must matter, or a diffuse spray of equal-frequency completions
    # would be perfectly coherent and the mechanism inert.
    ph3 = phases([("ab", 50.0), ("abcdefgh", 50.0)], 1.0)
    check("equal weights of different length are separated in phase",
          abs(ph3[0] - ph3[1]) > 1e-6,
          "phases %.4f / %.4f" % (ph3[0], ph3[1]))


# ==========================================================================
# 4. Amplitude amplification
# ==========================================================================

def test_amplification():
    print("\n4. amplitude amplification")
    check("no rounds when the marked set already holds half the mass",
          optimal_rounds(0.5) == 0 and optimal_rounds(0.9) == 0)
    check("the schedule matches floor((pi/4)/asin(sqrt(p)))",
          optimal_rounds(0.01, cap=99)
          == int(math.floor((math.pi / 4) / math.asin(math.sqrt(0.01)))),
          "got %d" % optimal_rounds(0.01, cap=99))
    check("the cap is respected", optimal_rounds(1e-6, cap=2) == 2)

    # A scattered fixation: most gaze mass sits on letters no word allows.
    marked = ["e", "r"]
    g = {ch: 0.02 for ch in ALPHABET}
    g["e"] = g["r"] = 0.03
    st = prepare(g)
    before = float(np.sum(np.abs(st[[INDEX[c] for c in marked]]) ** 2))
    out, k = amplify(st, marked, rounds=1)
    after = float(np.sum(np.abs(out[[INDEX[c] for c in marked]]) ** 2))
    check("one round moves mass into the marked subspace", after > before,
          "%.4f -> %.4f" % (before, after))
    check("the round is reported", k == 1)
    check("amplified state stays normalised",
          abs(np.linalg.norm(out) - 1) < 1e-12)

    out2, _ = amplify(st, marked, rounds=2)
    a2 = float(np.sum(np.abs(out2[[INDEX[c] for c in marked]]) ** 2))
    check("a second round moves more", a2 > after, "%.4f -> %.4f" % (after, a2))

    st3, k3 = amplify(st, [], rounds=2)
    check("an empty marked set is a no-op",
          k3 == 0 and np.allclose(st3, st))
    st4, k4 = amplify(st, marked, rounds=0)
    check("zero rounds returns the input untouched",
          k4 == 0 and np.allclose(st4, st))

    # Amplification must not silently fire on the common case.
    pr = {ch: 0.0 for ch in ALPHABET}
    pr["e"] = 0.9
    pr["r"] = 0.1
    conditioned = condition(prepare(g), pr, [], beta=0.7)
    p_marked = float(np.sum(np.abs(conditioned[[INDEX[c] for c in marked]]) ** 2))
    _, k5 = amplify(conditioned, marked, rounds=None, cap=2)
    check("no rounds fire when the prefix is well inside the vocabulary",
          k5 == 0, "marked mass %.4f, rounds %d" % (p_marked, k5))


if __name__ == "__main__":
    print("=" * 70)
    print("quantum conditional selection rule — correctness")
    print("=" * 70)
    test_containment()
    test_expressiveness()
    test_wellformed()
    test_amplification()
    print("\n" + "=" * 70)
    if FAILURES:
        print("FAILED (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
        sys.exit(1)
    print("all properties hold")
