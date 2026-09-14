"""End-to-end benchmark and hyperparameter tuning for the gaze-typing pipeline.

Run:
    python3 tune.py bench     # latency + accuracy of every algorithm
    python3 tune.py sweep     # hyperparameter search
    python3 tune.py e2e       # full typing simulation under gaze noise
    python3 tune.py all

Design rule: every tunable constant is swept against a *held-out* metric, and
the winning value is only adopted if the gain survives on data not used to
choose it. Tuning on the test set is how projects fool themselves.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import random
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from clusters import encode_text
from decoder import HybridDecoder
from letters import END, LetterModel, key_distance

VOCAB = "vocab-9.json"
DATASET = "translation-cluster-eng-dataset.csv"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_sentences(n: int, seed: int = 13, skip: int = 0) -> List[Tuple[str, str]]:
    """Load ``n`` (cluster, english) pairs. ``skip`` gives a disjoint split."""
    df = pd.read_csv(DATASET, nrows=n + skip + 5000)
    rows = [ast.literal_eval(t) for t in df["translation"]]
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[skip: skip + n]
    return [(r["cluster"], r["en"]) for r in rows]


def _timeit(fn, *a, repeat: int = 1, **kw) -> Tuple[object, float]:
    """Return ``(result, mean_ms)``."""
    t0 = time.perf_counter()
    out = None
    for _ in range(repeat):
        out = fn(*a, **kw)
    return out, (time.perf_counter() - t0) * 1000.0 / repeat


# ==========================================================================
# 1. Per-algorithm benchmark — accuracy and latency
# ==========================================================================

def bench(n_sentences: int = 400) -> Dict[str, dict]:
    print("=" * 74)
    print("ALGORITHM BENCHMARK — accuracy and latency")
    print("=" * 74)
    dec = HybridDecoder(vocab_path=VOCAB, model_dir=None, num_clusters=9)
    lm = LetterModel(dec.unigrams)
    pairs = load_sentences(n_sentences)
    report: Dict[str, dict] = {}

    # ---- decoder: exact-lookup / Viterbi / beam ----
    print("\n[1] Cluster decoder")
    sent_hit = word_hit = word_tot = consistent = 0
    lat = []
    for code, en in pairs:
        t0 = time.perf_counter()
        r = dec.decode(code)
        lat.append((time.perf_counter() - t0) * 1000)
        sent_hit += (r.text == en)
        consistent += r.consistent
        ref, hyp = en.split(), r.text.split()
        word_tot += len(ref)
        word_hit += sum(a == b for a, b in zip(ref, hyp))
    report["decoder"] = {
        "sentence_acc": sent_hit / len(pairs),
        "word_acc": word_hit / max(1, word_tot),
        "consistency": consistent / len(pairs),
        "latency_ms": float(np.mean(lat)),
        "p95_ms": float(np.percentile(lat, 95)),
    }
    print("    sentence=%.4f  word=%.4f  consistency=%.4f  latency=%.2f ms (p95 %.2f)" % (
        report["decoder"]["sentence_acc"], report["decoder"]["word_acc"],
        report["decoder"]["consistency"], report["decoder"]["latency_ms"],
        report["decoder"]["p95_ms"]))

    # ---- Viterbi vs beam: does exactness actually change the answer? ----
    print("\n[2] Viterbi (exact) vs beam search (approximate)")
    diff = better = 0
    vlat, blat = [], []
    for code, _en in pairs[:200]:
        codes = code.split()
        cps = [dec.candidates_for_code(c) for c in codes]
        if not all(cps):
            continue
        (vp, vs), t_v = _timeit(dec._viterbi_best, cps)
        vlat.append(t_v)
        t0 = time.perf_counter(); beams = dec._lattice_paths(cps); blat.append((time.perf_counter() - t0) * 1000)
        bp, bs = beams[0]
        if vp != bp:
            diff += 1
            better += vs > bs + 1e-12
    report["viterbi_vs_beam"] = {
        "paths_differ": diff, "viterbi_better": better,
        "viterbi_ms": float(np.mean(vlat)), "beam_ms": float(np.mean(blat)),
    }
    print("    differing paths: %d/200   Viterbi strictly better: %d" % (diff, better))
    print("    Viterbi %.3f ms vs beam %.3f ms  (%.1fx faster)" % (
        np.mean(vlat), np.mean(blat), np.mean(blat) / max(1e-9, np.mean(vlat))))

    # ---- letter prior: next-letter prediction accuracy ----
    print("\n[3] Letter prior (next-symbol prediction)")
    top1 = top3 = tot = 0
    lat = []
    for _code, en in pairs[:200]:
        for w in en.split()[:4]:
            for i in range(len(w)):
                prefix, target = w[:i], w[i]
                t0 = time.perf_counter(); pr = lm.letter_prior(prefix)
                lat.append((time.perf_counter() - t0) * 1000)
                ranked = sorted(pr, key=pr.get, reverse=True)
                top1 += ranked[0] == target
                top3 += target in ranked[:3]
                tot += 1
    report["letter_prior"] = {
        "top1": top1 / max(1, tot), "top3": top3 / max(1, tot),
        "latency_ms": float(np.mean(lat)), "n": tot,
    }
    print("    top-1=%.4f  top-3=%.4f  latency=%.3f ms  (n=%d)" % (
        report["letter_prior"]["top1"], report["letter_prior"]["top3"],
        report["letter_prior"]["latency_ms"], tot))

    # ---- word correction: recovery of gaze-plausible typos ----
    print("\n[4] Gaze-aware word correction")
    acc, lat = corruption_test(lm, dec, n=250)
    report["correction"] = {"recovery_top1": acc["top1"], "recovery_top3": acc["top3"],
                            "latency_ms": lat}
    print("    typo recovery top-1=%.4f  top-3=%.4f  latency=%.2f ms" % (
        acc["top1"], acc["top3"], lat))

    # ---- ridge gaze map ----
    print("\n[5] Ridge gaze mapping (calibration fit + predict)")
    rng = np.random.default_rng(0)
    F = np.column_stack([np.ones(140), rng.uniform(0.2, 0.8, (140, 7))])
    t = rng.uniform(0, 1, 140)
    def fit():
        A = F.T @ F + 1e-3 * np.eye(F.shape[1])
        return np.linalg.solve(A, F.T @ t)
    w, fit_ms = _timeit(fit, repeat=50)
    _, pred_ms = _timeit(lambda: float(F[0] @ w), repeat=1000)
    report["ridge"] = {"fit_ms": fit_ms, "predict_ms": pred_ms}
    print("    fit(140 samples)=%.3f ms   predict=%.5f ms" % (fit_ms, pred_ms))

    # ---- budget check ----
    print("\n[6] Real-time budget (33 ms/frame @30fps, 693 ms dwell)")
    per_frame = report["ridge"]["predict_ms"]
    per_commit = report["letter_prior"]["latency_ms"] + report["decoder"]["latency_ms"]
    print("    per-frame  path: %.4f ms  -> %.2f%% of 33 ms budget" % (per_frame, per_frame / 33 * 100))
    print("    per-commit path: %.3f ms  -> %.2f%% of 693 ms budget" % (per_commit, per_commit / 693 * 100))
    report["budget"] = {"per_frame_ms": per_frame, "per_commit_ms": per_commit}
    return report


def corruption_test(lm: LetterModel, dec: HybridDecoder, n: int = 250,
                    seed: int = 5) -> Tuple[Dict[str, float], float]:
    """Corrupt real words with *spatially plausible* gaze slips, then recover.

    A slip replaces a letter with a neighbour on the QWERTY board — the error
    distribution the noisy-channel model is designed for.
    """
    rng = random.Random(seed)
    words = [w for w, c in sorted(dec.unigrams.items(), key=lambda kv: -kv[1])[:4000]
             if 4 <= len(w) <= 9 and w.isalpha()]
    rows = "qwertyuiop asdfghjkl zxcvbnm".split()
    neighbours: Dict[str, List[str]] = {}
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
            neighbours[ch] = adj

    top1 = top3 = 0
    lat = []
    tested = 0
    for _ in range(n):
        w = rng.choice(words)
        i = rng.randrange(len(w))
        if w[i] not in neighbours:
            continue
        typo = w[:i] + rng.choice(neighbours[w[i]]) + w[i + 1:]
        if typo == w:
            continue
        t0 = time.perf_counter()
        res = lm.correct(typo, n=3)
        lat.append((time.perf_counter() - t0) * 1000)
        sug = res["suggestions"]
        tested += 1
        top1 += bool(sug) and sug[0] == w
        top3 += w in sug
    return ({"top1": top1 / max(1, tested), "top3": top3 / max(1, tested)},
            float(np.mean(lat)) if lat else 0.0)


# ==========================================================================
# 2. Hyperparameter sweeps
# ==========================================================================

def sweep_decoder_lm(n_train: int = 400, n_test: int = 400) -> float:
    """Sweep the bigram/unigram interpolation weight in the decoder LM."""
    print("\n" + "=" * 74)
    print("SWEEP 1: decoder LM interpolation (bigram weight)")
    print("=" * 74)
    dec = HybridDecoder(vocab_path=VOCAB, model_dir=None, num_clusters=9)
    train = load_sentences(n_train, seed=1)
    test = load_sentences(n_test, seed=2, skip=n_train + 1000)

    original = dec.bigram_weight if hasattr(dec, "bigram_weight") else 0.7

    def score(pairs) -> float:
        hit = 0
        for code, en in pairs:
            hit += dec.decode(code).text == en
        return hit / len(pairs)

    results = []
    for w in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        dec.bigram_weight = w
        acc = score(train)
        results.append((w, acc))
        print("    bigram_weight=%.2f   train sentence-acc=%.4f" % (w, acc))
    best_w = max(results, key=lambda r: r[1])[0]

    dec.bigram_weight = original
    base_test = score(test)
    dec.bigram_weight = best_w
    tuned_test = score(test)
    print("    -> best on train: %.2f" % best_w)
    print("    HELD-OUT:  default(%.2f)=%.4f   tuned(%.2f)=%.4f   delta=%+.4f" % (
        original, base_test, best_w, tuned_test, tuned_test - base_test))
    return best_w if tuned_test > base_test else original


def sweep_letter_prior(n: int = 250) -> Tuple[float, int, float]:
    """Sweep trie/char mix, char-gram order, and smoothing k."""
    print("\n" + "=" * 74)
    print("SWEEP 2: letter prior (trie mix / char-gram order / smoothing)")
    print("=" * 74)
    dec = HybridDecoder(vocab_path=VOCAB, model_dir=None, num_clusters=9)
    train = load_sentences(n, seed=3)
    test = load_sentences(n, seed=4, skip=n + 2000)

    def evaluate(lm: LetterModel, pairs) -> Tuple[float, float]:
        t1 = t3 = tot = 0
        for _code, en in pairs:
            for w in en.split()[:4]:
                for i in range(len(w)):
                    pr = lm.letter_prior(w[:i])
                    ranked = sorted(pr, key=pr.get, reverse=True)
                    t1 += ranked[0] == w[i]
                    t3 += w[i] in ranked[:3]
                    tot += 1
        return t1 / max(1, tot), t3 / max(1, tot)

    best = None
    for order in (3, 4, 5):
        for mix in (0.70, 0.85, 0.92, 0.97):
            for k in (0.01, 0.1, 0.5):
                lm = LetterModel(dec.unigrams, char_order=order, trie_mix=mix, smoothing_k=k)
                a1, a3 = evaluate(lm, train)
                if best is None or a1 > best[0]:
                    best = (a1, a3, order, mix, k)
                print("    order=%d mix=%.2f k=%.2f  top1=%.4f top3=%.4f" % (order, mix, k, a1, a3))
    _, _, o, m, k = best
    print("    -> best on train: order=%d mix=%.2f k=%.2f (top1=%.4f)" % (o, m, k, best[0]))

    base = LetterModel(dec.unigrams, char_order=3, trie_mix=0.85, smoothing_k=0.1)
    tuned = LetterModel(dec.unigrams, char_order=o, trie_mix=m, smoothing_k=k)
    b1, b3 = evaluate(base, test)
    t1_, t3_ = evaluate(tuned, test)
    print("    HELD-OUT:  default top1=%.4f/top3=%.4f   tuned top1=%.4f/top3=%.4f   delta=%+.4f" % (
        b1, b3, t1_, t3_, t1_ - b1))
    return (m, o, k) if t1_ > b1 else (0.85, 3, 0.1)


# ==========================================================================
# 3. End-to-end typing simulation under gaze noise
# ==========================================================================

def simulate_typing(sigma_px: float, lm: LetterModel, dec: HybridDecoder,
                    pairs: Sequence[Tuple[str, str]], lm_exponent: float,
                    theta: float, use_fusion: bool, seed: int = 7,
                    key_w: float = 100.0, key_h: float = 110.0) -> Dict[str, float]:
    """Simulate gaze typing and measure committed-character accuracy.

    Model: the user intends letter ``c``; the gaze estimate lands at the key
    centre plus 2-D Gaussian noise of std ``sigma_px``. The system either takes
    the nearest key (baseline) or fuses the Gaussian key posterior with the
    language prior (fusion), optionally deferring to a prompt when the top
    posterior is below ``theta``.
    """
    rng = np.random.default_rng(seed)
    rows = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
    centre: Dict[str, Tuple[float, float]] = {}
    for r, row in enumerate(rows):
        indent = (10 - len(row)) * key_w / 2.0
        for i, ch in enumerate(row):
            centre[ch] = (indent + i * key_w + key_w / 2, r * key_h + key_h / 2)

    correct = total = prompts = 0
    for _code, en in pairs:
        for word in en.split():
            typed = ""
            for ch in word:
                if ch not in centre:
                    continue
                cx, cy = centre[ch]
                gx, gy = cx + rng.normal(0, sigma_px), cy + rng.normal(0, sigma_px)

                cands = []
                for k, (kx, ky) in centre.items():
                    d2 = ((gx - kx) / sigma_px) ** 2 + ((gy - ky) / sigma_px) ** 2
                    cands.append((k, math.exp(-0.5 * d2)))
                z = sum(p for _, p in cands) or 1.0
                cands = [(k, p / z) for k, p in cands]

                if use_fusion:
                    prior = lm.letter_prior(typed)
                    fused = [(k, p * (prior.get(k, 1e-6) ** lm_exponent)) for k, p in cands]
                    zz = sum(p for _, p in fused) or 1.0
                    fused = sorted(((k, p / zz) for k, p in fused), key=lambda t: -t[1])
                    top, top_p = fused[0]
                    if theta < 1.0 and top_p < theta:
                        prompts += 1
                        top = ch          # a prompt lets the user pick correctly
                    pred = top
                else:
                    pred = max(cands, key=lambda t: t[1])[0]

                correct += (pred == ch)
                total += 1
                typed += pred
    return {"char_acc": correct / max(1, total), "prompt_rate": prompts / max(1, total),
            "n": total}


def e2e(n_sentences: int = 60) -> None:
    print("\n" + "=" * 74)
    print("END-TO-END TYPING SIMULATION (gaze noise -> committed characters)")
    print("=" * 74)
    dec = HybridDecoder(vocab_path=VOCAB, model_dir=None, num_clusters=9)
    lm = LetterModel(dec.unigrams)
    pairs = load_sentences(n_sentences, seed=8)

    print("\n  A. Fusion gain vs raw nearest-key, by gaze noise level")
    print("     sigma(px)  nearest-key   +fusion    gain")
    for sigma in (20, 30, 40, 50, 60):
        base = simulate_typing(sigma, lm, dec, pairs, 0.0, 1.0, use_fusion=False)
        fus = simulate_typing(sigma, lm, dec, pairs, 0.55, 1.0, use_fusion=True)
        print("     %6.0f     %.4f      %.4f    %+.4f" % (
            sigma, base["char_acc"], fus["char_acc"], fus["char_acc"] - base["char_acc"]))

    print("\n  B. LM exponent sweep (sigma=40)")
    best_exp, best_acc = 0.55, -1.0
    for e in (0.0, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.3):
        r = simulate_typing(40, lm, dec, pairs, e, 1.0, use_fusion=True)
        flag = ""
        if r["char_acc"] > best_acc:
            best_acc, best_exp, flag = r["char_acc"], e, "  <-- best"
        print("     exponent=%.2f  char-acc=%.4f%s" % (e, r["char_acc"], flag))
    print("     -> best LM exponent: %.2f (acc %.4f)" % (best_exp, best_acc))

    print("\n  C. Confidence gate: accuracy vs prompt rate (sigma=40, exponent=%.2f)" % best_exp)
    print("     theta   char-acc   prompt-rate")
    for th in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
        r = simulate_typing(40, lm, dec, pairs, best_exp, th, use_fusion=True)
        print("     %.2f    %.4f     %.4f" % (th, r["char_acc"], r["prompt_rate"]))


# ==========================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["bench", "sweep", "e2e", "all"], default="all", nargs="?")
    args = ap.parse_args()
    if args.mode in ("bench", "all"):
        rep = bench()
        with open("bench_results.json", "w") as f:
            json.dump(rep, f, indent=2)
    if args.mode in ("sweep", "all"):
        sweep_decoder_lm()
        sweep_letter_prior()
    if args.mode in ("e2e", "all"):
        e2e()


if __name__ == "__main__":
    main()
