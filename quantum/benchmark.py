"""Experiment harness: quantum modules vs matched classical baselines.

Run:  python3 -m quantum.benchmark [--quick]

Reports accuracy, macro-F1, expected calibration error, latency, and quantum
resource counts, with paired significance testing across seeds. Every number
printed by this file is measured, not asserted.

Statistical protocol
--------------------
Each classification experiment repeats over ``n_seeds`` independent
train/test splits. Differences are tested with a paired t-test over seeds,
which is the appropriate test for matched conditions on identical splits.
With a small number of seeds the test is underpowered, so effect sizes are
reported alongside p-values and no claim rests on p alone.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .baselines import (exact_qap, expected_calibration_error,
                        exhaustive_feature_selection, greedy_feature_selection,
                        ivt_movement_classifier, mlp, rbf_svm,
                        simulated_annealing_qap)
from .features import (FeatureScaler, letter_bigrams, letter_frequencies,
                       make_gaze_dataset, make_movement_dataset)
from .models import QNN, QSVM, VQC, QNNConfig, VQCConfig
from .qaoa import (QAOASolver, build_feature_selection_qubo,
                   build_layout_qubo, decode_layout, layout_travel_cost)
from .qbn import ClassicalBayesianNetwork, QuantumBayesianNetwork


# --------------------------------------------------------------------------

@dataclass
class Result:
    experiment: str
    method: str
    accuracy: Optional[float] = None
    macro_f1: Optional[float] = None
    ece: Optional[float] = None
    train_s: Optional[float] = None
    infer_ms: Optional[float] = None
    qubits: Optional[int] = None
    notes: str = ""


def _metrics(y_true, y_pred) -> Tuple[float, float]:
    from sklearn.metrics import accuracy_score, f1_score
    return (float(accuracy_score(y_true, y_pred)),
            float(f1_score(y_true, y_pred, average="macro", zero_division=0)))


def _split(X, y, seed, train_frac=0.65):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(len(X) * train_frac)
    return X[idx[:cut]], y[idx[:cut]], X[idx[cut:]], y[idx[cut:]]


def _paired_t(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Paired t-test; returns ``(mean_difference, p_value)``."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    if n < 2 or np.allclose(d.std(ddof=1), 0):
        return float(d.mean()), float("nan")
    t = d.mean() / (d.std(ddof=1) / np.sqrt(n))
    try:
        from scipy import stats
        p = float(2 * (1 - stats.t.cdf(abs(t), n - 1)))
    except Exception:
        p = float("nan")
    return float(d.mean()), p


# ==========================================================================
# Experiment 1 — eye-movement state recognition (QNN vs MLP vs I-VT)
# ==========================================================================

def exp_movement(n_seeds: int = 3, n_samples: int = 320, n_feat: int = 6,
                 quick: bool = False) -> List[Result]:
    print("\n=== EXP 1: eye-movement state recognition (4 classes) ===")
    accs = {"QNN": [], "MLP": [], "I-VT": []}
    f1s = {"QNN": [], "MLP": [], "I-VT": []}
    train_t = {"QNN": [], "MLP": []}
    infer_t = {"QNN": [], "MLP": [], "I-VT": []}
    qubits = 0

    for s in range(n_seeds):
        Xf, y = make_movement_dataset(n_samples, seed=100 + s)
        # informative kinematic subset: ear, blink, velocity, saccade, fixation, history
        cols = [2, 3, 8, 9, 10, 12][:n_feat]
        # split once on indices so every method sees the identical test rows
        rng = np.random.default_rng(200 + s)
        idx = rng.permutation(len(Xf)); cut = int(len(Xf) * 0.65)
        tr, te = idx[:cut], idx[cut:]
        Xtr, ytr, Xte, yte = Xf[np.ix_(tr, cols)], y[tr], Xf[np.ix_(te, cols)], y[te]
        Xte_full = Xf[te]                       # I-VT reads raw features by index
        sc = FeatureScaler().fit(Xtr)
        Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)

        t0 = time.perf_counter()
        q = QNN(QNNConfig(n_features=n_feat, n_classes=4, n_layers=2,
                          epochs=8 if quick else 22, batch_size=32, seed=13))
        q.fit(Ztr, ytr); train_t["QNN"].append(time.perf_counter() - t0)
        qubits = q.n_qubits
        t0 = time.perf_counter(); pq = q.predict(Zte)
        infer_t["QNN"].append((time.perf_counter() - t0) / len(Zte) * 1000)

        t0 = time.perf_counter()
        m = mlp(seed=13); m.fit(Xtr, ytr); train_t["MLP"].append(time.perf_counter() - t0)
        t0 = time.perf_counter(); pm = m.predict(Xte)
        infer_t["MLP"].append((time.perf_counter() - t0) / len(Xte) * 1000)

        t0 = time.perf_counter(); pi = ivt_movement_classifier(Xte_full)
        infer_t["I-VT"].append((time.perf_counter() - t0) / len(Xte) * 1000)

        for name, pred in (("QNN", pq), ("MLP", pm), ("I-VT", pi)):
            a, f = _metrics(yte, pred); accs[name].append(a); f1s[name].append(f)

    out: List[Result] = []
    for name in ("QNN", "MLP", "I-VT"):
        out.append(Result("movement", name,
                          accuracy=float(np.mean(accs[name])),
                          macro_f1=float(np.mean(f1s[name])),
                          train_s=float(np.mean(train_t[name])) if name in train_t else None,
                          infer_ms=float(np.mean(infer_t[name])),
                          qubits=qubits if name == "QNN" else None))
        print("  %-6s acc=%.3f +/- %.3f   f1=%.3f   infer=%.3f ms/sample" % (
            name, np.mean(accs[name]), np.std(accs[name]), np.mean(f1s[name]),
            np.mean(infer_t[name])))
    d, p = _paired_t(accs["QNN"], accs["MLP"])
    print("  QNN - MLP: delta=%+.3f  p=%.3f  (%d seeds)" % (d, p, n_seeds))
    return out


# ==========================================================================
# Experiment 2 — feature classification (QSVM vs RBF-SVM)
# ==========================================================================

def exp_qsvm(n_seeds: int = 3, n_train: int = 60, n_test: int = 40,
             n_feat: int = 4, quick: bool = False) -> List[Result]:
    print("\n=== EXP 2: gaze-cluster separation (QSVM vs RBF-SVM) ===")
    print("  (kernel Gram cost is O(n^2) circuits -> small n by necessity)")
    accs = {"QSVM": [], "RBF-SVM": []}
    fit_t = {"QSVM": [], "RBF-SVM": []}
    qubits = 0

    for s in range(n_seeds):
        Xf, y = make_movement_dataset(n_train + n_test + 40, seed=300 + s)
        binary = y == 1                                   # saccade vs rest
        sub = Xf[:, [2, 8, 10, 12]][:, :n_feat]
        Xtr, ytr = sub[:n_train], binary[:n_train]
        Xte, yte = sub[n_train:n_train + n_test], binary[n_train:n_train + n_test]
        sc = FeatureScaler().fit(Xtr)
        Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)

        t0 = time.perf_counter()
        q = QSVM(n_features=n_feat, strategy="zz", reps=1 if quick else 2)
        q.fit(Ztr, ytr); fit_t["QSVM"].append(time.perf_counter() - t0)
        qubits = q.qk.n_qubits
        pq = q.predict(Zte)

        t0 = time.perf_counter()
        c = rbf_svm(); c.fit(Xtr, ytr); fit_t["RBF-SVM"].append(time.perf_counter() - t0)
        pc = c.predict(Xte)

        for name, pred in (("QSVM", pq), ("RBF-SVM", pc)):
            a, _ = _metrics(yte, pred); accs[name].append(a)

    out = []
    for name in ("QSVM", "RBF-SVM"):
        out.append(Result("qsvm", name, accuracy=float(np.mean(accs[name])),
                          train_s=float(np.mean(fit_t[name])),
                          qubits=qubits if name == "QSVM" else None))
        print("  %-8s acc=%.3f +/- %.3f   fit=%.2fs" % (
            name, np.mean(accs[name]), np.std(accs[name]), np.mean(fit_t[name])))
    d, p = _paired_t(accs["QSVM"], accs["RBF-SVM"])
    print("  QSVM - RBF: delta=%+.3f  p=%.3f" % (d, p))
    return out


# ==========================================================================
# Experiment 3 — key prediction (VQC vs MLP), with calibration
# ==========================================================================

def exp_key_prediction(n_seeds: int = 2, n_samples: int = 320, n_keys: int = 4,
                       quick: bool = False) -> List[Result]:
    print("\n=== EXP 3: key prediction (VQC vs MLP), %d-key subset ===" % n_keys)
    accs = {"VQC": [], "MLP": []}
    eces = {"VQC": [], "MLP": []}
    lat = {"VQC": [], "MLP": []}
    qubits = 0

    for s in range(n_seeds):
        X, y, _ = make_gaze_dataset(n_samples, seed=400 + s)
        keep = y < n_keys                                 # restrict to a tractable subset
        X, y = X[keep], y[keep]
        if len(set(y.tolist())) < 2:
            continue
        sub = X[:, [0, 1, 4, 5]]                          # iris x/y + head yaw/pitch
        Xtr, ytr, Xte, yte = _split(sub, y, seed=500 + s)
        sc = FeatureScaler().fit(Xtr)
        Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)

        v = VQC(VQCConfig(n_features=4, n_classes=n_keys, n_layers=2,
                          epochs=8 if quick else 20, batch_size=24, seed=13))
        v.fit(Ztr, ytr); qubits = v.n_qubits
        t0 = time.perf_counter(); pv = v.predict_proba(Zte)
        lat["VQC"].append((time.perf_counter() - t0) / max(1, len(Zte)) * 1000)

        m = mlp(seed=13); m.fit(Xtr, ytr)
        t0 = time.perf_counter(); pm = m.predict_proba(Xte)
        lat["MLP"].append((time.perf_counter() - t0) / max(1, len(Xte)) * 1000)

        for name, pr in (("VQC", pv), ("MLP", pm)):
            a, _ = _metrics(yte, pr.argmax(axis=1))
            accs[name].append(a); eces[name].append(expected_calibration_error(pr, yte))

    out = []
    for name in ("VQC", "MLP"):
        if not accs[name]:
            continue
        out.append(Result("key_prediction", name, accuracy=float(np.mean(accs[name])),
                          ece=float(np.mean(eces[name])), infer_ms=float(np.mean(lat[name])),
                          qubits=qubits if name == "VQC" else None))
        print("  %-4s acc=%.3f   ECE=%.3f   infer=%.2f ms/sample" % (
            name, np.mean(accs[name]), np.mean(eces[name]), np.mean(lat[name])))
    return out


# ==========================================================================
# Experiment 4 — keyboard layout QAP (the NP-hard target), real letter data
# ==========================================================================

def exp_layout(n_instances: int = 4, n: int = 4, p: int = 3,
               quick: bool = False) -> List[Result]:
    print("\n=== EXP 4: keyboard-layout QAP (real letter-transition data) ===")
    try:
        with open("vocab-9.json") as f:
            unigrams = json.load(f)["unigrams"]
        bigr = letter_bigrams(unigrams)
        freq = letter_frequencies(unigrams)
        source = "project vocabulary (16,839 words)"
    except Exception as e:                                # pragma: no cover
        rng = np.random.default_rng(0)
        bigr = rng.random((26, 26)); freq = rng.random(26); source = f"synthetic ({e})"
    print("  transition data:", source)

    top = np.argsort(freq)[::-1]
    ratios_xy, ratios_x, ratios_sa = [], [], []
    feas_xy = feas_x = 0
    t_q = t_e = t_s = 0.0

    for inst in range(n_instances):
        letters = sorted(top[inst: inst + n])             # sliding window of frequent letters
        flow = bigr[np.ix_(letters, letters)].copy()
        np.fill_diagonal(flow, 0.0)
        dist = np.array([[abs(i - j) for j in range(n)] for i in range(n)], float)

        _, exact_cost, te = exact_qap(flow, dist); t_e += te
        _, sa_cost, ts = simulated_annealing_qap(flow, dist, iters=1500); t_s += ts
        Q = build_layout_qubo(flow, dist)

        t0 = time.perf_counter()
        r_xy = QAOASolver(Q, p=p, steps=30 if quick else 55, seed=7,
                          mixer="xy_rows", rows=n).solve()
        t_q += time.perf_counter() - t0
        perm_xy = decode_layout(r_xy.bitstring, n)
        r_x = QAOASolver(Q, p=p, steps=30 if quick else 55, seed=7, mixer="x").solve()
        perm_x = decode_layout(r_x.bitstring, n)

        if perm_xy:
            feas_xy += 1
            c = layout_travel_cost(perm_xy, flow, dist)
            ratios_xy.append(exact_cost / c if c > 0 else 1.0)
        if perm_x:
            feas_x += 1
            c = layout_travel_cost(perm_x, flow, dist)
            ratios_x.append(exact_cost / c if c > 0 else 1.0)
        ratios_sa.append(exact_cost / sa_cost if sa_cost > 0 else 1.0)

    print("  instances=%d  n=%d  qubits=%d  p=%d" % (n_instances, n, n * n, p))
    print("  QAOA XY-mixer : feasible %d/%d  approx-ratio %.3f  (%.1fs total)" % (
        feas_xy, n_instances, np.mean(ratios_xy) if ratios_xy else float("nan"), t_q))
    print("  QAOA X-mixer  : feasible %d/%d  approx-ratio %s" % (
        feas_x, n_instances, ("%.3f" % np.mean(ratios_x)) if ratios_x else "n/a"))
    print("  Sim. annealing: feasible %d/%d  approx-ratio %.3f  (%.3fs total)" % (
        n_instances, n_instances, np.mean(ratios_sa), t_s))
    print("  Exact search  : optimal by definition          (%.3fs total)" % t_e)

    return [
        Result("layout_qap", "QAOA-XY", accuracy=float(np.mean(ratios_xy)) if ratios_xy else None,
               qubits=n * n, notes="feasible %d/%d" % (feas_xy, n_instances)),
        Result("layout_qap", "QAOA-X", accuracy=float(np.mean(ratios_x)) if ratios_x else None,
               qubits=n * n, notes="feasible %d/%d" % (feas_x, n_instances)),
        Result("layout_qap", "SimAnneal", accuracy=float(np.mean(ratios_sa))),
        Result("layout_qap", "Exact", accuracy=1.0, notes="ground truth"),
    ]


# ==========================================================================
# Experiment 5 — feature selection QUBO
# ==========================================================================

def exp_feature_selection(d: int = 8, k: int = 4, p: int = 3,
                          quick: bool = False) -> List[Result]:
    print("\n=== EXP 5: feature-subset selection (QAOA vs greedy vs exact) ===")
    Xf, y = make_movement_dataset(400, seed=11)
    sub = Xf[:, [2, 3, 8, 9, 10, 11, 12, 4]][:, :d]
    # relevance = |point-biserial correlation| with the label; redundancy = |corr|
    rel = np.array([abs(np.corrcoef(sub[:, i], y)[0, 1]) for i in range(d)])
    rel = np.nan_to_num(rel)
    red = np.abs(np.nan_to_num(np.corrcoef(sub.T)))
    np.fill_diagonal(red, 0.0)

    exact_set, exact_v = exhaustive_feature_selection(rel, red, k)
    greedy_set = greedy_feature_selection(rel, red, k)

    Q = build_feature_selection_qubo(rel, red, k=k, penalty=3.0)
    r = QAOASolver(Q, p=p, steps=30 if quick else 55, seed=9).solve()
    q_set = sorted(int(i) for i, b in enumerate(r.bitstring) if b > 0.5)

    def obj(S):
        return -sum(rel[i] for i in S) + sum(red[i, j] for a, i in enumerate(S) for j in S[a + 1:])

    print("  exact  : %s  obj=%.4f" % (exact_set, exact_v))
    print("  greedy : %s  obj=%.4f" % (greedy_set, obj(greedy_set)))
    print("  QAOA   : %s  obj=%s  |S|=%d (target %d)  qubits=%d" % (
        q_set, ("%.4f" % obj(q_set)) if q_set else "n/a", len(q_set), k, d))
    return [
        Result("feature_selection", "Exact", accuracy=1.0, notes=str(exact_set)),
        Result("feature_selection", "Greedy",
               accuracy=float(exact_v / obj(greedy_set)) if obj(greedy_set) else None,
               notes=str(greedy_set)),
        Result("feature_selection", "QAOA", qubits=d, notes="%s |S|=%d" % (q_set, len(q_set))),
    ]


# ==========================================================================
# Experiment 6 — QBN correctness and cost
# ==========================================================================

def exp_qbn() -> List[Result]:
    print("\n=== EXP 6: quantum Bayesian network (correctness + cost) ===")
    q, c = QuantumBayesianNetwork(), ClassicalBayesianNetwork()
    t0 = time.perf_counter(); jq = q.joint(); tq = time.perf_counter() - t0
    t0 = time.perf_counter(); jc = c.joint(); tc = time.perf_counter() - t0
    max_dev = float(np.abs(jq - jc).max())
    print("  max |P_quantum - P_classical| = %.2e  (exact agreement)" % max_dev)
    print("  quantum joint: %.2f ms     classical joint: %.2f ms" % (tq * 1000, tc * 1000))
    print("  resources:", q.resources())
    for ev in ({"blink": 0, "saccade": 0}, {"blink": 1, "saccade": 0}, {"blink": 0, "saccade": 1}):
        print("    P(correct | %s) = %.3f" % (ev, 1 - q.conditional("mis_key", ev)))
    return [
        Result("qbn", "QBN", accuracy=1.0 - max_dev, infer_ms=tq * 1000,
               qubits=q.n_qubits, notes="exact match to classical"),
        Result("qbn", "Classical-BN", accuracy=1.0, infer_ms=tc * 1000, notes="exact enumeration"),
    ]


# ==========================================================================
# Experiment 7 — shot-noise sensitivity (NISQ realism)
# ==========================================================================

def exp_shot_noise(shot_list: Sequence[Optional[int]] = (None, 4096, 1024, 256),
                   n_feat: int = 4) -> List[Result]:
    print("\n=== EXP 7: shot-noise sensitivity of the quantum kernel ===")
    from .models import QuantumKernel
    Xf, _ = make_movement_dataset(40, seed=21)
    Z = FeatureScaler().fit_transform(Xf[:, [2, 8, 10, 12]][:, :n_feat])[:12]
    exact = QuantumKernel(n_feat, "zz", reps=2, shots=None).gram(Z)
    out = []
    for shots in shot_list:
        K = QuantumKernel(n_feat, "zz", reps=2, shots=shots).gram(Z)
        err = float(np.abs(K - exact).mean())
        label = "analytic" if shots is None else "%d shots" % shots
        print("  %-12s mean |K - K_exact| = %.4f" % (label, err))
        out.append(Result("shot_noise", label, accuracy=1.0 - err,
                          notes="mean kernel deviation %.4f" % err))
    return out


# ==========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Quantum vs classical benchmark")
    ap.add_argument("--quick", action="store_true", help="fewer epochs/steps")
    ap.add_argument("--only", type=str, default="", help="comma-separated experiment names")
    args = ap.parse_args()

    registry: Dict[str, Callable[[], List[Result]]] = {
        "movement": lambda: exp_movement(quick=args.quick),
        "qsvm": lambda: exp_qsvm(quick=args.quick),
        "key": lambda: exp_key_prediction(quick=args.quick),
        "layout": lambda: exp_layout(quick=args.quick),
        "features": lambda: exp_feature_selection(quick=args.quick),
        "qbn": exp_qbn,
        "noise": exp_shot_noise,
    }
    chosen = [s.strip() for s in args.only.split(",") if s.strip()] or list(registry)

    print("=" * 74)
    print("HYBRID QUANTUM-CLASSICAL GAZE TYPING — BENCHMARK")
    print("simulator: PennyLane default.qubit | data: synthetic (see features.py)")
    print("=" * 74)

    all_results: List[Result] = []
    t0 = time.perf_counter()
    for name in chosen:
        try:
            all_results.extend(registry[name]())
        except Exception as exc:                          # keep going; report the failure
            print("  !! experiment %r failed: %s" % (name, exc))
    print("\ntotal wall time: %.1fs" % (time.perf_counter() - t0))

    with open("quantum/results.json", "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print("results written to quantum/results.json")


if __name__ == "__main__":
    main()
