"""Probability calibration trials (scikit-learn) for the commit gate.

The keyboard commits a key when the fused posterior clears ``THETA = 0.80`` and
otherwise asks. That threshold is a probability claim, so it is worth asking
whether the number means what it says. ``sklearn.calibration.calibration_curve``
answers it directly, and the reliability gap turns out to be large.

Three experiments, which are different claims and must not be conflated:

**B1 — univariate calibration of the commit confidence.** Platt scaling and
isotonic regression applied to the top posterior. These improve how honest the
confidence is, measured by expected calibration error and Brier score. They
*cannot* improve selection accuracy, and we verify rather than assume it: both
are monotone in the score, so thresholding a calibrated score at $\\theta'$
selects exactly the keystrokes that thresholding the raw score at
$g^{-1}(\\theta')$ selects. The accuracy/prompt-rate frontier is invariant by
construction, and any apparent gain would be a bug in the harness.

**B2 — a multivariate confidence model.** The way to move the frontier is to
predict correctness from evidence the top posterior alone does not carry: the
margin to the runner-up, the entropy of the posterior, the gaze uncertainty,
how much prior mass the language model put on the winner. That model is not
monotone in the top posterior, so it can reorder keystrokes, and reordering is
what changes accuracy at matched prompt rate.

**B3 — calibration of the gaze uncertainty itself.** The key posterior is an
integral of a Gaussian whose width the system supplies. If that width is wrong,
every downstream probability is wrong. We check the predicted standard deviation
against measured residuals on the recorded sessions and measure the effect of
correcting it.

Run: ``python3 eval/calibration_trials.py [b1|b2|b3|all]``
"""

from __future__ import annotations

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

from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier                    # noqa: E402
from sklearn.isotonic import IsotonicRegression                            # noqa: E402
from sklearn.linear_model import LogisticRegression                        # noqa: E402
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score       # noqa: E402
from sklearn.pipeline import make_pipeline                                  # noqa: E402
from sklearn.preprocessing import StandardScaler                            # noqa: E402

import eval.selection_rule_bench as B                                       # noqa: E402
from decoder import HybridDecoder                                           # noqa: E402
from letters import LetterModel                                             # noqa: E402
from quantum.conditional_selection import loglinear                         # noqa: E402
from tune import VOCAB, load_sentences                                      # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "calibration_trials_results.json"


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


N_BINS = 10
MATCH_RATES = (0.02, 0.05, 0.10, 0.20, 0.30)


# ==========================================================================
# Metrics
# ==========================================================================

def ece(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS,
        strategy: str = "uniform") -> float:
    """Expected calibration error, weighted by bin population.

    ``calibration_curve`` supplies the reliability diagram; ECE is the
    population-weighted mean gap between the two curves it returns.
    """
    pt, pp = calibration_curve(y, p, n_bins=n_bins, strategy=strategy)
    edges = (np.quantile(p, np.linspace(0, 1, n_bins + 1)) if strategy == "quantile"
             else np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins).astype(float)
    keep = counts > 0
    w = counts[keep][:len(pt)] / max(1, len(p))
    return float(np.sum(w * np.abs(pt - pp[:len(w)])))


def frontier(y: np.ndarray, p: np.ndarray) -> Dict[float, float]:
    """Commit accuracy at each matched prompt rate, by quantile threshold."""
    out = {}
    order = np.argsort(-p)
    ys = y[order]
    for r in MATCH_RATES:
        k = int(round((1.0 - r) * len(p)))
        out[r] = float(ys[:k].mean()) if k > 0 else 1.0
    return out


def describe(name: str, y, p) -> dict:
    return {"name": name,
            "ece_uniform": ece(y, p, strategy="uniform"),
            "ece_quantile": ece(y, p, strategy="quantile"),
            "brier": float(brier_score_loss(y, p)),
            "logloss": float(log_loss(y, np.clip(p, 1e-9, 1 - 1e-9))),
            "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
            "frontier": frontier(y, p)}


# ==========================================================================
# Data: keystroke records with the features a confidence model could use
# ==========================================================================

def keystroke_records(lm: LetterModel, pairs, sigma: float, seed: int = 7
                      ) -> Dict[str, np.ndarray]:
    """Type every sentence, recording the commit evidence per keystroke.

    Features are exactly what the live system has in hand at commit time; none
    of them peeks at the answer.
    """
    rng = np.random.default_rng(seed)
    rows, ok = [], []
    for _code, en in pairs:
        for word in en.split():
            typed = ""
            for ch in word:
                if ch not in B.CENTRES:
                    continue
                cx, cy = B.CENTRES[ch]
                gx, gy = cx + rng.normal(0, sigma), cy + rng.normal(0, sigma)
                pg = B.gaze_posterior(gx, gy, sigma)
                if not pg:
                    continue
                prior = lm.letter_prior(typed)
                post = loglinear(pg, prior, beta=B.LM_BETA, floor=B.LM_FLOOR)
                order = sorted(post.items(), key=lambda t: -t[1])
                top, tp = order[0]
                second = order[1][1] if len(order) > 1 else 0.0
                vals = np.array([v for _, v in order])
                ent = float(-np.sum(vals * np.log(vals + 1e-12)))
                gvals = np.array(sorted(pg.values(), reverse=True))
                gent = float(-np.sum(gvals * np.log(gvals + 1e-12)))
                rows.append([
                    tp,                                   # top posterior
                    tp - second,                          # margin to runner-up
                    ent,                                  # posterior entropy
                    gvals[0],                             # top gaze mass
                    gent,                                 # gaze entropy
                    float(prior.get(top, 0.0)),           # prior on the winner
                    float(len(pg)),                       # candidate keys in play
                    float(len(typed)),                    # position in the word
                    sigma,                                # gaze uncertainty
                ])
                ok.append(top == ch)
                typed += ch
    X = np.asarray(rows, dtype=float)
    return {"X": X, "y": np.asarray(ok, dtype=int), "p": X[:, 0]}


FEATURES = ["top_p", "margin", "entropy", "top_gaze", "gaze_entropy",
            "prior_on_winner", "n_candidates", "position", "sigma"]


# ==========================================================================
# B1 — univariate calibration
# ==========================================================================

def part_b1(tr, te) -> List[dict]:
    y_tr, p_tr = tr["y"], tr["p"]
    y_te, p_te = te["y"], te["p"]
    rows = [describe("raw fused posterior (deployed)", y_te, p_te)]

    platt = LogisticRegression(C=1e6).fit(p_tr.reshape(-1, 1), y_tr)
    rows.append(describe("Platt / sigmoid scaling",
                         y_te, platt.predict_proba(p_te.reshape(-1, 1))[:, 1]))

    iso = IsotonicRegression(out_of_bounds="clip").fit(p_tr, y_tr)
    rows.append(describe("isotonic regression", y_te, iso.predict(p_te)))

    # temperature scaling on the logit
    lg_tr = np.log(np.clip(p_tr, 1e-9, 1 - 1e-9) / np.clip(1 - p_tr, 1e-9, 1))
    lg_te = np.log(np.clip(p_te, 1e-9, 1 - 1e-9) / np.clip(1 - p_te, 1e-9, 1))
    best_T, best_ll = 1.0, 1e9
    for T in np.linspace(0.2, 5.0, 97):
        q = 1.0 / (1.0 + np.exp(-lg_tr / T))
        ll = log_loss(y_tr, np.clip(q, 1e-9, 1 - 1e-9))
        if ll < best_ll:
            best_ll, best_T = ll, T
    rows.append(describe("temperature scaling (T=%.2f)" % best_T,
                         y_te, 1.0 / (1.0 + np.exp(-lg_te / best_T))))
    return rows


# ==========================================================================
# B2 — multivariate confidence model
# ==========================================================================

def part_b2(tr, te) -> List[dict]:
    Xtr, ytr = tr["X"], tr["y"]
    Xte, yte = te["X"], te["y"]
    rows = [describe("raw fused posterior (deployed)", yte, te["p"])]

    def add(name, clf, cols=None):
        c = slice(None) if cols is None else cols
        try:
            clf.fit(Xtr[:, c], ytr)
            rows.append(describe(name, yte, clf.predict_proba(Xte[:, c])[:, 1]))
        except Exception as exc:
            rows.append({"name": name, "failed": "%s: %s" % (type(exc).__name__, exc)})

    add("logistic on all evidence",
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)))
    add("logistic + CalibratedClassifierCV(sigmoid)",
        CalibratedClassifierCV(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            method="sigmoid", cv=5))
    add("logistic + CalibratedClassifierCV(isotonic)",
        CalibratedClassifierCV(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            method="isotonic", cv=5))
    add("gradient boosting on all evidence",
        GradientBoostingClassifier(random_state=0))
    add("gradient boosting + CalibratedClassifierCV(isotonic)",
        CalibratedClassifierCV(GradientBoostingClassifier(random_state=0),
                               method="isotonic", cv=5))
    add("logistic on top_p alone (control)",
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        cols=[0])
    add("logistic on top_p + margin",
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        cols=[0, 1])
    return rows


# ==========================================================================
# B3 — calibration of the gaze uncertainty
# ==========================================================================

def part_b3() -> dict:
    """Is the predicted gaze uncertainty calibrated against actual residuals?

    Ridge supplies a per-sample predictive variance through the leverage,
    $\\sigma^2 (1 + x^\\top A^{-1} x)$. If that number is right, the standardised
    residual has unit variance and the fraction of residuals inside $k\\sigma$
    matches the normal quantiles. We measure both on the recorded sessions.
    """
    from eval.algorithm_trials import sessions
    from eval.feature_ablation import columns
    from eval.gaze_bench import fit_ridge

    S = sessions()
    pred_sd, resid, ppd = [], [], []
    for s in S:
        X = columns(s, "canthus")
        for g in np.unique(s.groups):
            te = s.groups == g
            Xtr, Ttr, Xte, Tte = X[~te], s.T[~te], X[te], s.T[te]
            A = Xtr.T @ Xtr
            lam = 1e-4 * (np.trace(A) / A.shape[0])
            Ainv = np.linalg.inv(A + lam * np.eye(A.shape[0]))
            W = Ainv @ (Xtr.T @ Ttr)
            R = Ttr - Xtr @ W
            s2 = float((R ** 2).sum() / max(1, len(R) - X.shape[1])) / 2.0
            lev = np.einsum("ij,jk,ik->i", Xte, Ainv, Xte)
            sd = np.sqrt(np.maximum(s2 * (1.0 + lev), 1e-12))
            r = np.linalg.norm(Xte @ W - Tte, axis=1) / math.sqrt(2.0)
            pred_sd.append(sd)
            resid.append(r)
            ppd.append(np.full(len(r), s.px_per_degree))

    sd = np.concatenate(pred_sd)
    r = np.concatenate(resid)
    ppd = np.concatenate(ppd)
    z = r / sd

    out = {"n": int(len(r)),
           "mean_predicted_sd_px": float(sd.mean()),
           "mean_actual_residual_px": float(r.mean()),
           "ratio_actual_over_predicted": float(r.mean() / sd.mean()),
           "std_of_standardised_residual": float(z.std()),
           "mean_predicted_sd_deg": float((sd / ppd).mean()),
           "mean_actual_residual_deg": float((r / ppd).mean())}

    # coverage: what fraction of residuals fall inside k predicted sigma
    out["coverage"] = {}
    for k, nominal in ((1.0, 0.6827), (2.0, 0.9545), (3.0, 0.9973)):
        out["coverage"]["%.0f_sigma" % k] = {
            "nominal": nominal, "observed": float(np.mean(r <= k * sd))}

    # a single global scale correction, fitted on two sessions and applied to
    # the third, so the correction is never scored on its own fitting data
    n1 = len(pred_sd[0]) if pred_sd else 0
    half = len(r) // 2
    scale = float(r[:half].mean() / max(sd[:half].mean(), 1e-9))
    zc = r[half:] / (sd[half:] * scale)
    out["fitted_scale"] = scale
    out["held_out_std_after_scaling"] = float(zc.std())
    out["held_out_coverage_after_scaling"] = {
        "1_sigma": float(np.mean(r[half:] <= 1.0 * sd[half:] * scale)),
        "2_sigma": float(np.mean(r[half:] <= 2.0 * sd[half:] * scale))}
    return out


# ==========================================================================
# Driver
# ==========================================================================

def show(rows: List[dict], title: str) -> None:
    print("\n  %-46s %8s %8s %8s %8s" % (title, "ECE", "Brier", "logloss", "AUC"))
    print("  " + "-" * 84)
    for r in rows:
        if r.get("failed"):
            print("  %-46s %8s   %s" % (r["name"], "FAILED", r["failed"][:34]))
            continue
        print("  %-46s %8.4f %8.4f %8.4f %8.4f"
              % (r["name"], r["ece_uniform"], r["brier"], r["logloss"], r["auc"]))
    print("\n  %-46s %s" % ("commit accuracy at matched prompt rate:",
                            "  ".join("%5.0f%%" % (x * 100) for x in MATCH_RATES)))
    print("  " + "-" * 84)
    base = rows[0]
    for r in rows:
        if r.get("failed"):
            continue
        cells = []
        for x in MATCH_RATES:
            v = r["frontier"][x]
            d = v - base["frontier"][x]
            cells.append("%.4f" % v if r is base else "%+.4f" % d)
        print("  %-46s %s" % (r["name"], "  ".join("%6s" % c for c in cells)))
    if len(rows) > 1:
        print("  (first row absolute; the rest are changes against it)")


def main() -> None:
    part = (sys.argv[1] if len(sys.argv) > 1 else "all").lower()
    out = {}
    print("=" * 88)
    print("CALIBRATION TRIALS — scikit-learn, on the commit gate")
    print("=" * 88)

    if part in ("b1", "b2", "all"):
        dec = HybridDecoder(vocab_path=VOCAB, model_dir=None, num_clusters=9)
        lm = LetterModel(dec.unigrams)
        tr_pairs = load_sentences(120, seed=3)
        te_pairs = load_sentences(120, seed=4, skip=2120)
        t = time.time()
        tr = keystroke_records(lm, tr_pairs, 40.0, seed=7)
        te = keystroke_records(lm, te_pairs, 40.0, seed=11)
        print("train %d keystrokes (%.1f%% correct) | held-out %d (%.1f%%) | %.0fs"
              % (len(tr["y"]), 100 * tr["y"].mean(),
                 len(te["y"]), 100 * te["y"].mean(), time.time() - t))
        print("THETA=%.2f commits %.1f%% of held-out keystrokes"
              % (B.THETA, 100 * np.mean(te["p"] >= B.THETA)))

    if part in ("b1", "all"):
        print("\n" + "=" * 88)
        print("B1. UNIVARIATE CALIBRATION OF THE COMMIT CONFIDENCE")
        print("=" * 88)
        rows = part_b1(tr, te)
        show(rows, "calibrator")
        out["b1"] = rows

    if part in ("b2", "all"):
        print("\n" + "=" * 88)
        print("B2. MULTIVARIATE CONFIDENCE MODEL")
        print("=" * 88)
        rows = part_b2(tr, te)
        show(rows, "model")
        out["b2"] = rows

    if part in ("b3", "all"):
        print("\n" + "=" * 88)
        print("B3. CALIBRATION OF THE GAZE UNCERTAINTY (recorded sessions)")
        print("=" * 88)
        r = part_b3()
        print("  samples ................................ %d" % r["n"])
        print("  mean predicted sd ...................... %7.1f px  (%.2f deg)"
              % (r["mean_predicted_sd_px"], r["mean_predicted_sd_deg"]))
        print("  mean actual residual ................... %7.1f px  (%.2f deg)"
              % (r["mean_actual_residual_px"], r["mean_actual_residual_deg"]))
        print("  actual / predicted ..................... %7.2fx" % r["ratio_actual_over_predicted"])
        print("  std of standardised residual (want 1.0)  %7.2f" % r["std_of_standardised_residual"])
        print("\n  coverage of the predictive interval:")
        for k, v in r["coverage"].items():
            print("    %-10s nominal %.4f   observed %.4f"
                  % (k.replace("_", " "), v["nominal"], v["observed"]))
        print("\n  single global scale correction fitted on the first half:")
        print("    scale ................................ %.2fx" % r["fitted_scale"])
        print("    held-out 1-sigma coverage ............ %.4f (nominal 0.6827)"
              % r["held_out_coverage_after_scaling"]["1_sigma"])
        print("    held-out 2-sigma coverage ............ %.4f (nominal 0.9545)"
              % r["held_out_coverage_after_scaling"]["2_sigma"])
        out["b3"] = r

    _save(RESULTS, out)
    print("\nwrote %s" % RESULTS)


if __name__ == "__main__":
    main()
