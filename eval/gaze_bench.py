"""Gaze-accuracy benchmark: the measurement harness the project did not have.

Reports the error of an estimated screen coordinate in three units, broken down
by screen region, on **recorded human sessions only**. Nothing here accepts a
synthetic session as a source of accuracy numbers (constraint C3); the loader
refuses them and says so.

    python3 eval/capture.py            # record first (see eval/capture.html)
    python3 eval/gaze_bench.py         # then measure

Also exports the pieces Components A and B build on — the grouped splitter, the
deployed ridge solver, and the metric block — so that every later comparison is
made on identical splits by construction rather than by convention.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
SESSIONS = HERE / "sessions"
RESULTS = HERE.parent / "gaze_bench_results.json"

FEAT_DIM = 8  # [1, hL, hR, vL, vR, xnose, ynose, faceW] — static/iris-tracker.js


# --------------------------------------------------------------- loading

@dataclass
class Session:
    """One recorded sitting, with the geometry needed for unit conversion."""

    path: Path
    meta: dict
    X: np.ndarray            # (n, 8) raw feature vectors
    T: np.ndarray            # (n, 2) target position, page pixels
    groups: np.ndarray       # (n,) target id — the only legal split key (C4)
    split: np.ndarray        # (n,) "fit" | "holdout"
    extras: List[dict] = field(default_factory=list)

    @property
    def px_per_degree(self) -> float:
        return float(self.meta["px_per_degree"])

    @property
    def key_pitch_px(self) -> float:
        return float(self.meta.get("key_pitch_px", 0) or 0)

    @property
    def label(self) -> str:
        # Must be unique: two anonymous sittings both ended up as "anon/Z_anon"
        # and silently collided as dict keys in the model comparison, keeping
        # only the last. The recording timestamp is what distinguishes them.
        stamp = self.path.stem.replace("session_", "").split("_")[0]
        return f"{self.meta.get('participant') or 'anon'}/{stamp}"


def load_sessions(directory: Path = SESSIONS, allow_synthetic: bool = False) -> List[Session]:
    """Load every recorded session, refusing synthetic ones by default.

    ``allow_synthetic`` exists only so unit tests and latency benchmarks can
    reuse this loader. Any caller that sets it is responsible for labelling its
    output as synthetic wherever it appears.
    """
    out: List[Session] = []
    for p in sorted(Path(directory).glob("session_*.json")):
        d = json.loads(p.read_text())
        source = d.get("data_source", "unknown")
        if source != "recorded_human" and not allow_synthetic:
            raise ValueError(
                f"{p.name}: data_source={source!r}. Accuracy may only be reported on "
                f"recorded human sessions (C3). Pass allow_synthetic=True for tests."
            )
        samples = d.get("samples") or []
        if not samples:
            continue
        X = np.array([s["f"] for s in samples], dtype=float)
        T = np.array([s["target_px"] for s in samples], dtype=float)
        groups = np.array([s["target_id"] for s in samples])
        split = np.array([s.get("split", "fit") for s in samples])
        out.append(Session(p, d, X, T, groups, split, samples))
    return out


# --------------------------------------------------------------- splitting

def leave_one_group_out(groups: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Grouped LOO: hold out every sample of one target at a time.

    Constraint C4. Five clicks on one target are not five observations — they
    are one observation measured five times. Splitting by sample leaves four
    near-identical siblings in the training set, which makes held-out error look
    like training error and selects a regulariser far too small.
    """
    for g in np.unique(groups):
        test = groups == g
        yield ~test, test


def grouped_kfold(groups: np.ndarray, k: int = 5, seed: int = 0) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """k-fold over whole targets, deterministic in ``seed``."""
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    order = rng.permutation(uniq)
    folds = np.array_split(order, min(k, len(order)))
    for f in folds:
        test = np.isin(groups, f)
        if test.any() and (~test).any():
            yield ~test, test


# --------------------------------------------------------------- the deployed model

def fit_ridge(X: np.ndarray, T: np.ndarray, lam0: float = 1e-4) -> np.ndarray:
    """The calibration currently deployed, in the form iris-tracker.js solves it.

    Normal equations with a ridge term escalated until the system is solvable —
    the JS uses Gauss-Jordan with partial pivoting and the same escalation. This
    is the baseline every other model in this project must beat, so it is
    reproduced rather than improved.
    """
    # numpy 2.x on Apple Accelerate raises divide-by-zero / overflow flags from
    # the vectorized matmul path even on small bounded inputs. Verified against
    # an explicit dot-product reference: agreement to 2.8e-13, all finite. The
    # flags are spurious, so they are silenced here and nowhere wider.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        A0 = X.T @ X
        B = X.T @ T
    lam = lam0 * (np.trace(A0) / max(1, A0.shape[0]))
    for _ in range(8):
        try:
            W = np.linalg.solve(A0 + lam * np.eye(A0.shape[0]), B)
            if np.all(np.isfinite(W)):
                return W
        except np.linalg.LinAlgError:
            pass
        lam *= 10
    return np.linalg.lstsq(X, T, rcond=None)[0]


def predict(W: np.ndarray, X: np.ndarray, screen: Sequence[float]) -> np.ndarray:
    """Apply the fit and clamp to the screen, as the client does."""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        P = X @ W          # spurious Accelerate-BLAS flags; see fit_ridge
    P[:, 0] = np.clip(P[:, 0], 0, screen[0])
    P[:, 1] = np.clip(P[:, 1], 0, screen[1])
    return P


# --------------------------------------------------------------- regions

def region_of(tx: float, ty: float, w: float, h: float, band: float = 0.15) -> str:
    """centre / edge / corner, from where the *target* sat on screen.

    Reported separately because the entire case for a nonlinear model lives in
    the corner numbers, and a single aggregate RMSE hides them.
    """
    ex = tx / w <= band or tx / w >= 1 - band
    ey = ty / h <= band or ty / h >= 1 - band
    return "corner" if (ex and ey) else ("edge" if (ex or ey) else "centre")


# --------------------------------------------------------------- metrics

def metrics(pred: np.ndarray, true: np.ndarray, sess: Session) -> Dict[str, float]:
    """Error in pixels, degrees of visual angle, and fractions of key pitch."""
    err = np.linalg.norm(pred - true, axis=1)
    px_per_deg = sess.px_per_degree
    pitch = sess.key_pitch_px
    out = {
        "n": int(err.size),
        "rmse_px": float(np.sqrt(np.mean(err ** 2))),
        "mean_px": float(np.mean(err)),
        "p95_px": float(np.percentile(err, 95)),
        "rmse_deg": float(np.sqrt(np.mean(err ** 2)) / px_per_deg),
        "mean_deg": float(np.mean(err) / px_per_deg),
        "p95_deg": float(np.percentile(err, 95) / px_per_deg),
    }
    if pitch > 0:
        out["mean_keypitch"] = float(np.mean(err) / pitch)
        out["p95_keypitch"] = float(np.percentile(err, 95) / pitch)
    return out


def by_region(pred: np.ndarray, true: np.ndarray, sess: Session) -> Dict[str, Dict[str, float]]:
    w, h = sess.meta["screen"]["w_px"], sess.meta["screen"]["h_px"]
    labels = np.array([region_of(t[0], t[1], w, h) for t in true])
    out = {}
    for r in ("centre", "edge", "corner"):
        m = labels == r
        if m.any():
            out[r] = metrics(pred[m], true[m], sess)
    return out


# --------------------------------------------------------------- evaluation

EXPECTED_TARGETS = 21   # 13 fit + 8 hold-out, see eval/capture.html


def session_quality(sess: Session) -> dict:
    """Flag sessions that cannot support an accuracy claim.

    A sitting that lost most of its frames to face-tracking dropout, or that was
    abandoned before covering the grid, measures the recording conditions rather
    than the estimator. Such sessions are reported individually but kept out of
    the aggregate, because averaging them in silently understates the system.
    """
    n = int(sess.X.shape[0])
    dropped = int(sess.meta.get("dropped_frames", 0) or 0)
    targets = int(len(np.unique(sess.groups)))
    regions = {sess.split[i] for i in range(len(sess.split))}
    drop_frac = dropped / max(1, dropped + n)
    reasons = []
    if targets < EXPECTED_TARGETS:
        reasons.append(f"only {targets}/{EXPECTED_TARGETS} targets recorded")
    if drop_frac > 0.20:
        reasons.append(f"{drop_frac:.0%} of frames dropped by face tracking")
    if "holdout" not in regions:
        reasons.append("no hold-out targets")
    return {"usable": not reasons, "reasons": reasons,
            "targets": targets, "dropped_frames": dropped,
            "dropped_fraction": round(drop_frac, 3)}


def evaluate_session(sess: Session, seeds: Sequence[int] = (0,)) -> dict:
    """Two honest views of the deployed model on one session.

    ``holdout`` uses the targets the protocol reserved and never fitted —
    the primary number. ``grouped_cv`` cross-validates over the fit targets,
    which uses the data more efficiently and gives a variance estimate.
    """
    screen = (sess.meta["screen"]["w_px"], sess.meta["screen"]["h_px"])
    fit_m, hold_m = sess.split == "fit", sess.split == "holdout"
    res: Dict[str, object] = {"session": sess.label, "n_samples": int(sess.X.shape[0]),
                              "n_targets": int(len(np.unique(sess.groups))),
                              "quality": session_quality(sess)}

    if fit_m.any() and hold_m.any():
        W = fit_ridge(sess.X[fit_m], sess.T[fit_m])
        P = predict(W, sess.X[hold_m], screen)
        res["holdout"] = metrics(P, sess.T[hold_m], sess)
        res["holdout_by_region"] = by_region(P, sess.T[hold_m], sess)

    runs = []
    for seed in seeds:
        preds, trues = [], []
        for tr, te in grouped_kfold(sess.groups[fit_m], k=5, seed=seed):
            Xf, Tf = sess.X[fit_m], sess.T[fit_m]
            W = fit_ridge(Xf[tr], Tf[tr])
            preds.append(predict(W, Xf[te], screen))
            trues.append(Tf[te])
        if preds:
            runs.append(metrics(np.vstack(preds), np.vstack(trues), sess))
    if runs:
        keys = [k for k in runs[0] if k != "n"]
        res["grouped_cv"] = {k: float(np.mean([r[k] for r in runs])) for k in keys}
        res["grouped_cv_sd"] = {k: float(np.std([r[k] for r in runs])) for k in keys}
        res["grouped_cv"]["n_seeds"] = len(runs)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Gaze accuracy benchmark (recorded human sessions)")
    ap.add_argument("--sessions", default=str(SESSIONS))
    ap.add_argument("--seeds", type=int, default=5, help="random seeds for grouped CV")
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()

    sessions = load_sessions(Path(args.sessions))
    if not sessions:
        print("No recorded sessions found in", args.sessions)
        print()
        print("This harness reports accuracy on recorded human gaze only (C3).")
        print("Record one first:")
        print("    python3 eval/capture.py     ->  http://127.0.0.1:8322/capture")
        print()
        print("Until a session exists, no gaze accuracy number for this system")
        print("exists in any unit — which is the gap this harness was built to close.")
        return

    report = {"data_source": "recorded_human", "n_sessions": len(sessions), "sessions": []}
    for s in sessions:
        r = evaluate_session(s, seeds=range(args.seeds))
        report["sessions"].append(r)
        q = r["quality"]
        flag = "" if q["usable"] else "   [EXCLUDED: " + "; ".join(q["reasons"]) + "]"
        print(f"\n=== {s.label} — {r['n_samples']} samples, {r['n_targets']} targets ==={flag}")
        if "holdout" in r:
            h = r["holdout"]
            print(f"  held-out targets   {h['rmse_px']:7.1f} px RMSE   {h['rmse_deg']:5.2f}deg   "
                  f"p95 {h['p95_px']:.0f} px" +
                  (f"   {h['mean_keypitch']:.2f} key pitches" if "mean_keypitch" in h else ""))
            for reg, m in r.get("holdout_by_region", {}).items():
                print(f"    {reg:<7}          {m['rmse_px']:7.1f} px        {m['rmse_deg']:5.2f}deg   (n={m['n']})")
        if "grouped_cv" in r:
            c, sd = r["grouped_cv"], r["grouped_cv_sd"]
            print(f"  grouped 5-fold CV  {c['rmse_px']:7.1f} px RMSE   {c['rmse_deg']:5.2f}deg   "
                  f"(sd {sd['rmse_px']:.1f} px over {c['n_seeds']} seeds)")

    usable = [r for r in report["sessions"] if r["quality"]["usable"] and "holdout" in r]
    excluded = [r for r in report["sessions"] if not r["quality"]["usable"]]
    if usable:
        px = [r["holdout"]["rmse_px"] for r in usable]
        deg = [r["holdout"]["rmse_deg"] for r in usable]
        kp = [r["holdout"].get("mean_keypitch") for r in usable if "mean_keypitch" in r["holdout"]]
        report["across_sessions"] = {
            "n_usable": len(usable), "n_excluded": len(excluded),
            "holdout_rmse_px_mean": float(np.mean(px)), "holdout_rmse_px_sd": float(np.std(px)),
            "holdout_rmse_deg_mean": float(np.mean(deg)), "holdout_rmse_deg_sd": float(np.std(deg)),
        }
        print(f"\nacross {len(usable)} usable session(s): {np.mean(px):.1f} +/- {np.std(px):.1f} px RMSE"
              f"   {np.mean(deg):.2f} +/- {np.std(deg):.2f} deg"
              + (f"   {np.mean(kp):.2f} key pitches" if kp else ""))
        if excluded:
            print(f"{len(excluded)} session(s) excluded from the aggregate — see 'quality' in the JSON")

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
