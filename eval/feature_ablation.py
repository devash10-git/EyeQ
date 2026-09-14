"""Do the optical features genuinely help, and is head pose still needed?

Leave-one-target-out over all 21 recorded positions, not the single hold-out
split. With 21 folds instead of one 8-target hold-out, the estimate is far less
dependent on which targets happened to be reserved — and with 15 features fitted
against 13 distinct positions in the original split, that mattered.

    python3 eval/feature_ablation.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np; np.seterr(all="ignore")
from gaze_bench import load_sessions, session_quality, fit_ridge


def columns(sess, kind):
    ex = sess.extras
    f = np.array([s["f"] for s in ex], float)
    hL, hR, vL, vR = f[:, 1], f[:, 2], f[:, 3], f[:, 4]
    h, v = (hL + hR) / 2, (vL + vR) / 2
    yaw = np.array([s.get("yaw", 0) for s in ex], float)
    pitch = np.array([s.get("pitch", 0) for s in ex], float)
    rl = np.array([s.get("iris_r_l", 0) for s in ex], float)
    rr = np.array([s.get("iris_r_r", 0) for s in ex], float)
    can = np.array([s.get("canthus", [0, 0, 0, 0]) for s in ex], float)

    cols = [f[:, 0], hL, hR, vL, vR, f[:, 5], f[:, 6], f[:, 7]]      # the deployed 8
    if "radius" in kind or kind in ("optics", "optics_pose", "all"):
        cols += [rl, rr, (rl + rr) / 2]
    if "canthus" in kind or kind in ("optics", "optics_pose", "all"):
        cols += [can[:, 0], can[:, 1], can[:, 2], can[:, 3]]
    if kind in ("pose", "optics_pose", "all"):
        cols += [yaw, pitch, h * yaw, v * pitch, h * pitch, v * yaw]
    if kind == "all":
        cols += [h * h, v * v, h * v]
    return np.column_stack(cols)


SETS = ["deployed", "radius", "canthus", "optics", "pose", "optics_pose", "all"]
LABEL = {"deployed": "deployed 8", "radius": "+ iris radius", "canthus": "+ canthus",
         "optics": "+ both optical", "pose": "+ head pose only",
         "optics_pose": "+ optical + pose", "all": "+ everything"}

sessions = [s for s in load_sessions() if session_quality(s)["usable"]]
print("Leave-one-target-out over all 21 positions, per session\n")
print(f"{'feature set':<18}{'dims':>5}" + "".join(f"{s.label[-7:]:>16}" for s in sessions) + f"{'pooled':>11}   vs deployed")

pooled = {}
for kind in SETS:
    per, dims = [], 0
    for s in sessions:
        X = columns(s, kind); dims = X.shape[1]
        W, H = s.meta["screen"]["w_px"], s.meta["screen"]["h_px"]
        errs = []
        for g in np.unique(s.groups):
            te = s.groups == g
            Wt = fit_ridge(X[~te], s.T[~te])
            P = X[te] @ Wt
            P[:, 0] = np.clip(P[:, 0], 0, W); P[:, 1] = np.clip(P[:, 1], 0, H)
            errs.append(np.sqrt((((P - s.T[te]) ** 2).sum(axis=1)).mean()))
        per.append((float(np.mean(errs)), float(np.std(errs))))
    pooled[kind] = float(np.mean([p[0] for p in per]))
    base = pooled["deployed"]
    d = (1 - pooled[kind] / base) * 100
    tag = "  <- baseline" if kind == "deployed" else f"  {d:+5.1f}%"
    cells = "".join(f"{m:9.1f}+-{sd:<4.0f}" for m, sd in per)
    print(f"{LABEL[kind]:<18}{dims:>5}{cells}{pooled[kind]:10.1f}px{tag}")

print("\nIs head pose still worth adding once the optical features are in?")
a, b = pooled["optics"], pooled["optics_pose"]
print(f"  optical only        {a:7.1f} px")
print(f"  optical + head pose {b:7.1f} px   {(1 - b / a) * 100:+.1f}%")
print(f"  -> {'pose still earns its place' if (1 - b / a) * 100 > 3 else 'pose adds little once canthus normalisation is present'}")

deg = float(np.mean([s.px_per_degree for s in sessions]))
print(f"\n  deployed {pooled['deployed'] / deg:.2f} deg  ->  optical {pooled['optics'] / deg:.2f} deg")
