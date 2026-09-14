"""Unit tests for the measurement harness.

**All data here is synthetic** and is generated only to exercise the code paths;
no number produced by this file is an accuracy result, and the loader's C3 guard
is itself one of the things under test.

    python3 eval/test_gaze_bench.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from gaze_bench import (Session, by_region, fit_ridge, grouped_kfold, leave_one_group_out,
                        load_sessions, metrics, predict, region_of)

W, H = 1512, 850
FIT_XS, FIT_YS = [0.06, 0.5, 0.94], [0.06, 0.5, 0.94]


def synth_session(path: Path, noise=0.004, source="synthetic_unit_test", seed=0) -> Path:
    """A projective iris->screen map plus landmark noise. SYNTHETIC — tests only."""
    rng = np.random.default_rng(seed)
    fit = [(x, y) for y in FIT_YS for x in FIT_XS] + [(x, y) for y in (0.30, 0.70) for x in (0.30, 0.70)]
    hold = [(0.10, 0.10), (0.90, 0.10), (0.10, 0.90), (0.90, 0.90),
            (0.50, 0.10), (0.50, 0.90), (0.40, 0.62), (0.62, 0.40)]
    samples = []
    for split, targets in (("fit", fit), ("holdout", hold)):
        for ti, (tx, ty) in enumerate(targets):
            for fx in range(5):
                for _ in range(6):
                    # inverse of a tan-like forward model, so the true relation
                    # is genuinely nonlinear in the iris ratios
                    ang_x = np.arctan((tx - 0.5) * 2 * 0.27)
                    ang_y = np.arctan((ty - 0.5) * 2 * 0.17)
                    h = 0.5 + 0.42 * np.sin(ang_x) + rng.normal(0, noise)
                    v = 0.5 + 0.55 * np.sin(ang_y) + rng.normal(0, noise)
                    samples.append({
                        "t": 0.0, "target_x": tx, "target_y": ty,
                        "target_px": [26 + tx * (W - 52), 26 + ty * (H - 52)],
                        "target_id": f"{split}_{ti:02d}", "split": split, "fixation": fx,
                        "f": [1.0, h, h + rng.normal(0, noise / 2), v, v + rng.normal(0, noise / 2),
                              0.5 + rng.normal(0, 0.002), 0.55 + rng.normal(0, 0.002), 0.28],
                        "iris_r_l": 0.012, "iris_r_r": 0.012, "ear_l": 0.30, "ear_r": 0.30,
                        "canthus": [0.0, 0.0, 0.0, 0.0], "gaze_velocity": 0.001,
                        "quality_geometric": 0.9, "yaw": 0.0, "pitch": 0.0, "roll": 0.0,
                    })
    path.write_text(json.dumps({
        "schema": 1, "session_id": "test", "participant": "synthetic",
        "recorded_utc": "1970-01-01T00:00:00Z", "data_source": source,
        "screen": {"w_px": W, "h_px": H, "dpr": 2, "w_cm": 29.4, "h_cm": 16.5},
        "viewing_distance_cm": 55.0, "px_per_cm": W / 29.4,
        "px_per_degree": np.tan(np.pi / 180) * 55.0 * (W / 29.4),
        "key_pitch_px": ((W - 24 - 90) / 10) + 10,
        "protocol": {}, "dropped_frames": 0, "samples": samples,
    }))
    return path


def main() -> None:
    failures = []
    def check(name, cond, detail=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        if not cond:
            failures.append(name)

    print("Unit tests (SYNTHETIC DATA — not accuracy results)\n")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        synth_session(d / "session_a.json")

        # --- C3: the loader must refuse synthetic data for accuracy by default
        refused = False
        try:
            load_sessions(d)
        except ValueError as exc:
            refused = "data_source" in str(exc)
        check("C3 guard refuses synthetic sessions by default", refused)

        sessions = load_sessions(d, allow_synthetic=True)
        check("loader reads a session with allow_synthetic", len(sessions) == 1)
        s = sessions[0]
        check("feature matrix is (n, 8)", s.X.ndim == 2 and s.X.shape[1] == 8, str(s.X.shape))
        check("fit and holdout are both present",
              (s.split == "fit").any() and (s.split == "holdout").any())

        # --- C4: splits must be grouped by target, never by sample
        groups = s.groups[s.split == "fit"]
        leaks = 0
        for tr, te in grouped_kfold(groups, k=5, seed=0):
            if set(np.unique(groups[tr])) & set(np.unique(groups[te])):
                leaks += 1
        check("C4 grouped k-fold leaks no target across the split", leaks == 0)
        n_loo = sum(1 for _ in leave_one_group_out(groups))
        check("leave-one-target-out yields one fold per target",
              n_loo == len(np.unique(groups)), f"{n_loo} folds")

        # --- units
        m = metrics(np.zeros((4, 2)), np.array([[40.0, 0], [0, 40.0], [40.0, 0], [0, 40.0]]), s)
        check("px and degrees are consistent",
              abs(m["mean_px"] / s.px_per_degree - m["mean_deg"]) < 1e-9,
              f"{m['mean_px']:.1f} px = {m['mean_deg']:.2f} deg")
        check("key-pitch fraction is reported", "mean_keypitch" in m)

        # --- regions
        check("corner classified", region_of(20, 20, W, H) == "corner")
        check("edge classified", region_of(W / 2, 12, W, H) == "edge")
        check("centre classified", region_of(W / 2, H / 2, W, H) == "centre")

        # --- the deployed model actually fits, and the harness sees edge error
        fm, hm = s.split == "fit", s.split == "holdout"
        Wt = fit_ridge(s.X[fm], s.T[fm])
        P = predict(Wt, s.X[hm], (W, H))
        check("ridge produces finite predictions", bool(np.all(np.isfinite(P))))
        reg = by_region(P, s.T[hm], s)
        check("hold-out spans all three regions", len(reg) == 3, ", ".join(sorted(reg)))
        if "corner" in reg and "centre" in reg:
            # Report the split, claim nothing about it. Whether the corners are
            # worse is an empirical question about real optics; this fixture's
            # curvature is mild by construction, so it must not be read as
            # evidence either way.
            print(f"        (synthetic fixture) centre {reg['centre']['rmse_px']:.1f} px, "
                  f"corner {reg['corner']['rmse_px']:.1f} px — regions separate correctly; "
                  f"the real comparison needs recorded data")

        # --- noise must move the number in the right direction
        synth_session(d / "session_b.json", noise=0.02, seed=1)
        two = load_sessions(d, allow_synthetic=True)
        errs = []
        for sx in two:
            f2, h2 = sx.split == "fit", sx.split == "holdout"
            Wx = fit_ridge(sx.X[f2], sx.T[f2])
            errs.append(metrics(predict(Wx, sx.X[h2], (W, H)), sx.T[h2], sx)["rmse_px"])
        check("more landmark noise gives more error", errs[1] > errs[0],
              f"{errs[0]:.1f} px -> {errs[1]:.1f} px")

    print(f"\n{'ALL PASS' if not failures else 'FAILED: ' + ', '.join(failures)}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
