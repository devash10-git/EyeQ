"""Does a neutral-head calibration survive a head that moves?

Consumes a session recorded with "Include head movement" ticked, where the 13
fit targets are held at a normal posture and the 8 hold-out targets cycle
through tilts, turns and nods. Fitting on neutral and testing on moved is the
real-world question: people calibrate sitting still and then stop sitting still.

Compares the two canthus formulations the capture records side by side:

    canthus       offset in IMAGE axes      — rotates under the head
    canthus_eye   offset in the EYE frame   — invariant to roll by construction

    python3 eval/head_robustness.py
"""
from __future__ import annotations
import sys, warnings, math
warnings.filterwarnings("ignore")
import numpy as np
np.seterr(all="ignore")
sys.path.insert(0, ".")
from eval.gaze_bench import load_sessions, session_quality, fit_ridge

DEG = lambda r: r * 180 / math.pi


def build(sess, which):
    f = np.array([s["f"] for s in sess.extras], float)
    key = "canthus_eye" if which == "eye" else "canthus"
    c = np.array([s.get(key, s.get("canthus", [0, 0, 0, 0])) for s in sess.extras], float)
    return np.column_stack([f, c])


def main() -> None:
    sessions = [s for s in load_sessions() if session_quality(s)["usable"]]
    moved = [s for s in sessions if s.meta.get("protocol", {}).get("head_movement")]
    has_eye = [s for s in sessions if "canthus_eye" in (s.extras[0] if s.extras else {})]

    if not moved:
        print("No head-movement session found in eval/sessions/.\n")
        print("Record one:")
        print("    python3 eval/capture.py        ->  http://127.0.0.1:8322/capture")
        print("    tick 'Include head movement', then calibrate sitting normally.")
        print("    The hold-out targets will prompt each posture and wait until you hold it.\n")
        print(f"Sessions present: {len(sessions)} usable, "
              f"{len(has_eye)} carrying both canthus formulations, 0 with head movement.")
        print("\nWithout one, the eye-frame alignment fix cannot be validated on real data —")
        print("the existing recordings hold the head to within 2-3 degrees, where the two")
        print("formulations are indistinguishable.")
        return

    for sess in moved:
        W, H = sess.meta["screen"]["w_px"], sess.meta["screen"]["h_px"]
        deg = sess.px_per_degree
        fm, hm = sess.split == "fit", sess.split == "holdout"
        post = np.array([s.get("posture", "none") for s in sess.extras])
        roll = np.array([abs(DEG(s.get("roll", 0))) for s in sess.extras])

        print(f"\n=== {sess.label} ===")
        print(f"  roll range over the session: {roll.min():.1f} to {roll.max():.1f} deg")
        print(f"\n  {'posture':<14}{'n':>6}{'|roll|':>9}{'image axes':>14}{'eye frame':>13}{'change':>10}")

        fits = {}
        for which in ("image", "eye"):
            X = build(sess, which)
            fits[which] = (X, fit_ridge(X[fm], sess.T[fm]))

        totals = {"image": [], "eye": []}
        for p in [p for p in dict.fromkeys(post[hm])]:
            sel = hm & (post == p)
            if not sel.any():
                continue
            row = {}
            for which in ("image", "eye"):
                X, Wt = fits[which]
                P = np.clip(X[sel] @ Wt, [0, 0], [W, H])
                e = float(np.sqrt((((P - sess.T[sel]) ** 2).sum(1)).mean())) / deg
                row[which] = e
                totals[which].append(e)
            d = (1 - row["eye"] / row["image"]) * 100
            print(f"  {p:<14}{int(sel.sum()):>6}{roll[sel].mean():8.1f}°"
                  f"{row['image']:12.2f}°{row['eye']:12.2f}°{d:+9.1f}%")

        mi, me = float(np.mean(totals["image"])), float(np.mean(totals["eye"]))
        print(f"  {'all postures':<14}{'':>6}{'':>9}{mi:12.2f}°{me:12.2f}°"
              f"{(1 - me / mi) * 100:+9.1f}%")
        print(f"\n  verdict: {'eye-frame alignment helps under head movement' if me < mi * 0.97 else 'no clear difference — check the roll range actually achieved'}")


if __name__ == "__main__":
    main()
