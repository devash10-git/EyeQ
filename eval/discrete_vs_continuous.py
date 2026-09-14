"""Two questions, measured on recorded human gaze.

1. Would a *quantum* nonlinear regressor beat the deployed linear ridge, now
   that the canthus features are in?
2. Would *discrete direction classification* — the approach nine of the ten
   reviewed papers take — be more reliable than pointing at a key directly?

    python3 eval/discrete_vs_continuous.py
"""
from __future__ import annotations
import sys, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
np.seterr(all="ignore")
sys.path.insert(0, ".")
from eval.gaze_bench import load_sessions, session_quality, fit_ridge

VEC = lambda s: np.array(s["f"] + list(s.get("canthus", [0, 0, 0, 0])), float)


def per_target(sess, mask):
    ids = sess.groups[mask]
    X = np.array([VEC(e) for e, m in zip(sess.extras, mask) if m])
    T = sess.T[mask]
    out_X, out_T = [], []
    for g in np.unique(ids):
        sel = ids == g
        out_X.append(X[sel].mean(axis=0)); out_T.append(T[sel].mean(axis=0))
    return np.array(out_X), np.array(out_T)


# ---------------------------------------------------------------- Q1
def q1(sessions):
    from quantum.models import QuantumKernel
    print("=" * 96)
    print("Q1  Quantum nonlinear regression on the 12 canthus features")
    print("=" * 96)
    print(f"  {'model':<34}{'deg':>9}{'vs ridge':>12}   note")
    agg = {}
    for sess in sessions:
        fm, hm = sess.split == "fit", sess.split == "holdout"
        Xf, Tf = per_target(sess, fm)
        # Hold-out collapsed per target too. The full 744 samples would need
        # ~9,700 twelve-qubit circuits for the cross-Gram; the 8 target means
        # carry the same per-target accuracy at 104.
        Xh, Th = per_target(sess, hm)
        W, H = sess.meta["screen"]["w_px"], sess.meta["screen"]["h_px"]
        deg = sess.px_per_degree

        mu, sd = Xf.mean(0), Xf.std(0) + 1e-9
        Zf, Zh = (Xf - mu) / sd, (Xh - mu) / sd

        def err(P):
            P = np.clip(P, [0, 0], [W, H])
            return float(np.sqrt((((P - Th) ** 2).sum(1)).mean())) / deg

        agg.setdefault("linear ridge (deployed)", []).append(err(Xh @ fit_ridge(Xf, Tf)))

        d2 = ((Zf[:, None, :] - Zf[None, :, :]) ** 2).sum(-1)
        med = np.median(d2[d2 > 0]) or 1.0
        best = None
        for g in [0.25, 0.5, 1, 2, 4]:
            gam = g / med
            K = np.exp(-gam * d2)
            Kh = np.exp(-gam * ((Zh[:, None, :] - Zf[None, :, :]) ** 2).sum(-1))
            for lam in [1e-4, 1e-3, 1e-2, 1e-1, 1]:
                a = np.linalg.solve(K + lam * np.eye(len(K)), Tf)
                e = err(Kh @ a)
                if best is None or e < best: best = e
        agg.setdefault("classical RBF kernel ridge", []).append(best)

        # angles into [0, pi] as the encoders expect
        Af = np.pi / (1 + np.exp(-Zf)); Ah = np.pi / (1 + np.exp(-Zh))
        for strategy in ["angle", "zz"]:
            t0 = time.time()
            qk = QuantumKernel(n_features=Af.shape[1], strategy=strategy, reps=2)
            Kq = qk.gram(Af); Kqh = qk.gram(Ah, Af)
            bq = None
            for lam in [1e-4, 1e-3, 1e-2, 1e-1, 1]:
                a = np.linalg.solve(Kq + lam * np.eye(len(Kq)), Tf)
                e = err(Kqh @ a)
                if bq is None or e < bq: bq = e
            agg.setdefault(f"quantum kernel ridge ({strategy})", []).append(bq)
            agg.setdefault(f"__t_{strategy}", []).append(time.time() - t0)
            agg.setdefault(f"__n_{strategy}", []).append(qk.n_evaluations)

    base = float(np.mean(agg["linear ridge (deployed)"]))
    for k in ["linear ridge (deployed)", "classical RBF kernel ridge",
              "quantum kernel ridge (angle)", "quantum kernel ridge (zz)"]:
        v = float(np.mean(agg[k]))
        tag = "baseline" if k.startswith("linear") else f"{(1 - v / base) * 100:+.1f}%"
        note = ""
        if k.startswith("quantum"):
            s = k.split("(")[1][:-1]
            note = f"{int(np.mean(agg['__n_' + s]))} circuits, {np.mean(agg['__t_' + s]):.1f}s"
        print(f"  {k:<34}{v:9.2f}{tag:>12}   {note}")


# ---------------------------------------------------------------- Q2
def q2(sessions):
    print()
    print("=" * 96)
    print("Q2  Discrete direction classification vs pointing at the key")
    print("=" * 96)
    ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
    for sess in sessions:
        W, H = sess.meta["screen"]["w_px"], sess.meta["screen"]["h_px"]
        PAD, GAP = 12, 10
        TOP, ROW_H = 0.30 * H, (0.66 * H) / 3
        keys = []
        for r, row in enumerate(ROWS):
            n = len(row); kw = (W - 2 * PAD - GAP * (n - 1)) / n
            for i, ch in enumerate(row):
                keys.append((ch, PAD + i * (kw + GAP), TOP + r * ROW_H, kw, ROW_H - GAP))
        kb = (PAD, TOP, W - 2 * PAD, 3 * ROW_H)

        fm, hm = sess.split == "fit", sess.split == "holdout"
        Xf = np.array([VEC(e) for e, m in zip(sess.extras, fm) if m])
        Xh = np.array([VEC(e) for e, m in zip(sess.extras, hm) if m])
        Wt = fit_ridge(Xf, sess.T[fm])
        P = np.clip(Xh @ Wt, [0, 0], [W, H])
        Tt = sess.T[hm]
        offs = P - Tt          # the error vector; applied to a notional key centre

        def hit_rate(cols, rows):
            """Fraction of gaze errors small enough to stay in the right cell."""
            cw, ch = kb[2] / cols, kb[3] / rows
            ok = (np.abs(offs[:, 0]) < cw / 2) & (np.abs(offs[:, 1]) < ch / 2)
            return float(ok.mean()), cw, ch

        kw_med = float(np.median([k[3] for k in keys])); kh_med = float(np.median([k[4] for k in keys]))
        direct = float((( np.abs(offs[:, 0]) < kw_med / 2) & (np.abs(offs[:, 1]) < kh_med / 2)).mean())
        print(f"\n  {sess.label}   error {np.sqrt((offs**2).sum(1).mean()) / sess.px_per_degree:.2f} deg")
        print(f"    {'scheme':<30}{'cell size':>16}{'lands in right cell':>22}{'dwells/letter':>15}")
        print(f"    {'direct key (deployed)':<30}{f'{kw_med:.0f}x{kh_med:.0f} px':>16}{direct * 100:21.1f}%{1:>15}")
        for cols, rows, label in [(2, 2, "2x2 regions"), (3, 3, "3x3 regions"), (2, 4, "2x4 grid (paper)")]:
            r, cw, ch = hit_rate(cols, rows)
            # level 2 blows the chosen region up to the full keyboard area
            n_in = int(np.ceil(len(keys) / (cols * rows)))
            r2, _, _ = hit_rate(max(1, int(np.ceil(np.sqrt(n_in)))), max(1, int(np.ceil(np.sqrt(n_in)))))
            print(f"    {label + ' -> then key':<30}{f'{cw:.0f}x{ch:.0f} px':>16}{r * r2 * 100:21.1f}%{2:>15}")

        DWELL = (693 + 450) / 1000
        print(f"    throughput at {DWELL:.2f}s per dwell, ignoring corrections:")
        print(f"      direct   {60 / DWELL * direct:5.1f} correct letters/min")
        r, _, _ = hit_rate(3, 3); n_in = int(np.ceil(len(keys) / 9))
        r2, _, _ = hit_rate(int(np.ceil(np.sqrt(n_in))), int(np.ceil(np.sqrt(n_in))))
        print(f"      3x3 two-level  {60 / (2 * DWELL) * r * r2:5.1f} correct letters/min")


if __name__ == "__main__":
    sessions = [s for s in load_sessions() if session_quality(s)["usable"]]
    q1(sessions)
    q2(sessions)
