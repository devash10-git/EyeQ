"""Does a residual-correction stage after the canthus ridge lower the 3.762 deg?

Pre-registered before any result was seen:

  Protocol   the paper's: 11 recorded sessions, 9 participants, leave-one-target-
             out over 21 targets, predictions clipped to the screen, mean
             Euclidean error per target, mean over targets per session, mean of
             sessions per participant. The "none" arm must reproduce 3.762 deg.
  Stage 1    the deployed ridge on the 12 canthus features, unchanged.
  Stage 2    learns the ridge's residual from the 14-d error state
             (quantum/residual_correction.py). Its training residuals are
             cross-fitted: each training target's residual comes from a ridge
             that did not see that target, as the held-out target's will.
  Arms       none | rbf-krr | svr | qk-fidelity | qk-projected   (continuous)
             qrgc, qrgc-ungated | twin, twin-ungated            (register)
  Selection  kernel arms tune on training targets only (grouped 5-fold);
             register and twin are fixed a priori (4 layers, 250 Adam steps).
             Every stage-2 arm trains on the same every-4th-frame subset.
  Adoption   QRGC enters the typing path only if it beats "none" AND its
             classical twin, both paired over participants at p < 0.05.

    python3 eval/qrgc_bench.py            # ~5 min on 10 cores
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from multiprocessing import Pool

for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

warnings.filterwarnings("ignore")
np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

from gaze_bench import load_sessions, session_quality, fit_ridge  # noqa: E402

OUT = os.path.join(HERE, "qrgc_results.json")
ARMS = ["none", "rbf-krr", "svr", "qk-fidelity", "qk-projected",
        "qrgc", "qrgc-ungated", "twin", "twin-ungated"]
SUB = 4                                   # kernel arms train on every 4th frame


def features(s):
    """The paper's 12: the deployed 8 plus the 4 image-axis canthus offsets."""
    f = np.array([e["f"] for e in s.extras], float)
    c = np.array([e["canthus"] for e in s.extras], float)
    return np.column_stack([f, c]), f, c


def clip(P, screen):
    P = P.copy()
    P[:, 0] = np.clip(P[:, 0], 0, screen[0])
    P[:, 1] = np.clip(P[:, 1], 0, screen[1])
    return P


def inner_folds(groups, k=5, seed=0):
    u = np.random.default_rng(seed).permutation(np.unique(groups))
    return [np.isin(groups, f) for f in np.array_split(u, k)]


def krr_fit(Ktr, R, lam):
    return np.linalg.solve(Ktr + lam * np.eye(len(Ktr)), R)


def tune_kernel(kernels, R, grp, lams):
    """kernels: {param: K(train,train)}. Grouped 5-fold on training targets."""
    folds, best = inner_folds(grp), None
    for p, Kfull in kernels.items():
        for lam in lams:
            err = 0.0
            for te in folds:
                tr = ~te
                mu = R[tr].mean(0)
                a = krr_fit(Kfull[np.ix_(tr, tr)], R[tr] - mu, lam)
                err += ((Kfull[np.ix_(te, tr)] @ a + mu - R[te]) ** 2).sum()
            if best is None or err < best[0]:
                best = (err, p, lam)
    return best[1], best[2]


def run_session(job):
    path, on = job
    import torch
    torch.set_num_threads(1)
    from sklearn.svm import SVR
    from quantum.residual_correction import (CorrectionRegister, ClassicalTwin, error_state,
                                             fidelity_kernel, fit_register, kernel_states,
                                             projected_features, readout)

    s = [x for x in load_sessions() if str(x.path) == path][0]
    X, f, C = features(s)
    T, G = s.T, s.groups
    screen = (s.meta["screen"]["w_px"], s.meta["screen"]["h_px"])
    ppd = s.px_per_degree
    t_ms = np.array([e["t"] for e in s.extras], float)
    fix = np.array([e["fixation"] for e in s.extras])
    stream = np.array([f"{g}|{k}" for g, k in zip(G, fix)])
    ear = np.array([[e["ear_l"], e["ear_r"]] for e in s.extras], float)
    bino_raw = np.column_stack([f[:, 1] - f[:, 2], f[:, 3] - f[:, 4]])

    err = {a: [] for a in on}
    diag = {"qrgc_conf": [], "twin_conf": [], "chosen": []}
    for g in np.unique(G):
        te, tr = G == g, G != g
        P_te = clip(X[te] @ fit_ridge(X[tr], T[tr]), screen)
        # cross-fitted stage-1 estimates on the training targets
        P_all = np.zeros_like(T)
        P_all[te] = P_te
        for h in np.unique(G[tr]):
            m = G == h
            P_all[m] = clip(X[m] @ fit_ridge(X[tr & ~m], T[tr & ~m]), screen)
        R = (T - P_all) / ppd                                         # degrees
        mb = bino_raw[tr].mean(0)
        binoc = np.linalg.norm(bino_raw - mb, axis=1)[:, None]
        E = error_state(C, P_all, t_ms, stream, ear, binoc, screen)
        mu, sd = E[tr].mean(0), E[tr].std(0) + 1e-12
        Z = (E - mu) / sd
        Ztr, Rtr, gtr, Zte = Z[tr], R[tr], G[tr], Z[te]
        sub = np.arange(len(Ztr)) % SUB == 0
        Zs, Rs, gs = Ztr[sub], Rtr[sub], gtr[sub]

        def score(name, D_deg):
            P = clip(P_te + D_deg * ppd, screen)
            err[name].append(np.linalg.norm(P - T[te], axis=1).mean())

        score("none", np.zeros_like(P_te))

        d2 = lambda A, B: ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        mu_r = Rs.mean(0)
        diag["chosen"].append({})

        # rbf kernel ridge on the residual
        if "rbf-krr" in on:
          Dss = d2(Zs, Zs)
          gams = [0.003, 0.01, 0.03, 0.1, 0.3]
          gam, lam = tune_kernel({gm: np.exp(-gm * Dss) for gm in gams}, Rs, gs, [0.01, 0.1, 1, 10])
          a = krr_fit(np.exp(-gam * Dss), Rs - mu_r, lam)
          score("rbf-krr", np.exp(-gam * d2(Zte, Zs)) @ a + mu_r)
          diag["chosen"][-1]["rbf"] = [gam, lam]

        # svr on the residual, one per axis
        if "svr" in on:
         best = None
         for Cc in [0.1, 1, 10]:
             for gm in [0.01, 0.03, 0.1]:
                 e = 0.0
                 for fte in inner_folds(gs):
                     for ax in range(2):
                         m = SVR(C=Cc, gamma=gm, epsilon=0.1).fit(Zs[~fte], Rs[~fte, ax])
                         e += ((m.predict(Zs[fte]) - Rs[fte, ax]) ** 2).sum()
                 if best is None or e < best[0]:
                     best = (e, Cc, gm)
         D = np.column_stack([SVR(C=best[1], gamma=best[2], epsilon=0.1).fit(Zs, Rs[:, ax]).predict(Zte)
                              for ax in range(2)])
         score("svr", D)

        # quantum fidelity kernel ridge on the residual
        if "qk-fidelity" in on or "qk-projected" in on:
         scales = [0.1, 0.25, 0.5, 1.0]
         st = {sc: kernel_states(Zs, sc) for sc in scales}
         sc, lam = tune_kernel({sc: fidelity_kernel(st[sc], st[sc]) for sc in scales}, Rs, gs,
                               [0.01, 0.1, 1, 10])
         a = krr_fit(fidelity_kernel(st[sc], st[sc]), Rs - mu_r, lam)
         if "qk-fidelity" in on: score("qk-fidelity", fidelity_kernel(kernel_states(Zte, sc), st[sc]) @ a + mu_r)
         diag["chosen"][-1]["qk-fid"] = [sc, lam]

         # projected quantum kernel ridge on the residual
         pf = {sc: projected_features(st[sc]) for sc in scales}
         cand = {(sc, gm): np.exp(-gm * d2(pf[sc], pf[sc])) for sc in scales for gm in [0.1, 0.3, 1.0]}
         (sc, gm), lam = tune_kernel(cand, Rs, gs, [0.01, 0.1, 1, 10])
         a = krr_fit(cand[(sc, gm)], Rs - mu_r, lam)
         Fte = projected_features(kernel_states(Zte, sc))
         if "qk-projected" in on: score("qk-projected", np.exp(-gm * d2(Fte, pf[sc])) @ a + mu_r)
         diag["chosen"][-1]["qk-proj"] = [sc, gm, lam]

        # the correction-hypothesis register, and its classical twin
        for name, model in (("qrgc", CorrectionRegister(Z.shape[1])), ("twin", ClassicalTwin(Z.shape[1]))):
            if name not in on:
                continue
            pred = fit_register(model, Zs, Rs)
            px, py = pred(Zte)
            score(name, readout(px, py, gate=True))
            score(name + "-ungated", readout(px, py, gate=False))
            conf = np.abs(readout(px, py, gate=True)).sum() / max(1e-12, np.abs(readout(px, py, gate=False)).sum())
            diag[name + "_conf"].append(float(conf))

    return {"session": s.label, "participant": s.meta["participant"], "ppd": ppd,
            "deg": {a: float(np.mean(v)) / ppd for a, v in err.items()},
            "per_target_px": {a: [float(x) for x in v] for a, v in err.items()},
            "mean_gate": {k: float(np.mean(v)) for k, v in diag.items() if k != "chosen"},
            "chosen": diag["chosen"]}


def main():
    import argparse
    from scipy.stats import ttest_rel, wilcoxon
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS),
                    help="subset, e.g. none,qrgc,twin (qrgc/twin bring their ungated readouts)")
    want = ap.parse_args().arms.split(",")
    on = [a for a in ARMS if a in want or a.replace("-ungated", "") in want]
    out = OUT if on == ARMS else OUT.replace(".json", "_" + "-".join(want) + ".json")
    S = [s for s in load_sessions() if session_quality(s)["usable"]]
    t0 = time.time()
    rows = []
    with Pool(len(S)) as pool:
        for r in pool.imap_unordered(run_session, [(str(s.path), tuple(on)) for s in S]):
            rows.append(r)
            print(f"  session {len(rows)}/{len(S)} done ({time.time() - t0:.0f}s)", flush=True)
    print(f"{len(rows)} sessions in {time.time() - t0:.0f}s\n")
    parts = sorted({r["participant"] for r in rows})
    P = {a: np.array([np.mean([r["deg"][a] for r in rows if r["participant"] == p]) for p in parts])
         for a in on}
    base = P["none"]
    print(f"{'stage 2':<14}{'error (deg)':>18}{'vs none':>10}{'t-test p':>11}{'wilcoxon p':>12}{'better':>9}")
    summary = {}
    for a in on:
        m, sdv = P[a].mean(), P[a].std(ddof=1)
        if a == "none":
            print(f"{a:<14}{m:>10.3f} +/- {sdv:.3f}{'baseline':>10}")
            summary[a] = {"mean": m, "sd": sdv}
            continue
        d = 100 * (base.mean() - m) / base.mean()
        pt = ttest_rel(base, P[a]).pvalue
        pw = wilcoxon(base, P[a]).pvalue if np.any(base != P[a]) else 1.0
        nb = int((P[a] < base).sum())
        print(f"{a:<14}{m:>10.3f} +/- {sdv:.3f}{d:>+9.1f}%{pt:>11.3g}{pw:>12.3g}{nb:>6}/{len(parts)}")
        summary[a] = {"mean": m, "sd": sdv, "reduction_pct": d, "t_p": pt, "wilcoxon_p": pw, "better": nb}

    print("\nquantum vs classical, paired over participants:")
    pairs = [("qrgc", "twin"), ("qrgc-ungated", "twin-ungated"), ("qk-fidelity", "rbf-krr"),
             ("qk-projected", "rbf-krr"), ("qrgc", "rbf-krr")]
    head = {}
    for q, c in [pc for pc in pairs if pc[0] in on and pc[1] in on]:
        d = 100 * (P[c].mean() - P[q].mean()) / P[c].mean()
        p = ttest_rel(P[c], P[q]).pvalue
        print(f"  {q:<13} vs {c:<13} {d:+6.1f}%   p = {p:.3g}   quantum better in {int((P[q] < P[c]).sum())}/{len(parts)}")
        head[f"{q}_vs_{c}"] = {"pct": d, "p": p}

    q = summary["qrgc"]
    adopt = ("qrgc_vs_twin" in head and q["reduction_pct"] > 0 and q["t_p"] < 0.05 and head["qrgc_vs_twin"]["pct"] > 0
             and head["qrgc_vs_twin"]["p"] < 0.05)
    print(f"\nadoption rule (beats none AND twin, p<0.05): {'PASS' if adopt else 'FAIL'}")
    gate = np.mean([r["mean_gate"]["qrgc_conf"] for r in rows])
    print(f"mean gate on the register's corrections: {gate:.2f} (1 = applied in full)")

    json.dump({"protocol": __doc__, "participants": parts, "summary": summary, "quantum_vs_classical": head,
               "adopt": bool(adopt), "arms": on, "per_participant_deg": {a: P[a].tolist() for a in on},
               "sessions": rows}, open(out, "w"), indent=1, default=float)
    print(f"\nwritten {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
