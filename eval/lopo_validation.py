"""Leave-one-participant-out model selection: does a nonlinear win transfer?

The estimator comparison in the paper is leave-one-target-out within a session,
and seven candidates beat the deployed ridge there. That is the wrong question
for a system that must work on a person it has never seen. This script asks the
right one: choose the model on eight participants, score it on the ninth, and
repeat. A margin that does not survive a new face is not a margin.

Model selection happens strictly inside the training fold, so the held-out
participant contributes nothing to the choice. Splits within a session remain
grouped by target throughout.

Run: python3 eval/lopo_validation.py
"""
from __future__ import annotations
import sys, os, warnings, json, glob
import numpy as np
warnings.filterwarnings("ignore"); np.seterr(all="ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval.gaze_bench import load_sessions, session_quality, fit_ridge
from eval.feature_ablation import columns
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.linear_model import HuberRegressor, RidgeCV
from sklearn.multioutput import MultiOutputRegressor

def incumbent(a,b,c): return c @ fit_ridge(a,b)
def krr(a,b,c):
    return make_pipeline(StandardScaler(), GridSearchCV(KernelRidge(kernel="rbf"),
        {"alpha":np.logspace(-4,2,7),"gamma":np.logspace(-3,1,5)}, cv=3,
        scoring="neg_mean_squared_error")).fit(a[:,1:],b).predict(c[:,1:])
def poly2(a,b,c):
    return make_pipeline(StandardScaler(), PolynomialFeatures(2,include_bias=False),
        GridSearchCV(KernelRidge(kernel="linear"),{"alpha":np.logspace(-4,2,7)},cv=3,
        scoring="neg_mean_squared_error")).fit(a[:,1:],b).predict(c[:,1:])
def svr(a,b,c):
    return make_pipeline(StandardScaler(), MultiOutputRegressor(SVR(kernel="rbf"))).fit(a[:,1:],b).predict(c[:,1:])
def huber(a,b,c):
    return make_pipeline(StandardScaler(), MultiOutputRegressor(HuberRegressor(max_iter=500))).fit(a[:,1:],b).predict(c[:,1:])
def ridgecv(a,b,c):
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4,2,13))).fit(a[:,1:],b).predict(c[:,1:])
CAND = [("KernelRidge RBF",krr), ("Polynomial d=2 + Ridge",poly2), ("SVR RBF",svr),
        ("HuberRegressor",huber), ("sklearn RidgeCV",ridgecv)]

def score(fn, ss, folds=None):
    """Mean error over sessions, splits grouped by target.

    ``folds`` sets how finely the targets are split. The OUTER score uses
    leave-one-target-out, matching the rest of the paper. The INNER selection
    uses three grouped folds instead: it only has to rank candidates, not
    measure them, and leave-one-target-out inside every training session costs
    21 fits per session per candidate, which makes nine folds of nested
    selection take hours rather than minutes. Both remain grouped by target, so
    neither leaks near-duplicate frames across the split.
    """
    e = []
    for s in ss:
        X, T = columns(s,"canthus"), s.T
        W, H = s.meta["screen"]["w_px"], s.meta["screen"]["h_px"]
        tids = np.unique(s.groups)
        groups = ([[t] for t in tids] if folds is None
                  else [tids[i::folds] for i in range(folds)])
        p = np.empty_like(T)
        for gg in groups:
            te = np.isin(s.groups, gg)
            if te.all() or not te.any(): continue
            p[te] = np.asarray(fn(X[~te],T[~te],X[te]), dtype=float)
        p[:,0] = np.clip(p[:,0],0,W); p[:,1] = np.clip(p[:,1],0,H)
        e.append(float(np.linalg.norm(p-T,axis=1).mean())/s.px_per_degree)
    return float(np.mean(e))

def main():
    S = [s for s in load_sessions() if session_quality(s)["usable"]]
    pid = {s.label: json.load(open([f for f in glob.glob("eval/sessions/*.json")
           if s.label.split("/")[-1][:15] in f][0]))["participant"] for s in S}
    parts = sorted(set(pid.values()))
    print("LEAVE-ONE-PARTICIPANT-OUT — %d folds, %d sessions" % (len(parts), len(S)))
    print("  %-6s %-24s %9s %9s %9s %9s" % ("fold","picked on training","inner","outer","incumbent","gain"))
    print("  " + "-"*72)
    rows = []
    for hp in parts:
        tr = [s for s in S if pid[s.label] != hp]
        te = [s for s in S if pid[s.label] == hp]
        inner = {n: score(f, tr, folds=3) for n, f in CAND}
        best = min(inner, key=inner.get)
        out = score(dict(CAND)[best], te); inc = score(incumbent, te)
        rows.append(dict(held_out=hp, picked=best, inner=inner[best],
                         outer=out, incumbent=inc, gain=100*(inc-out)/inc))
        print("  %-6s %-24s %8.3f° %8.3f° %8.3f° %+8.1f%%"
              % (hp, best, inner[best], out, inc, rows[-1]["gain"]), flush=True)
    g = np.array([r["gain"] for r in rows])
    print("  " + "-"*72)
    print("  distinct winners across folds: %d  %s"
          % (len(set(r["picked"] for r in rows)), sorted(set(r["picked"] for r in rows))))
    print("  mean gain over the incumbent: %+.2f%%  (range %+.1f%% to %+.1f%%)" % (g.mean(), g.min(), g.max()))
    print("  folds where the pick beat the incumbent: %d / %d" % (int((g>0).sum()), len(g)))
    json.dump({"folds": rows, "mean_gain_pct": float(g.mean()),
               "n_participants": len(parts), "n_sessions": len(S)},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "lopo_results.json"), "w"), indent=2)

if __name__ == "__main__":
    main()
