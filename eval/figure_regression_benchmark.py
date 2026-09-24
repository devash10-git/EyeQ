"""IEEE figure: the four regressors on accuracy and on cost.

Two panels because the two axes disagree, and collapsing them would hide the
actual finding. Kernel ridge has the lowest gaze error; ridge is four orders of
magnitude cheaper to fit and to evaluate, and the accuracy gap does not survive
into typing outcome. A single-bar "best model" chart cannot say that.

Reads eval/regression_benchmark.json. Run after eval/regression_benchmark.py.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "regression_benchmark.json")))
arms = d["arms"]
order = ["Ridge (deployed)", "Ordinary least squares",
         "Kernel ridge (RBF)", "MLP (16-8)"]
short = ["Ridge", "OLS", "KRR-RBF", "MLP"]
# linear arms share one colour, non-linear another, so family is readable
COL = ["#1f5fa9", "#5b9bd5", "#e8b52e", "#f0d080"]

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 8, "axes.labelsize": 8,
                     "figure.dpi": 300, "savefig.dpi": 300})

fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.5, 4.3))
x = np.arange(len(order))

deg = [arms[k]["deg"] for k in order]
b = a1.bar(x, deg, color=COL, width=0.62, edgecolor="black", linewidth=0.4)
a1.set_ylabel("Mean gaze error (deg)")
a1.set_ylim(0, max(deg) * 1.22)
for xi, v in zip(x, deg):
    a1.text(xi, v + max(deg) * 0.03, "%.3f" % v, ha="center", fontsize=6.5)
a1.set_xticks(x); a1.set_xticklabels([]); a1.grid(axis="y", alpha=0.3, lw=0.5)
a1.set_axisbelow(True)

fit = [arms[k]["fit_ms"] for k in order]
inf = [arms[k]["predict_us"] / 1000.0 for k in order]     # ms per sample
w = 0.34
a2.bar(x - w/2, fit, w, color="#1f5fa9", edgecolor="black", lw=0.4, label="Fit")
a2.bar(x + w/2, inf, w, color="#e8b52e", edgecolor="black", lw=0.4,
       label="Inference / sample")
a2.set_yscale("log")
a2.set_ylabel("Time (ms, log scale)")
a2.set_xticks(x); a2.set_xticklabels(short, fontsize=7.5)
a2.grid(axis="y", alpha=0.3, lw=0.5); a2.set_axisbelow(True)
a2.legend(frameon=False, fontsize=6.5, loc="upper left")

for ax in (a1, a2):
    ax.tick_params(labelsize=7)
    for sp in ax.spines.values(): sp.set_linewidth(0.6)

fig.tight_layout(pad=0.3)
out = os.path.join(HERE, "fig_regression_benchmark.png")
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print("wrote fig_regression_benchmark.png (+.pdf)")
for k, s in zip(order, short):
    t = arms[k]["typing"]
    print("  %-8s %.3f deg | typing %.2f%% +-%.2f | fit %.2f ms | inf %.2f us | hp %d"
          % (s, arms[k]["deg"], t["commit_accuracy_pct"], t["commit_accuracy_sd"],
             arms[k]["fit_ms"], arms[k]["predict_us"], arms[k]["hyperparams_searched"]))
