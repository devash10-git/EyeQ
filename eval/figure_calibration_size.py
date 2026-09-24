"""IEEE figure: regressor accuracy against calibration size.

The single most important thing this plot shows is that the ranking DEPENDS on
the x-axis. Kernel ridge wins at 20 targets and loses at 9, so a benchmark run
only at leave-one-target-out would recommend replacing the deployed regressor
with one that is worse where it would actually run. The deployed calibration
size is marked for that reason.

Reads eval/calibration_size_benchmark.json.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "calibration_size_benchmark.json")))
sizes = d["sizes"]; dep = d["deployed_targets"]
arms = ["Ridge", "OLS", "KRR-RBF", "MLP"]
STYLE = {"Ridge":   ("#1f5fa9", "-o", 2.0),
         "OLS":     ("#5b9bd5", "--s", 1.3),
         "KRR-RBF": ("#e8b52e", "-^", 1.3),
         "MLP":     ("#c0863a", ":v", 1.3)}

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 8, "axes.labelsize": 8,
                     "figure.dpi": 300, "savefig.dpi": 300})
fig, ax = plt.subplots(figsize=(3.5, 2.7))

ax.axvline(dep, color="#999", lw=0.9, ls="-", zorder=0)

for a in arms:
    c, m, lw = STYLE[a]
    ys = [d["sweep_mean_deg"][str(n)][a] for n in sizes]
    ax.plot(sizes, ys, m, color=c, lw=lw, ms=3.6, label=a,
            zorder=3 if a == "Ridge" else 2)

ax.set_xlabel("Calibration targets used for training")
ax.set_ylabel("Mean gaze error (deg)")
ax.set_xticks(sizes)
ax.grid(True, alpha=0.3, lw=0.5); ax.set_axisbelow(True)
ax.tick_params(labelsize=7)
for s in ax.spines.values(): s.set_linewidth(0.6)
ax.legend(frameon=False, fontsize=6.8, loc="upper right", ncol=2)
# annotate AFTER plotting, or the y-limits are still the empty-axes defaults
# and the label lands off-canvas
ax.annotate("deployed calibration", (dep, ax.get_ylim()[0]), xytext=(4, 6),
            textcoords="offset points", fontsize=6.2, color="#555",
            rotation=90, va="bottom")
fig.tight_layout(pad=0.3)
out = os.path.join(HERE, "fig_calibration_size.png")
fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".png",".pdf"), bbox_inches="tight")
print("wrote fig_calibration_size.png (+.pdf)")

print("\nat the deployed %d targets (45 paired draws):" % dep)
for r in d["deployed"]:
    print("  %-9s %.3f +-%.3f deg  %s" % (r["model"], r["deg"], r["sd"],
          "baseline" if r["model"]=="Ridge" else
          "%+.1f%% vs ridge, p=%.4f" % (r["vs_ridge_pct"], r["paired_p"])))
