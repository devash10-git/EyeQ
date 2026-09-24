"""IEEE figure: commit accuracy against entry rate, quantum vs classical.

Plots the operating curve each system reaches as gaze noise rises. Both axes
are things a user feels -- how often the keyboard is right, and how fast it
lets them type -- so a system that sits up and to the right is better in a way
that needs no further argument. Points are the five noise levels of the
benchmark sweep; sigma is the parameter that traces each curve.

Entry rate is derived, not stopwatch-measured, and the derivation is the
honest part. Every keystroke costs one dwell. A keystroke the system declines
to commit costs a second one, because the user then has to dwell again on a
suggestion chip to supply the character. So for N presented characters of
which P were prompted:

    time    = DWELL * (N + P)
    wpm     = (N / 5) / (time / 60)

This makes the prompt rate -- not just the accuracy -- visible as a speed cost,
which is the whole reason a threshold cannot be tuned for free. DWELL is the
deployed 693 ms from static/app.js.

Reads:
    eval/commit_log_quantum.csv             this repo, quantum selection rule
    eval/commit_log_classical.csv           this repo, log-linear control
    <classical>/commit_log_classical_raw.csv  the other implementation

Writes a two-line figure (the requested one) and a three-line figure that adds
the log-linear control. See the module note in the report for why the control
matters.

Run: python3 eval/figure_wpm_vs_accuracy.py [path/to/commit_log_classical_raw.csv]
"""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

DWELL_S = 0.693          # static/app.js DWELL_MS
PRESENTED = 599          # letters per trial in the 60-sentence protocol
SIGMAS = [30, 40, 50, 60, 75]

BLUE, YELLOW, GREY = "#1f5fa9", "#e8b52e", "#8a8a8a"


def per_trial(path: str) -> Dict[int, List[dict]]:
    by: Dict[int, List[dict]] = defaultdict(list)
    for r in csv.DictReader(open(path)):
        by[int(r["trial"])].append(r)
    return by


def curve(by: Dict[int, List[dict]]) -> Tuple[List[float], List[float]]:
    """(wpm, accuracy%) per trial, ordered by trial index."""
    xs, ys = [], []
    for t in sorted(by):
        rows = by[t]
        commits = len(rows)
        prompts = PRESENTED - commits
        mins = DWELL_S * (PRESENTED + prompts) / 60.0
        xs.append((PRESENTED / 5.0) / mins)
        ys.append(100.0 * sum(int(r["correct"]) for r in rows) / commits)
    return xs, ys


def style(ax) -> None:
    ax.set_xlabel("Entry rate (words per minute)")
    ax.set_ylabel("Commit accuracy (\\%)" if plt.rcParams["text.usetex"]
                  else "Commit accuracy (%)")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.margins(x=0.13, y=0.11)          # room for the end-point sigma labels
    ax.tick_params(labelsize=7)
    for s in ax.spines.values():
        s.set_linewidth(0.6)


def annotate(ax, xs, ys, colour, above: bool) -> None:
    """Label the end points with their noise level, so sigma is readable.

    The two curves converge at low sigma, so the labels are pushed apart
    vertically -- one series above its point, the other below -- rather than
    both defaulting to the same side and overlapping.
    """
    dy = 7 if above else -11
    for i, ha in ((0, "right"), (len(xs) - 1, "left")):
        ax.annotate("$\\sigma$=%d px" % SIGMAS[i], (xs[i], ys[i]),
                    textcoords="offset points",
                    xytext=(-6 if ha == "right" else 6, dy),
                    ha=ha, fontsize=6, color=colour)


def main() -> None:
    cls_path = (sys.argv[1] if len(sys.argv) > 1 else
                "/Users/devash10/Documents/duplicated2/SUMMERINTERNPROJECT/"
                "commit_log_classical_raw.csv")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "figure.dpi": 300, "savefig.dpi": 300,
    })

    qx, qy = curve(per_trial(os.path.join(HERE, "commit_log_quantum.csv")))
    cx, cy = curve(per_trial(cls_path))
    lx, ly = curve(per_trial(os.path.join(HERE, "commit_log_classical.csv")))

    # ---- requested figure: two lines ----
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.plot(cx, cy, "-s", color=YELLOW, lw=1.4, ms=4, label="Classical")
    ax.plot(qx, qy, "-o", color=BLUE, lw=1.4, ms=4, label="Quantum-inspired")
    annotate(ax, cx, cy, YELLOW, above=False)
    annotate(ax, qx, qy, BLUE, above=True)
    style(ax)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout(pad=0.3)
    out1 = os.path.join(HERE, "fig_wpm_vs_accuracy.png")
    fig.savefig(out1, bbox_inches="tight")
    fig.savefig(out1.replace(".png", ".pdf"), bbox_inches="tight")

    # ---- same figure plus the log-linear control ----
    fig2, ax2 = plt.subplots(figsize=(3.5, 2.7))
    ax2.plot(cx, cy, "-s", color=YELLOW, lw=1.4, ms=4, label="Classical (other impl.)")
    ax2.plot(qx, qy, "-o", color=BLUE, lw=1.4, ms=4, label="Quantum-inspired")
    ax2.plot(lx, ly, "--^", color=GREY, lw=1.2, ms=4,
             label="Log-linear control (same impl.)")
    style(ax2)
    ax2.legend(loc="lower right", frameon=False)
    fig2.tight_layout(pad=0.3)
    out2 = os.path.join(HERE, "fig_wpm_vs_accuracy_with_control.png")
    fig2.savefig(out2, bbox_inches="tight")
    fig2.savefig(out2.replace(".png", ".pdf"), bbox_inches="tight")

    print("  %-6s %-22s %-22s %s" % ("sigma", "classical", "quantum", "log-linear control"))
    print("  " + "-" * 74)
    for i, s in enumerate(SIGMAS):
        print("  %-6d %7.2f%% @ %5.2f wpm  %7.2f%% @ %5.2f wpm  %7.2f%% @ %5.2f wpm"
              % (s, cy[i], cx[i], qy[i], qx[i], ly[i], lx[i]))
    print("\nwrote %s (+.pdf)" % os.path.basename(out1))
    print("wrote %s (+.pdf)" % os.path.basename(out2))


if __name__ == "__main__":
    main()
