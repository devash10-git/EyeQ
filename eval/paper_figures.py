"""The four figures the paper reserves space for, built from the real data.

``report/paper/gaze_paper.tex`` holds four \\figholder slots. Three of them are
plots of measurements this repository can reproduce, and one is a schematic.
This script produces all four as vector PDFs sized for an IEEEtran column.

    fig_geometry    Fig. 2  schematic: eyelid- vs canthus-relative normalisation
    fig_ablation    Fig. 3  held-out error by feature set and session
    fig_budget      Fig. 4  error decomposition against head roll
    fig_selection   Fig. 5  accuracy/prompt-rate frontier, both selection rules

Nothing is hard-coded from the paper's tables. Every number is recomputed here
from the recorded sessions using the same helpers the benchmarks use, so if a
figure disagrees with a table in the text, the table is what is stale.

Figures 3 and 4 fit on the recorded sessions directly. Figure 5 replays the
typing benchmark, which takes a couple of minutes. Figure 2 draws geometry and
needs no data.

Run: python3 eval/paper_figures.py [geometry|ablation|budget|selection|all]
"""

from __future__ import annotations

import json
import math
import os
import sys
import warnings
from typing import Dict, List

import numpy as np

warnings.filterwarnings("ignore")
np.seterr(all="ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
COL = 3.5                      # IEEEtran single-column width, inches

BLUE, YELLOW, GREY, DARK = "#1f5fa9", "#e8b52e", "#8a8a8a", "#333333"
LIGHTBLUE, BROWN = "#5b9bd5", "#c0863a"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.8,
    "figure.dpi": 300, "savefig.dpi": 300,
})


def save(fig, name: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, "%s.%s" % (name, ext)), bbox_inches="tight")
    plt.close(fig)
    print("  wrote figures/%s.pdf (+.png)" % name)


def tidy(ax) -> None:
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7)
    for s in ax.spines.values():
        s.set_linewidth(0.6)


# ==========================================================================
# Fig. 2 — normalisation geometry (schematic)
# ==========================================================================

def eye_outline(ax, cx, cy, w=1.0, ap=0.42, colour=DARK, lw=1.1, ls="-"):
    """A lens-shaped eye: two arcs meeting at the canthi."""
    t = np.linspace(0, 1, 120)
    x = cx + (t - 0.5) * w
    bulge = np.sin(np.pi * t)
    ax.plot(x, cy + ap * bulge, ls, color=colour, lw=lw)
    ax.plot(x, cy - ap * bulge, ls, color=colour, lw=lw)


def fig_geometry() -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL, 1.72))

    # ---- left: eyelid-relative, and why it moves ----
    cx, cy = 0.0, 0.0
    eye_outline(a1, cx, cy, 1.6, 0.46)
    eye_outline(a1, cx, cy, 1.6, 0.22, colour=BROWN, lw=0.9, ls="--")
    a1.add_patch(plt.Circle((0.12, 0.02), 0.17, fill=False, color=BLUE, lw=1.1))
    a1.plot(0.12, 0.02, "o", color=BLUE, ms=2.2)
    # the aperture that is used as the denominator
    a1.annotate("", xy=(-0.30, 0.44), xytext=(-0.30, -0.44),
                arrowprops=dict(arrowstyle="<->", color=DARK, lw=0.8))
    a1.text(-0.38, 0.0, "$b_y-t_y$", fontsize=6.6, ha="right", va="center")
    a1.annotate("", xy=(0.50, 0.21), xytext=(0.50, -0.21),
                arrowprops=dict(arrowstyle="<->", color=BROWN, lw=0.8))
    a1.text(0.56, 0.30, "squint", fontsize=6.2, color=BROWN, va="center")
    a1.set_title("eyelid-relative", fontsize=7.4, pad=3)
    a1.text(0.0, -0.78, "denominator contracts", fontsize=6.2, ha="center",
            color=BROWN)

    # ---- right: canthus-relative, rigid ----
    eye_outline(a2, cx, cy, 1.6, 0.46)
    ix, iy = 0.28, 0.06
    a2.add_patch(plt.Circle((ix, iy), 0.15, fill=False, color=BLUE, lw=1.1))
    inner, outer = (-0.8, 0.0), (0.8, 0.0)
    for p in (inner, outer):
        a2.plot(*p, "o", color=YELLOW, ms=3.4, markeredgecolor=DARK,
                markeredgewidth=0.4, zorder=5)
    # the inter-canthal baseline w, and the offset d resolved on it
    a2.annotate("", xy=outer, xytext=inner,
                arrowprops=dict(arrowstyle="<->", color=YELLOW, lw=0.9))
    a2.text(0.0, -0.34, "$w$", fontsize=6.8, ha="center", color="#a07c14")
    mid = ((inner[0] + outer[0]) / 2, (inner[1] + outer[1]) / 2)
    a2.add_patch(FancyArrowPatch(mid, (ix, iy), arrowstyle="-|>",
                                 mutation_scale=7, color=BLUE, lw=1.0))
    a2.text(ix - 0.20, iy + 0.16, r"$\mathbf{d}$", fontsize=7, color=BLUE)
    a2.annotate("", xy=(-0.18, 0.0), xytext=(-0.50, 0.0),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=0.7))
    a2.text(-0.36, -0.20, r"$\mathbf{u}$", fontsize=6.6)
    a2.annotate("", xy=(-0.50, 0.30), xytext=(-0.50, 0.0),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=0.7))
    a2.text(-0.48, 0.30, r"$\mathbf{u}^{\perp}$", fontsize=6.6)
    a2.set_title("canthus-relative", fontsize=7.4, pad=3)
    a2.text(0.0, -0.78, "baseline is rigid", fontsize=6.2, ha="center",
            color="#1f5fa9")

    for a in (a1, a2):
        a.set_xlim(-1.05, 1.05)
        a.set_ylim(-0.92, 0.72)
        a.set_aspect("equal")
        a.axis("off")
    fig.tight_layout(pad=0.2)
    save(fig, "fig_geometry")


# ==========================================================================
# Shared: leave-one-target-out predictions
# ==========================================================================

def loto(sess, kind: str) -> np.ndarray:
    from eval.feature_ablation import columns
    from eval.gaze_bench import fit_ridge
    X, T = columns(sess, kind), sess.T
    W, H = sess.meta["screen"]["w_px"], sess.meta["screen"]["h_px"]
    pred = np.empty_like(T)
    for g in np.unique(sess.groups):
        te = sess.groups == g
        pred[te] = X[te] @ fit_ridge(X[~te], T[~te])
    pred[:, 0] = np.clip(pred[:, 0], 0, W)
    pred[:, 1] = np.clip(pred[:, 1], 0, H)
    return pred


def sessions():
    from eval.gaze_bench import load_sessions, session_quality
    return [s for s in load_sessions() if session_quality(s)["usable"]]


# ==========================================================================
# Fig. 3 — feature ablation by session
# ==========================================================================

def fig_ablation() -> None:
    """Feature ablation aggregated by participant, not by session.

    Frames within a session are not independent and sessions within a person
    are not either, so the unit of analysis is the participant: each one
    contributes the mean over their sessions, and the bar is the mean over
    participants with a standard-deviation whisker. Individual participants are
    overlaid as points, because a bar chart that hides 9 of 9 improving is
    hiding the strongest thing in the figure.
    """
    import json as _json, glob as _glob
    from collections import defaultdict
    SETS = ["deployed", "radius", "pose", "canthus", "optics_pose"]
    LABEL = {"deployed": "deployed\n(8)", "radius": "+radius\n(11)",
             "pose": "+pose\n(14)", "canthus": "+canthus\n(12)",
             "optics_pose": "+all\n(21)"}
    S = sessions()
    from eval.feature_ablation import columns as _cols
    by = defaultdict(lambda: defaultdict(list))
    for sess in S:
        raw = _json.load(open([f for f in _glob.glob("eval/sessions/*.json")
                               if sess.label.split("/")[-1][:15] in f][0]))
        pid = raw["participant"]
        for k in SETS:
            pr = loto(sess, k)
            by[pid][k].append(float(np.linalg.norm(pr - sess.T, axis=1).mean())
                              / sess.px_per_degree)
    pids = sorted(by)
    vals = {k: np.array([np.mean(by[p][k]) for p in pids]) for k in SETS}
    for k in SETS:
        print("    %-12s %.3f +- %.3f deg" % (k, vals[k].mean(), vals[k].std(ddof=1)))

    fig, ax = plt.subplots(figsize=(COL, 2.5))
    x = np.arange(len(SETS))
    m = [vals[k].mean() for k in SETS]
    sd = [vals[k].std(ddof=1) for k in SETS]
    cols = [BLUE if k == "canthus" else LIGHTBLUE for k in SETS]
    ax.bar(x, m, 0.6, yerr=sd, capsize=2.5, color=cols, edgecolor="black",
           lw=0.4, error_kw=dict(lw=0.7))
    for i, k in enumerate(SETS):
        jitter = (np.arange(len(pids)) - (len(pids) - 1) / 2) * 0.035
        ax.plot(x[i] + jitter, vals[k], "o", ms=1.9, color=DARK, alpha=0.65,
                zorder=3, mew=0)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[k] for k in SETS], fontsize=6.6)
    for lbl, k in zip(ax.get_xticklabels(), SETS):
        if k == "canthus":
            lbl.set_fontweight("bold"); lbl.set_color(BLUE)
    ax.set_ylabel("Held-out error (deg)")
    ax.set_ylim(0, max(np.array(m) + np.array(sd)) * 1.24)
    # State the sample on the figure itself. A bar chart of per-participant
    # means is indistinguishable at a glance from one of per-session means, and
    # the difference decides whether the paired test in the text is valid.
    nsess = sum(len(by[p][SETS[0]]) for p in pids)
    ax.text(0.015, 0.965,
            "$n$ = %d participants, %d sessions\neach point is one participant"
            % (len(pids), nsess),
            transform=ax.transAxes, ha="left", va="top", fontsize=6.2,
            color=DARK)
    tidy(ax)
    fig.tight_layout(pad=0.3)
    save(fig, "fig_ablation")


# ==========================================================================
# Fig. 4 — error budget against head roll
# ==========================================================================

def fig_budget() -> None:
    """Split each fixation's error into per-frame scatter and a fixed offset.

    Within-fixation is the spread of the predictions around their own mean --
    noise the dwell centroid can average away. Between-fixation is the distance
    from that mean to the true target -- a bias no amount of averaging removes.
    Which one dominates is what changes with roll, and it is the reason a larger
    target helps a tilted head more than a better regressor would.
    """
    BINS = [(0, 2), (2, 5), (5, 10), (10, 90)]
    S = sessions()
    rows = []
    for lo, hi in BINS:
        w2, b2, tot, n = [], [], [], 0
        for s in S:
            pred = loto(s, "canthus")
            roll = np.degrees(np.abs([e.get("roll", 0.0) for e in s.extras]))
            ppd = s.px_per_degree
            for g in np.unique(s.groups):
                m = s.groups == g
                if not (lo <= float(np.mean(roll[m])) < hi):
                    continue
                P, T = pred[m], s.T[m]
                mu = P.mean(axis=0)
                w2.append(float((np.linalg.norm(P - mu, axis=1) ** 2).mean()))
                b2.append(float(np.linalg.norm(mu - T[0]) ** 2))
                tot.append(float(np.linalg.norm(P - T, axis=1).mean() / ppd))
                n += 1
        if not n:
            continue
        ppd = float(np.mean([s.px_per_degree for s in S]))
        within = math.sqrt(np.mean(w2)) / ppd
        between = math.sqrt(np.mean(b2)) / ppd
        rows.append({"bin": "%d-%d" % (lo, hi) if hi < 90 else "$\\geq$10",
                     "total": float(np.mean(tot)), "within": within,
                     "between": between,
                     "noise_share": 100.0 * within ** 2 / (within ** 2 + between ** 2),
                     "n": n})
        print("    |roll| %-7s n=%3d  total %.2f  within %.2f  between %.2f"
              "  noise %.1f%%" % (rows[-1]["bin"], n, rows[-1]["total"],
                                  within, between, rows[-1]["noise_share"]))

    fig, ax = plt.subplots(figsize=(COL, 2.35))
    x = np.arange(len(rows))
    wi = [r["within"] for r in rows]
    be = [r["between"] for r in rows]
    ax.bar(x, be, 0.55, label="Between-fixation (bias)", color=BLUE,
           edgecolor="black", lw=0.35)
    ax.bar(x, wi, 0.55, bottom=be, label="Within-fixation (noise)",
           color=YELLOW, edgecolor="black", lw=0.35)
    ax.set_xticks(x)
    # n on the tick label, not hidden in a caption: the upper bins hold 2-5
    # fixations from a single session and a reader has to see that to weigh
    # the trend correctly.
    ax.set_xticklabels(["%s\n$n{=}%d$" % (r["bin"], r["n"]) for r in rows],
                       fontsize=6.6)
    ax.set_xlabel(r"$|\mathrm{roll}|$ (deg)")
    ax.set_ylabel("Error component (deg)")
    tidy(ax)

    ax2 = ax.twinx()
    ns = [r["noise_share"] for r in rows]
    ax2.plot(x, ns, "-o", color=DARK, lw=1.3, ms=3.6, label="Noise share")
    ax2.axhline(50, color=GREY, lw=0.8, ls=":")
    ax2.set_ylabel("Noise share (%)", fontsize=8)
    ax2.set_ylim(0, 100)
    ax2.tick_params(labelsize=7)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left", fontsize=6.2)
    fig.tight_layout(pad=0.3)
    save(fig, "fig_budget")
    with open(os.path.join(OUT, "fig_budget_data.json"), "w") as fh:
        json.dump(rows, fh, indent=2)


# ==========================================================================
# Fig. 5 — the accuracy / prompt-rate frontier
# ==========================================================================

def fig_selection() -> None:
    """Both rules across the whole threshold sweep, at two noise levels.

    A single operating point cannot show that two rules are equivalent -- either
    could be ahead at one threshold by chance. The whole frontier can, and here
    the curves lie on top of each other, which is the paper's claim.

    The coherence panel is separate rather than an inset: an inset large enough
    to read covered the sigma=40 curve, and the thing being covered is the
    result.

    Curves are cached, because replaying the typing benchmark costs minutes and
    the layout usually needs more than one pass. Delete the cache or pass
    --force to recompute.
    """
    cache = os.path.join(OUT, "fig_selection_data.json")
    styles = [(40.0, 0.0, BLUE, "-", "log-linear, $\\sigma$=40"),
              (40.0, 0.4, YELLOW, "--", "rule, $\\sigma$=40"),
              (60.0, 0.0, LIGHTBLUE, "-", "log-linear, $\\sigma$=60"),
              (60.0, 0.4, BROWN, "--", "rule, $\\sigma$=60")]

    if os.path.exists(cache) and "--force" not in sys.argv:
        curves = json.load(open(cache))
        print("    reusing cached curves (--force to recompute)")
    else:
        from decoder import HybridDecoder
        from letters import LetterModel
        from tune import VOCAB, load_sentences
        from eval.selection_rule_bench import curve, run_keystrokes
        lm = LetterModel(HybridDecoder(vocab_path=VOCAB, model_dir=None,
                                       num_clusters=9).unigrams)
        test = load_sentences(60, seed=4, skip=2060)
        curves = {}
        for sg, scale, _c, _ls, _lbl in styles:
            cv = [r for r in curve(run_keystrokes(lm, test, sg, scale, 0))
                  if 0.0 < r["prompt_rate"] < 0.9]
            cv.sort(key=lambda r: r["prompt_rate"])
            curves["%g_%.1f" % (sg, scale)] = cv
            print("    sigma=%g scale=%.1f: %d threshold points" % (sg, scale, len(cv)))
        os.makedirs(OUT, exist_ok=True)
        json.dump(curves, open(cache, "w"), indent=2)

    ps = json.load(open(os.path.join(HERE, "selection_rule_results.json")))
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(COL, 3.1), gridspec_kw={"height_ratios": [3, 1.15]})

    for sg, scale, c, ls, lbl in styles:
        cv = curves["%g_%.1f" % (sg, scale)]
        ax.plot([100 * r["prompt_rate"] for r in cv],
                [100 * r["commit_acc"] for r in cv], ls, color=c, lw=1.5,
                label=lbl)
    ax.set_xlabel("Prompt rate (%)", labelpad=1)
    ax.set_ylabel("Commit accuracy (%)")
    ax.legend(frameon=False, loc="lower right", ncol=2, fontsize=6.2)
    tidy(ax)

    disc = ps["discrimination"]
    ax2.plot([d["scale"] for d in disc], [d["separation"] for d in disc],
             "-o", color=DARK, lw=1.2, ms=3.0)
    ax2.axhline(0, color="#c33", lw=0.8, ls=":")
    ax2.set_xlabel("Phase scale", labelpad=1)
    ax2.set_ylabel("Coherence\nseparation", fontsize=7)
    tidy(ax2)
    fig.tight_layout(pad=0.3)
    save(fig, "fig_selection")


def main() -> None:
    which = (sys.argv[1] if len(sys.argv) > 1 else "all").lower()
    jobs = [("geometry", fig_geometry), ("ablation", fig_ablation),
            ("budget", fig_budget), ("selection", fig_selection)]
    for name, fn in jobs:
        if which in ("all", name):
            print("Fig. %s" % name)
            fn()


if __name__ == "__main__":
    main()
