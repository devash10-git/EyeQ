"""Per-keystroke commit logging for the quantum-inspired selection rule.

This is an **evaluation layer only**. It does not contain a selection rule, a
gaze model, a calibration, a language model or a threshold of its own. Every
one of those is imported from the modules that already implement them:

    gaze estimate      eval.selection_rule_bench.gaze_posterior  (exact 2-D
                       Gaussian integral over key rectangles, the same
                       computation static/app.js::keyPosterior runs)
    selection rule     eval.selection_rule_bench.fuse -> either
                       quantum.conditional_selection.select   (quantum arm)
                       quantum.conditional_selection.loglinear (classical arm)
    language model     letters.LetterModel, built on HybridDecoder's unigrams
    thresholds         THETA / LM_BETA / LM_FLOOR from selection_rule_bench,
                       which takes them from static/app.js
    key geometry       selection_rule_bench.CENTRES / RECTS

The typing loop below is a line-for-line copy of
``selection_rule_bench.run_keystrokes`` with a ``yield`` added. Same RNG, same
call order, same prefix-advance rule. Nothing about the model's behaviour
changes because this file is running it.

Why one file serves both arms
-----------------------------
The requirement is that the quantum log be comparable with the classical log
under *exactly* the same protocol. The strongest way to guarantee that is not
to write two scripts carefully — it is to write one and switch a single
argument, so the sentences, the noise draws, the geometry, the threshold and
the logging arithmetic are literally the same code. ``--model`` picks the arm:

    quantum    phase_scale = 0.40   (the value tuned on the *train* split in
                                     eval/selection_rule_results.json)
    classical  phase_scale = 0.00   which makes select() reduce exactly to the
                                     deployed log-linear rule, and fuse() call
                                     loglinear() directly

Both arms are seeded identically, so keystroke *k* of trial *t* sees the same
gaze draw in both logs and the CSVs are row-comparable.

What counts as a committed keystroke
------------------------------------
The keyboard commits when the top fused posterior reaches ``THETA`` and prompts
otherwise. Only commits are written to the CSV, which is what makes the metric
*commit* accuracy rather than character accuracy: a prompt always yields the
right character, so counting prompts as correct would let any arm reach 100% by
refusing to decide. This is the same commit definition
``selection_rule_bench.curve()`` uses, including its treatment of the
end-of-word symbol ``$``: if ``$`` clears the threshold mid-word it is a commit
and it is wrong, because the ground truth there is a letter.

Run:
    python3 eval/commit_accuracy_log.py                  # quantum arm
    python3 eval/commit_accuracy_log.py --model classical
    python3 eval/commit_accuracy_log.py --both           # both, then compare
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from decoder import HybridDecoder                              # noqa: E402
from letters import LetterModel                                # noqa: E402
from quantum.conditional_selection import END                  # noqa: E402
from tune import VOCAB, load_sentences                         # noqa: E402

# The protocol itself. Imported, never redefined — if the deployed constants
# move, this logger moves with them.
from eval.selection_rule_bench import (                        # noqa: E402
    CENTRES, LM_BETA, LM_FLOOR, N_HYP, SIGMAS, THETA,
    fuse, gaze_posterior,
)

HERE = os.path.dirname(os.path.abspath(__file__))

# The tuned phase scale, chosen on the train split in selection_rule_bench and
# recorded in eval/selection_rule_results.json. Read from that file so the two
# can never drift apart; the literal is only a fallback.
PHASE_SCALE_DEFAULT = 0.40

# The held-out split selection_rule_bench reports on. Same seed, same skip, so
# both models are scored on the same sentences in the same order.
TEST_N, TEST_SEED, TEST_SKIP = 60, 4, 2060
NOISE_SEED = 7

CSV_COLUMNS = ["trial", "sentence", "keystroke_index", "ground_truth",
               "prediction", "correct", "cumulative_accuracy"]


def tuned_phase_scale() -> float:
    """The phase scale the bench settled on, or the fallback if absent."""
    path = os.path.join(HERE, "selection_rule_results.json")
    try:
        with open(path) as fh:
            return float(json.load(fh)["phase_scale"])
    except (OSError, KeyError, ValueError, TypeError):
        return PHASE_SCALE_DEFAULT


# ==========================================================================
# The typing loop — selection_rule_bench.run_keystrokes, instrumented
# ==========================================================================

def typing_trial(lm: LetterModel, pairs: Sequence[Tuple[str, str]],
                 sigma: float, phase_scale: float, amplify_cap: int,
                 stage: str = "pre",
                 seed: int = NOISE_SEED) -> Iterator[Dict[str, object]]:
    """Type every sentence, yielding one record per keystroke decision.

    Identical to ``selection_rule_bench.run_keystrokes`` except that it yields
    instead of accumulating. The yield sits *after* the decision and *before*
    the prefix advance, so consuming this generator cannot influence what the
    model does next.
    """
    rng = np.random.default_rng(seed)          # same seed => same noise per arm

    for si, (_code, en) in enumerate(pairs):
        for word in en.split():
            typed = ""
            for ch in word:
                if ch not in CENTRES:
                    continue
                cx, cy = CENTRES[ch]
                pg = gaze_posterior(cx + rng.normal(0, sigma),
                                    cy + rng.normal(0, sigma), sigma)
                if not pg:
                    continue
                prior = lm.letter_prior(typed)
                comps = (lm.completions(typed, n=N_HYP)
                         if (phase_scale > 0.0 or amplify_cap > 0) else [])
                post = fuse(pg, prior, typed, comps, phase_scale,
                            amplify_cap, stage)

                best = max(post, key=post.get)
                top_p = post[best]

                yield {"sentence_index": si, "sentence": en,
                       "ground_truth": ch, "prediction": best,
                       "top_p": float(top_p),
                       "committed": bool(top_p >= THETA)}

                # the live system would prompt below THETA and get the right
                # character, so the prefix advances correctly either way
                typed += ch if (top_p < THETA or best == END) else best


# ==========================================================================
# Live chart
# ==========================================================================

class AccuracyChart:
    """Cumulative commit accuracy against keystroke number, updated per commit.

    Uses a GUI backend when one is available so the line really does redraw on
    every committed character. With no display attached — CI, ssh, a headless
    run — it falls back to Agg and flushes a PNG every ``png_every`` commits.
    The caller is told which mode it got rather than being left to assume.
    """

    def __init__(self, title: str, out_png: str, live: bool = True,
                 png_every: int = 50, full_scale: bool = False) -> None:
        self.out_png = out_png
        self.png_every = max(1, png_every)
        self.full_scale = full_scale
        self._ymin = 100.0
        self.interactive = False
        self.plt = None
        self.series: Dict[int, Tuple[List[int], List[float]]] = {}
        self.lines: Dict[int, object] = {}
        self._since_flush = 0

        import matplotlib
        if live:
            for backend in ("macosx", "qtagg", "tkagg"):
                try:
                    matplotlib.use(backend, force=True)
                    import matplotlib.pyplot as plt
                    fig = plt.figure()
                    plt.close(fig)
                    self.interactive = True
                    break
                except Exception:
                    continue
        if not self.interactive:
            matplotlib.use("Agg", force=True)

        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig, self.ax = plt.subplots(figsize=(9.0, 5.0))
        self.ax.set_xlabel("Keystroke Number")
        self.ax.set_ylabel("Cumulative Commit Accuracy (%)")
        self.ax.set_title(title)
        self.ax.grid(True, alpha=0.3)
        # A fixed 0-100 axis renders this metric as a flat line at the top: the
        # whole signal lives in the last few percent. Autoscale the floor to the
        # data and keep 100 in frame, unless --full-scale asks for 0-100.
        self.ax.set_ylim(0, 101) if full_scale else self.ax.set_ylim(90, 100.5)
        if self.interactive:
            plt.ion()
            self.fig.show()

    def start_trial(self, trial: int, label: str) -> None:
        self.series[trial] = ([], [])
        (line,) = self.ax.plot([], [], lw=1.6, label=label)
        self.lines[trial] = line
        self.ax.legend(loc="lower right", fontsize=8)

    def update(self, trial: int, k: int, acc_pct: float) -> None:
        xs, ys = self.series[trial]
        xs.append(k)
        ys.append(acc_pct)
        self.lines[trial].set_data(xs, ys)
        self.ax.relim()
        self.ax.autoscale_view(scalex=True, scaley=False)
        if not self.full_scale and acc_pct < self._ymin:
            self._ymin = acc_pct
            self.ax.set_ylim(max(0.0, min(90.0, self._ymin - 2.0)), 100.5)
        if self.interactive:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        else:
            self._since_flush += 1
            if self._since_flush >= self.png_every:
                self._since_flush = 0
                self.fig.savefig(self.out_png, dpi=130, bbox_inches="tight")

    def finish(self) -> None:
        self.fig.savefig(self.out_png, dpi=130, bbox_inches="tight")
        if self.interactive:
            self.plt.ioff()


# ==========================================================================
# Driver
# ==========================================================================

def run(model: str, sigmas: Sequence[float], repeats: int, n_sentences: int,
        live: bool, png_every: int, out_csv: str, out_json: str,
        out_png: str, full_scale: bool = False) -> Dict[str, object]:
    phase_scale = tuned_phase_scale() if model == "quantum" else 0.0
    amplify_cap = 0                     # off in the deployed rule and the bench

    dec = HybridDecoder(vocab_path=VOCAB, model_dir=None, num_clusters=9)
    lm = LetterModel(dec.unigrams)
    pairs = load_sentences(n_sentences, seed=TEST_SEED, skip=TEST_SKIP)

    print("=" * 78)
    print("COMMIT-ACCURACY LOG — %s model" % model.upper())
    print("=" * 78)
    print("selection rule : %s" %
          ("quantum conditional selection, phase_scale=%.2f" % phase_scale
           if phase_scale > 0 else "deployed log-linear fusion (phase_scale=0)"))
    print("thresholds     : THETA=%.2f  LM_BETA=%.2f  LM_FLOOR=%.0e  (static/app.js)"
          % (THETA, LM_BETA, LM_FLOOR))
    print("test sentences : %d, load_sentences(seed=%d, skip=%d) — held-out split"
          % (len(pairs), TEST_SEED, TEST_SKIP))
    print("gaze noise     : sigma in %s px, RNG seed %d"
          % (", ".join("%g" % s for s in sigmas), NOISE_SEED))
    print("vocabulary     : %d words" % len(dec.unigrams))

    chart = AccuracyChart("Cumulative Commit Accuracy — %s model" % model,
                          out_png, live=live, png_every=png_every,
                          full_scale=full_scale)
    print("chart          : %s"
          % ("live window, redrawn every committed character"
             if chart.interactive
             else "headless (Agg); %s flushed every %d commits"
                  % (os.path.basename(out_png), png_every)))
    print()

    conditions = [(s, r) for r in range(repeats) for s in sigmas]
    rows: List[Dict[str, object]] = []
    trials: List[Dict[str, object]] = []

    for t, (sigma, rep) in enumerate(conditions, start=1):
        label = "trial %d (sigma=%gpx)" % (t, sigma)
        chart.start_trial(t, label)
        seed = NOISE_SEED + rep      # identical across models, varied per repeat

        committed = correct = seen = 0
        t0 = time.time()
        for rec in typing_trial(lm, pairs, sigma, phase_scale, amplify_cap,
                                seed=seed):
            seen += 1
            if not rec["committed"]:
                continue
            committed += 1
            is_correct = int(rec["prediction"] == rec["ground_truth"])
            correct += is_correct
            cum = 100.0 * correct / committed
            rows.append({"trial": t,
                         "sentence": rec["sentence"],
                         "keystroke_index": committed,
                         "ground_truth": rec["ground_truth"],
                         "prediction": rec["prediction"],
                         "correct": is_correct,
                         "cumulative_accuracy": round(cum, 6)})
            chart.update(t, committed, cum)

        incorrect = committed - correct
        final = 100.0 * correct / committed if committed else float("nan")
        prompts = seen - committed
        trials.append({"trial": t, "sigma_px": sigma, "repeat": rep,
                       "noise_seed": seed,
                       "final_commit_accuracy_pct": final,
                       "total_committed": committed,
                       "correct": correct, "incorrect": incorrect,
                       "keystrokes_presented": seen,
                       "prompts": prompts,
                       "prompt_rate": prompts / seen if seen else float("nan"),
                       "seconds": round(time.time() - t0, 2)})

        print("-" * 78)
        print("TRIAL %d  (sigma = %g px, noise seed %d)" % (t, sigma, seed))
        print("-" * 78)
        print("  final commit accuracy   : %.4f %%" % final)
        print("  total committed keys    : %d" % committed)
        print("  correct keystrokes      : %d" % correct)
        print("  incorrect keystrokes    : %d" % incorrect)
        print("  (context: %d keystrokes presented, %d prompted below THETA,"
              " prompt rate %.1f%%)" % (seen, prompts, 100.0 * prompts / max(1, seen)))
        print()

    chart.finish()

    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    tot_c = sum(t["total_committed"] for t in trials)
    tot_ok = sum(t["correct"] for t in trials)
    summary = {
        "model": model,
        "selection_rule": ("quantum conditional selection"
                           if phase_scale > 0 else "log-linear fusion"),
        "phase_scale": phase_scale,
        "amplify_cap": amplify_cap,
        "theta": THETA, "lm_beta": LM_BETA, "lm_floor": LM_FLOOR,
        "n_hypotheses": N_HYP,
        "sentences": len(pairs),
        "sentence_split": {"n": n_sentences, "seed": TEST_SEED,
                           "skip": TEST_SKIP},
        "sigmas_px": list(sigmas), "repeats": repeats,
        "base_noise_seed": NOISE_SEED,
        "trials": trials,
        "pooled": {"total_committed": tot_c, "correct": tot_ok,
                   "incorrect": tot_c - tot_ok,
                   "commit_accuracy_pct": 100.0 * tot_ok / tot_c if tot_c else None},
        "csv": os.path.basename(out_csv),
        "csv_columns": CSV_COLUMNS,
    }
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)

    print("=" * 78)
    print("POOLED OVER %d TRIALS" % len(trials))
    print("=" * 78)
    print("  commit accuracy : %.4f %%" % summary["pooled"]["commit_accuracy_pct"])
    print("  committed / correct / incorrect : %d / %d / %d"
          % (tot_c, tot_ok, tot_c - tot_ok))
    print()
    print("wrote %s  (%d rows)" % (out_csv, len(rows)))
    print("wrote %s" % out_json)
    print("wrote %s" % out_png)
    return summary


def compare(a: Dict[str, object], b: Dict[str, object]) -> None:
    """Trial-by-trial table of the two logs, which share every draw."""
    print()
    print("=" * 78)
    print("QUANTUM vs CLASSICAL — identical sentences, identical gaze draws")
    print("=" * 78)
    print("  %-7s %9s %14s %14s %11s" %
          ("trial", "sigma", a["model"], b["model"], "delta pp"))
    print("  " + "-" * 60)
    for ta, tb in zip(a["trials"], b["trials"]):
        print("  %-7d %8gpx %13.4f%% %13.4f%% %+10.4f"
              % (ta["trial"], ta["sigma_px"],
                 ta["final_commit_accuracy_pct"], tb["final_commit_accuracy_pct"],
                 ta["final_commit_accuracy_pct"] - tb["final_commit_accuracy_pct"]))
    print("  " + "-" * 60)
    pa, pb = a["pooled"]["commit_accuracy_pct"], b["pooled"]["commit_accuracy_pct"]
    print("  %-7s %8s  %13.4f%% %13.4f%% %+10.4f"
          % ("POOLED", "", pa, pb, pa - pb))
    print("\n  Commit counts differ between arms because the threshold is fixed:")
    print("  a rule that is less certain commits less often. Compare the CSVs at")
    print("  matched prompt rate (eval/selection_rule_bench.py) before concluding")
    print("  anything from the accuracy column alone.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", choices=("quantum", "classical"), default="quantum")
    p.add_argument("--both", action="store_true",
                   help="run both arms on identical draws and compare")
    p.add_argument("--sigmas", type=float, nargs="+", default=list(SIGMAS),
                   help="gaze noise levels in px (default: the bench sweep)")
    p.add_argument("--repeats", type=int, default=1,
                   help="repetitions of the sigma set, each with a new noise seed")
    p.add_argument("--sentences", type=int, default=TEST_N)
    p.add_argument("--no-live", action="store_true",
                   help="force the headless Agg backend")
    p.add_argument("--full-scale", action="store_true",
                   help="force a 0-100%% y-axis instead of scaling to the data")
    p.add_argument("--png-every", type=int, default=50,
                   help="headless only: flush the PNG every N commits (1 = every)")
    p.add_argument("--outdir", default=HERE)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    models = ("quantum", "classical") if args.both else (args.model,)
    out: Dict[str, Dict[str, object]] = {}
    for m in models:
        out[m] = run(
            m, args.sigmas, args.repeats, args.sentences,
            live=not args.no_live, png_every=args.png_every,
            out_csv=os.path.join(args.outdir, "commit_log_%s.csv" % m),
            out_json=os.path.join(args.outdir, "commit_log_%s_summary.json" % m),
            out_png=os.path.join(args.outdir, "commit_accuracy_%s.png" % m),
            full_scale=args.full_scale)
        print()

    if args.both:
        compare(out["quantum"], out["classical"])


if __name__ == "__main__":
    main()
