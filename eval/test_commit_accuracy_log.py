"""Guard: the logging layer must not perturb the model it is logging.

``eval/commit_accuracy_log.py`` re-types the test sentences so it can emit a
record per keystroke. That is only legitimate if the instrumented loop makes
*exactly* the decisions the untouched benchmark makes. This test asserts it
directly: for both arms and across the noise sweep, the logger's per-keystroke
top posterior and its correct/incorrect verdict must equal
``selection_rule_bench.run_keystrokes``'s arrays element for element.

A float comparison is used rather than a tolerance on purpose. Same seed, same
call order, same arithmetic — the answer is not merely close, it is the same
number. If it ever stops being the same number, something in the model path
changed and the log is no longer the benchmark's protocol.

It also checks the CSV arithmetic: cumulative accuracy must be the running
mean of the correct column, and keystroke_index must count commits from 1.

Run: ``python3 eval/test_commit_accuracy_log.py``
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from decoder import HybridDecoder                                  # noqa: E402
from letters import LetterModel                                    # noqa: E402
from tune import VOCAB, load_sentences                             # noqa: E402

from eval.commit_accuracy_log import (                             # noqa: E402
    CSV_COLUMNS, run, tuned_phase_scale, typing_trial,
)
from eval.selection_rule_bench import SIGMAS, run_keystrokes       # noqa: E402

N_SENTENCES = 20


def check_matches_bench(lm) -> None:
    pairs = load_sentences(N_SENTENCES, seed=4, skip=2060)
    scale = tuned_phase_scale()
    arms = (("classical", 0.0), ("quantum", scale))

    print("  %-12s %8s %7s %12s %12s" %
          ("arm", "sigma", "n", "top_p", "decisions"))
    print("  " + "-" * 56)
    for name, ps in arms:
        for sigma in SIGMAS:
            ref = run_keystrokes(lm, pairs, sigma, ps, 0)
            mine = list(typing_trial(lm, pairs, sigma, ps, 0))
            assert len(mine) == ref["n"], (
                "%s sigma=%g: logger saw %d keystrokes, bench saw %d"
                % (name, sigma, len(mine), ref["n"]))
            tp = np.array([r["top_p"] for r in mine])
            ok = np.array([r["prediction"] == r["ground_truth"] for r in mine])
            assert np.array_equal(tp, ref["top_p"]), \
                "%s sigma=%g: posteriors differ from the bench" % (name, sigma)
            assert np.array_equal(ok, ref["ok"]), \
                "%s sigma=%g: decisions differ from the bench" % (name, sigma)
            print("  %-12s %7gpx %7d %12s %12s"
                  % (name, sigma, len(mine), "identical", "identical"))


def check_arms_share_draws(lm) -> None:
    """Both arms must see the same gaze evidence, or the CSVs are not paired."""
    pairs = load_sentences(N_SENTENCES, seed=4, skip=2060)
    a = list(typing_trial(lm, pairs, 40.0, 0.0, 0))
    b = list(typing_trial(lm, pairs, 40.0, tuned_phase_scale(), 0))
    assert len(a) == len(b), "arms disagree on keystroke count"
    assert [r["ground_truth"] for r in a] == [r["ground_truth"] for r in b], \
        "arms disagree on the ground-truth character sequence"
    assert [r["sentence_index"] for r in a] == [r["sentence_index"] for r in b], \
        "arms disagree on sentence order"
    print("  %d keystrokes, same ground truth and same order in both arms"
          % len(a))


def check_csv_arithmetic(tmp: str) -> None:
    run("quantum", [40.0], 1, 8, live=False, png_every=10_000,
        out_csv=os.path.join(tmp, "c.csv"),
        out_json=os.path.join(tmp, "c.json"),
        out_png=os.path.join(tmp, "c.png"))
    with open(os.path.join(tmp, "c.csv")) as fh:
        rows = list(csv.DictReader(fh))
    assert rows, "no rows written"
    assert list(rows[0].keys()) == CSV_COLUMNS, \
        "CSV header is %s, expected %s" % (list(rows[0].keys()), CSV_COLUMNS)

    correct = 0
    for i, r in enumerate(rows, start=1):
        assert int(r["keystroke_index"]) == i, \
            "keystroke_index jumped at row %d: %s" % (i, r["keystroke_index"])
        assert r["correct"] in ("0", "1"), "correct must be 0 or 1"
        assert int(r["correct"]) == int(r["prediction"] == r["ground_truth"]), \
            "correct column disagrees with prediction vs ground_truth at row %d" % i
        correct += int(r["correct"])
        want = 100.0 * correct / i
        got = float(r["cumulative_accuracy"])
        assert abs(got - want) < 1e-6, \
            "cumulative_accuracy at row %d is %.6f, running mean is %.6f" % (i, got, want)
    print("  %d rows: header exact, index contiguous from 1, cumulative accuracy"
          " is the running mean" % len(rows))


def main() -> None:
    print("=" * 66)
    print("EVALUATION LAYER — NON-INTERFERENCE AND ARITHMETIC")
    print("=" * 66)
    dec = HybridDecoder(vocab_path=VOCAB, model_dir=None, num_clusters=9)
    lm = LetterModel(dec.unigrams)

    print("\n1. logger reproduces selection_rule_bench.run_keystrokes")
    check_matches_bench(lm)

    print("\n2. both arms see identical gaze draws")
    check_arms_share_draws(lm)

    print("\n3. CSV arithmetic")
    with tempfile.TemporaryDirectory() as tmp:
        check_csv_arithmetic(tmp)

    print("\n" + "=" * 66)
    print("PASS — the evaluation layer leaves the model's decisions untouched.")
    print("=" * 66)


if __name__ == "__main__":
    main()
