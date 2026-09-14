"""Did the quantum gaze->letter stack help or hurt typing? (It hurt.)

The keyboard used to multiply this stack's letter posterior into each
candidate's score, behind ``ENABLE_QGAZE=1``::

    score = P(key|gaze) * P(letter|prefix)^0.70 * P_q(letter)^0.50

This script is the measurement that removed that term. It replays real recorded
fixations through the live ``/api/gaze_letter`` endpoint -- real iris position,
real head pose, real kinematics, not synthetic inputs -- and scores the letter
decision with and without the quantum factor on identical gaze evidence.

Ground truth is the key nearest the recorded target, so "correct" means the
decision agreed with where the participant was actually looking.

Two mechanisms make the term harmful, and both are reported here:

* the endpoint returns only the **top-5** posterior, so every other letter is
  floored at ``QGAZE_FLOOR = 1e-3`` and suppressed no matter how confident the
  gaze was; and
* the stack's own fusion reports which channels it used, and on real data it
  reports ``['geom']`` -- the *classical* geometric channel -- essentially
  always. It discards its quantum channels and returns the classical answer,
  having spent ~68 ms doing so.

Requires the server: ``python3 -m uvicorn app:app --port 8321``.
The ``ENABLE_QGAZE`` flag no longer exists; this script calls the endpoint
directly, so it measures the same thing the flag used to switch on.

Run: ``python3 eval/qgaze_influence.py``
"""

from __future__ import annotations

import collections
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from decoder import HybridDecoder                                   # noqa: E402
from eval.feature_ablation import columns                           # noqa: E402
from eval.gaze_bench import (                                       # noqa: E402
    fit_ridge, load_sessions, session_quality,
)
from letters import LetterModel                                     # noqa: E402

URL = "http://127.0.0.1:8321/api/gaze_letter"
RESULTS = Path(__file__).resolve().parent / "qgaze_influence.json"

# the deployed constants this term lived among (static/app.js)
LM_BETA, LM_FLOOR = 0.70, 1e-4
QGAZE_BETA, QGAZE_FLOOR = 0.50, 1e-3
SIGMA_PX = 120.0            # representative of the measured live gaze sigma
PER_SESSION = 220
ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]


def ncdf(x: float, mu: float, s: float) -> float:
    return 0.5 * (1.0 + math.erf((x - mu) / (s * math.sqrt(2.0))))


def ask(payload: dict) -> Optional[dict]:
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=10).read())
    except (urllib.error.URLError, TimeoutError):
        return None


def keyboard(w: float, h: float):
    """Key rectangles and centres, as the deployed layout arranges them."""
    kw = w / 10.0
    kh = kw * 1.1
    y0 = h - 3 * kh - 20
    rects, cen = {}, {}
    for r, row in enumerate(ROWS):
        indent = (10 - len(row)) * kw / 2.0
        for i, ch in enumerate(row):
            rects[ch] = (indent + i * kw, y0 + r * kh,
                         indent + (i + 1) * kw, y0 + (r + 1) * kh)
            cen[ch] = (indent + (i + 0.5) * kw, y0 + (r + 0.5) * kh)
    return rects, cen


def main() -> None:
    if ask({"iris_x": 0.5, "iris_y": 0.5}) is None:
        print("server not reachable at %s — start it first:" % URL)
        print("  python3 -m uvicorn app:app --host 127.0.0.1 --port 8321")
        sys.exit(1)

    S = [s for s in load_sessions() if session_quality(s)["usable"]]
    lm = LetterModel(HybridDecoder(vocab_path="vocab-9.json", model_dir=None,
                                   num_clusters=9).unigrams)
    prior = lm.letter_prior("")
    rng = np.random.default_rng(0)

    print("=" * 72)
    print("QUANTUM GAZE->LETTER STACK: influence on the letter decision")
    print("=" * 72)
    print("Real recorded fixations -> live /api/gaze_letter -> the QGAZE term.")
    print("Ground truth: the key nearest the recorded target.\n")
    print("  %-11s %7s %13s %13s %11s" %
          ("session", "n", "classical", "with QGAZE", "change"))
    print("  " + "-" * 60)

    rows, channels, methods, lat = [], collections.Counter(), collections.Counter(), []
    tot_c = tot_q = tot_n = 0
    flips_bad = flips_good = 0

    for s in S:
        rects, cen = keyboard(s.meta["screen"]["w_px"], s.meta["screen"]["h_px"])
        X = columns(s, "canthus")
        pred = np.empty_like(s.T)
        for g in np.unique(s.groups):                   # leave-one-target-out
            te = s.groups == g
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                pred[te] = X[te] @ fit_ridge(X[~te], s.T[~te])

        idx = rng.choice(len(X), size=min(PER_SESSION, len(X)), replace=False)
        okc = okq = n = 0
        for i in idx:
            e = s.extras[int(i)]
            f = e["f"]
            t0 = time.perf_counter()
            q = ask({
                "iris_x": float((f[1] + f[2]) / 2.0),
                "iris_y": float((f[3] + f[4]) / 2.0),
                "yaw": float(e.get("yaw", 0.0)),
                "pitch": float(e.get("pitch", 0.0)),
                "ear": float((e.get("ear_l", 0.0) + e.get("ear_r", 0.0)) / 2.0),
                "velocity": float(e.get("gaze_velocity", 0.0)),
                "stability": float(e.get("quality_geometric", 0.9)),
                "history": 0.0,
            })
            lat.append((time.perf_counter() - t0) * 1000.0)
            if not q or not q.get("available"):
                continue
            channels[tuple(q.get("channels_used") or [])] += 1
            methods[q.get("method")] += 1
            qpost = dict(q.get("posterior") or [])

            gx, gy = pred[i]
            tx, ty = s.T[i]
            want = min(cen, key=lambda c: (cen[c][0] - tx) ** 2 + (cen[c][1] - ty) ** 2)

            pg, tot = {}, 0.0
            for ch, (x0, y0, x1, y1) in rects.items():
                p = ((ncdf(x1, gx, SIGMA_PX) - ncdf(x0, gx, SIGMA_PX)) *
                     (ncdf(y1, gy, SIGMA_PX) - ncdf(y0, gy, SIGMA_PX)))
                if p > 1e-9:
                    pg[ch] = p
                    tot += p
            if tot <= 0:
                continue

            classical = {c: (v / tot) * max(prior.get(c, 0.0), LM_FLOOR) ** LM_BETA
                         for c, v in pg.items()}
            withq = {c: v * max(qpost.get(c, 0.0), QGAZE_FLOOR) ** QGAZE_BETA
                     for c, v in classical.items()}
            bc = max(classical, key=classical.get)
            bq = max(withq, key=withq.get)
            okc += bc == want
            okq += bq == want
            n += 1
            flips_bad += (bc == want and bq != want)
            flips_good += (bc != want and bq == want)

        if n:
            print("  %-11s %7d %12.1f%% %12.1f%% %+10.1f pp"
                  % (s.label[-9:], n, 100 * okc / n, 100 * okq / n,
                     100 * (okq - okc) / n))
            rows.append({"session": s.label, "n": n, "classical": okc / n,
                         "with_qgaze": okq / n, "delta_pp": 100 * (okq - okc) / n})
            tot_c, tot_q, tot_n = tot_c + okc, tot_q + okq, tot_n + n

    print("  " + "-" * 60)
    print("  %-11s %7d %12.1f%% %12.1f%% %+10.1f pp"
          % ("POOLED", tot_n, 100 * tot_c / tot_n, 100 * tot_q / tot_n,
             100 * (tot_q - tot_c) / tot_n))

    print("\n  commits flipped correct -> wrong by the term: %d" % flips_bad)
    print("  commits flipped wrong -> correct by the term: %d" % flips_good)

    print("\n  Which channels did the stack actually use?")
    for k, v in channels.most_common():
        print("    %-30s %4d  (%.0f%%)" % (str(list(k)), v, 100 * v / max(1, sum(channels.values()))))
    print("    ('geom' is the classical geometric channel; the quantum ones are"
          " qsvm / qknn / qhier / vqc)")

    print("\n  Cost: median %.1f ms per call (frame budget is 16.7 ms at 60fps)"
          % statistics.median(lat))

    out = {"rows": rows, "pooled_classical": tot_c / tot_n,
           "pooled_with_qgaze": tot_q / tot_n,
           "pooled_delta_pp": 100 * (tot_q - tot_c) / tot_n,
           "flips_correct_to_wrong": flips_bad,
           "flips_wrong_to_correct": flips_good,
           "channels_used": {",".join(k) or "(none)": v for k, v in channels.items()},
           "median_latency_ms": statistics.median(lat), "sigma_px": SIGMA_PX}
    RESULTS.write_text(json.dumps(out, indent=2, default=float))
    print("\nwrote %s" % RESULTS)


if __name__ == "__main__":
    main()
