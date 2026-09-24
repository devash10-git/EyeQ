"""Live two-line chart for a running test session, in its own window.

Deliberately a separate process. The chart used to be drawn inside the
keyboard page and it was in the way: anything rendered next to the keys either
covers them or shrinks them, and key geometry is part of what the session is
measuring. Put it on a second monitor, or behind the browser window — it never
touches the page.

Polls ``/api/eval/series`` while a session runs and draws:

    panel 1   cumulative commit accuracy (%)      quantum vs classical
    panel 2   cumulative mean decision latency (ms)

Both arms are scored on identical gaze evidence inside one session, so the two
lines are paired keystroke for keystroke.

Run alongside the server, while the session is open in the browser:

    python3 eval/live_dual_chart.py
    python3 eval/live_dual_chart.py --url http://127.0.0.1:8321 --interval 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Dict, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ("quantum", "classical")
COLOURS = {"quantum": "#2b7cd3", "classical": "#d2691e"}


def fetch(url: str, path: str) -> Optional[Dict]:
    try:
        with urllib.request.urlopen(url + path, timeout=3) as r:
            d = json.loads(r.read())
        return d if d.get("ok") else None
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--url", default="http://127.0.0.1:8321")
    p.add_argument("--interval", type=float, default=0.5,
                   help="seconds between polls")
    p.add_argument("--out", default=os.path.join(HERE, "live_dual_chart.png"))
    args = p.parse_args()

    import matplotlib
    interactive = False
    for backend in ("macosx", "qtagg", "tkagg"):
        try:
            matplotlib.use(backend, force=True)
            import matplotlib.pyplot as plt
            plt.close(plt.figure())
            interactive = True
            break
        except Exception:
            continue
    if not interactive:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.set_ylabel("Cumulative Commit Accuracy (%)")
    ax2.set_ylabel("Cumulative Mean Decision Latency (ms)")
    ax2.set_xlabel("Keystroke Number (committed)")
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
    lines = {}
    for a in ARMS:
        lines[(a, 1)] = ax1.plot([], [], lw=1.8, color=COLOURS[a], label=a)[0]
        lines[(a, 2)] = ax2.plot([], [], lw=1.8, color=COLOURS[a], label=a)[0]
    ax1.legend(loc="lower right", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)
    if interactive:
        plt.ion()
        fig.show()

    print("polling %s every %.1fs — %s" %
          (args.url, args.interval,
           "live window" if interactive
           else "headless, writing %s" % os.path.basename(args.out)))
    print("start a session at %s/?test=1&rule=quantum&n=20" % args.url)

    waiting = True
    try:
        while True:
            d = fetch(args.url, "/api/eval/series")
            if d is None:
                if waiting:
                    print("  waiting for a session…", end="\r", flush=True)
                time.sleep(args.interval)
                continue
            if waiting:
                waiting = False
                print("\n  session %s, commits driven by %s"
                      % (d.get("trial"), d.get("rule")))

            acc_all, lat_all, n_max = [], [], 0
            for a in ARMS:
                s = d["series"].get(a) or {}
                x = s.get("x") or []
                y1 = s.get("cumulative_accuracy_pct") or []
                y2 = s.get("cumulative_mean_latency_ms") or []
                lines[(a, 1)].set_data(x, y1)
                lines[(a, 2)].set_data(x, y2)
                acc_all += y1
                lat_all += y2
                n_max = max(n_max, len(x))

            if n_max >= 2:
                for ax in (ax1, ax2):
                    ax.set_xlim(1, n_max)
                # Accuracy lives in the top few percent; a 0-100 axis would
                # render it as a flat line, so track the data and keep 100 in
                # frame.
                lo = max(0.0, min(min(acc_all) - 2.0, 99.0))
                ax1.set_ylim(min(lo, 96.0) if lo > 96.0 else lo, 100.4)
                if lat_all:
                    hi = max(lat_all)
                    ax2.set_ylim(0, hi * 1.25 if hi > 0 else 1.0)
                sm = fetch(args.url, "/api/eval/summary") or {}
                sp = sm.get("speed") or {}
                ttl = "Trial %s — commits driven by %s" % (d.get("trial"), d.get("rule"))
                if sp.get("wpm"):
                    ttl += "   |   %.1f wpm, %.0f cpm" % (sp["wpm"], sp["cpm"])
                ax1.set_title(ttl)
                if interactive:
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
                else:
                    fig.savefig(args.out, dpi=130, bbox_inches="tight")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        fig.savefig(args.out, dpi=130, bbox_inches="tight")
        print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
