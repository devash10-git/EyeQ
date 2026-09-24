"""Real-time transcription test session: gaze-type a given word, score it live.

The participant is shown a target sentence, types it by gaze, and every commit
is scored as it happens. Nothing here is simulated: real webcam, real
fixations, real dwell commits, real wall-clock latency.

Two rule modes, and the difference matters
------------------------------------------
The deployed keyboard commits with log-linear fusion, computed locally in
``static/app.js::smartCommit``. The quantum conditional selection rule lives in
``quantum/conditional_selection.py`` and has never been in the typing path. So
"real-time typing with the quantum model" cannot mean shadow-scoring alone —
the rule has to actually decide the keystroke.

    rule=classical   The keyboard decides locally, exactly as deployed. This
                     module is never on the commit path; it receives the
                     decision afterwards, fire-and-forget, and scores it. The
                     model is provably untouched.

    rule=quantum     The keyboard sends the gaze posterior here, waits, and
                     commits what select() returns. This module IS the decision
                     for that session. That is the point of the mode, but it
                     means the added round-trip is part of the measurement, so
                     both the model-only time and the round-trip are recorded
                     separately.

In both modes the *other* rule is scored on the same evidence at the same
instant, so one session yields a paired within-session comparison for free.

Prefix handling. Both arms are scored against the prefix actually in the text
field, not a ground-truth prefix. A live session has one text field and one
typing history; inventing a second, cleaner prefix for one arm would measure
something that did not happen. The consequence is that after an error the
shared prefix came from whichever rule was driving — recorded in the CSV as
``rule_used`` so it can be accounted for.

Ground truth. The target sentence supplies the expected character by position.
The pointer advances on every commit, including wrong ones, and participants
are told not to correct mid-trial: per-keystroke ground truth requires a fixed
alignment, and backspacing destroys it.

Endpoints (all additive):

    POST /api/eval/start      begin a session, get the target sentences
    POST /api/eval/decide     quantum mode: the commit decision, plus scoring
    POST /api/eval/keystroke  classical mode: score a decision already made
    GET  /api/eval/series     the live chart's lines
    GET  /api/eval/summary    end-of-trial totals
    POST /api/eval/reset      discard the session

No selection rule is implemented in this file. ``loglinear`` and ``select`` are
imported from ``quantum.conditional_selection`` exactly as the offline bench
imports them.
"""

from __future__ import annotations

import csv
import json
import os
import threading
import time
from typing import Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from quantum.conditional_selection import loglinear, select

HERE = os.path.dirname(os.path.abspath(__file__))

# Deployed constants. Same values as static/app.js and selection_rule_bench.
LM_BETA = 0.70
LM_FLOOR = 1e-4
THETA = 0.80
N_HYP = 32

# Fixed target pool; sessions slice its first n. See eval_start().
POOL_N = 60

CSV_COLUMNS = ["trial", "sentence", "keystroke_index", "ground_truth",
               "prediction", "correct", "cumulative_accuracy"]
RAW_COLUMNS = CSV_COLUMNS + ["top_posterior", "decision_latency_ms",
                             "rule_used", "prefix"]

router = APIRouter()

_letters = None          # LetterModel, injected by app.py
_lock = threading.Lock()


def configure(letters_model) -> None:
    """Hand this module the app's own LetterModel, not a second copy."""
    global _letters
    _letters = letters_model


def phase_scale() -> float:
    """The scale tuned on the train split, from selection_rule_results.json."""
    try:
        with open(os.path.join(HERE, "selection_rule_results.json")) as fh:
            return float(json.load(fh)["phase_scale"])
    except (OSError, KeyError, ValueError, TypeError):
        return 0.40


# ==========================================================================
# Per-arm running totals
# ==========================================================================

class Arm:
    """Running totals for one selection rule within one session."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.rows: List[Dict[str, object]] = []
        self.raw: List[Dict[str, object]] = []
        self.committed = 0
        self.correct = 0
        self.presented = 0
        self.latency_ms: List[float] = []
        self.x: List[int] = []           # committed keystroke number
        self.y: List[float] = []         # cumulative commit accuracy (%)
        self.prompt_y: List[float] = []  # cumulative prompt rate (%)
        self.lat_y: List[float] = []     # cumulative mean latency (ms)

    def record(self, trial: int, sentence: str, truth: str, pred: str,
               top_p: float, latency_ms: float, rule_used: str,
               prefix: str) -> None:
        self.presented += 1
        self.latency_ms.append(latency_ms)
        if top_p < THETA:
            return                        # prompted, not committed
        self.committed += 1
        ok = int(pred == truth)
        self.correct += ok
        acc = 100.0 * self.correct / self.committed
        row = {"trial": trial, "sentence": sentence,
               "keystroke_index": self.committed,
               "ground_truth": truth, "prediction": pred,
               "correct": ok, "cumulative_accuracy": round(acc, 6)}
        self.rows.append(row)
        self.raw.append({**row, "top_posterior": round(top_p, 6),
                         "decision_latency_ms": round(latency_ms, 4),
                         "rule_used": rule_used, "prefix": prefix})
        self.x.append(self.committed)
        self.y.append(acc)
        self.prompt_y.append(
            100.0 * (self.presented - self.committed) / self.presented)
        self.lat_y.append(sum(self.latency_ms) / len(self.latency_ms))

    def summary(self) -> Dict[str, object]:
        n, lat = self.committed, self.latency_ms
        return {
            "model": self.name,
            "final_commit_accuracy_pct": (100.0 * self.correct / n) if n else None,
            "total_committed": n,
            "correct": self.correct,
            "incorrect": n - self.correct,
            "keystrokes_presented": self.presented,
            "prompts": self.presented - n,
            "prompt_rate_pct": (100.0 * (self.presented - n) / self.presented)
                               if self.presented else None,
            "mean_latency_ms": (sum(lat) / len(lat)) if lat else None,
            "median_latency_ms": (sorted(lat)[len(lat) // 2]) if lat else None,
        }


class Session:
    def __init__(self, trial: int, rule: str, outdir: str,
                 sentences: List[str]) -> None:
        self.trial = trial
        self.rule = rule                 # which rule drives the commits
        self.outdir = outdir
        self.sentences = sentences
        self.started = time.time()
        self.phase_scale = phase_scale()
        self.arms = {"quantum": Arm("quantum"), "classical": Arm("classical")}
        self.elapsed_s = 0.0
        self.chars_entered = 0
        self.finished = False

    def speed(self) -> Dict[str, object]:
        """Entry rate. Characters that reached the text field over elapsed time.

        This counts characters the participant resolved from a prompt, because
        those cost time and produce text — it is throughput, not rule accuracy.
        ``commit_wpm`` is the part the selection rule produced on its own.
        WPM uses the text-entry convention that a word is five characters.
        """
        mins = self.elapsed_s / 60.0
        if mins <= 0:
            return {"elapsed_s": self.elapsed_s, "chars_entered": self.chars_entered,
                    "wpm": None, "cpm": None, "commit_wpm": None}
        committed = max(a.committed for a in self.arms.values())
        return {"elapsed_s": round(self.elapsed_s, 2),
                "chars_entered": self.chars_entered,
                "wpm": (self.chars_entered / 5.0) / mins,
                "cpm": self.chars_entered / mins,
                "commit_wpm": (committed / 5.0) / mins,
                "committed_chars": committed}


_session: Optional[Session] = None


# ==========================================================================
# Request models
# ==========================================================================

class StartRequest(BaseModel):
    trial: int = 1
    rule: str = "classical"          # "classical" | "quantum"
    n_sentences: int = 20
    outdir: str = HERE


class DecideRequest(BaseModel):
    """One keystroke's gaze evidence, from the live keyboard."""
    p_gaze: Dict[str, float]         # raw P(key | gaze), app.js keyPosterior()
    prefix: str = ""                 # the word actually typed so far
    truth: str                       # expected character at this position
    sentence: str = ""
    # classical mode only: what the keyboard already decided locally, so the
    # log is the keyboard's own verdict rather than a recomputation of it
    prediction: Optional[str] = None
    top_p: Optional[float] = None
    latency_ms: Optional[float] = None
    # Entry rate, measured by the client because only it can see the text
    # field and the participant's wall clock.
    elapsed_s: float = 0.0
    chars_entered: int = 0


def _score(req: DecideRequest) -> Dict[str, object]:
    """Run both rules on one piece of gaze evidence. Returns both verdicts."""
    pg = {k: float(v) for k, v in req.p_gaze.items() if v > 0}
    if not pg:
        return {}
    z = sum(pg.values())
    pg = {k: v / z for k, v in pg.items()}

    prior = _letters.letter_prior(req.prefix)
    comps = _letters.completions(req.prefix, n=N_HYP)

    t0 = time.perf_counter()
    post_c = loglinear(pg, prior, beta=LM_BETA, floor=LM_FLOOR)
    lat_c = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    post_q = select(pg, prior, comps, prefix=req.prefix, beta=LM_BETA,
                    floor=LM_FLOOR, phase_scale=_session.phase_scale)
    lat_q = (time.perf_counter() - t0) * 1000.0

    out = {}
    for name, post, lat in (("classical", post_c, lat_c),
                            ("quantum", post_q, lat_q)):
        best = max(post, key=post.get)
        out[name] = {"prediction": best, "top_p": float(post[best]),
                     "latency_ms": lat}
    return out


def _commit_both(req: DecideRequest, scored: Dict[str, object]) -> Dict[str, object]:
    """Record both arms and return the per-arm live state for the client."""
    if _session.finished:
        pass
    elif req.elapsed_s > _session.elapsed_s:
        _session.elapsed_s = req.elapsed_s
    if not _session.finished:
        _session.chars_entered = max(_session.chars_entered, req.chars_entered)
    out: Dict[str, object] = {"ok": True, "trial": _session.trial,
                              "rule": _session.rule, "speed": _session.speed()}
    for name in ("classical", "quantum"):
        v = scored[name]
        pred, top_p, lat = v["prediction"], v["top_p"], v["latency_ms"]
        # In classical mode the keyboard already made its own decision locally.
        # Log that, not a server-side recomputation of it, so the CSV is the
        # keyboard's real behaviour including its own timing.
        if (name == "classical" and _session.rule == "classical"
                and req.prediction is not None and req.top_p is not None):
            pred, top_p = req.prediction, float(req.top_p)
            if req.latency_ms is not None:
                lat = float(req.latency_ms)
        arm = _session.arms[name]
        arm.record(_session.trial, req.sentence, req.truth, pred, top_p, lat,
                   _session.rule, req.prefix)
        out[name] = {"prediction": pred, "top_p": top_p,
                     "committed": top_p >= THETA,
                     "correct": int(pred == req.truth),
                     "latency_ms": lat,
                     "cumulative_accuracy": arm.y[-1] if arm.y else None,
                     "committed_count": arm.committed,
                     "correct_count": arm.correct}
    _write_files(_session)
    return out


# ==========================================================================
# Endpoints
# ==========================================================================

@router.post("/api/eval/start")
def eval_start(req: StartRequest):
    """Begin a trial and hand back the target sentences to transcribe."""
    global _session
    if req.rule not in ("classical", "quantum"):
        return {"ok": False, "error": "rule must be classical or quantum"}
    sentences, source = [], "unavailable"
    try:
        from tune import load_sentences
        # Always draw from a FIXED pool and slice, never load_sentences(n).
        # That helper sizes its CSV read as nrows=n+skip+5000, so the row pool
        # -- and therefore the shuffle -- changes with n: asking for 3
        # sentences and asking for 20 return different text, and two sessions
        # run at different n would not be transcribing the same task.
        pool = [en for _c, en in load_sentences(POOL_N, seed=4, skip=2060)]
        sentences = pool[:req.n_sentences]
        source = ("load_sentences(n=%d, seed=4, skip=2060)[:%d]"
                  % (POOL_N, req.n_sentences))
    except Exception as exc:                     # noqa: BLE001
        source = "%s: %s" % (type(exc).__name__, exc)
    with _lock:
        _session = Session(req.trial, req.rule, req.outdir, sentences)
    return {"ok": True, "trial": req.trial, "rule": req.rule,
            "sentences": sentences, "source": source,
            "phase_scale": _session.phase_scale, "theta": THETA,
            "lm_beta": LM_BETA, "lm_floor": LM_FLOOR}


@router.post("/api/eval/decide")
def eval_decide(req: DecideRequest):
    """Quantum mode: decide this keystroke, and score both rules.

    The client awaits this and commits ``quantum.prediction`` when
    ``quantum.committed`` is true. This call is therefore on the commit path by
    design; ``latency_ms`` is the model's own time, and the client separately
    records the round-trip.
    """
    if _session is None:
        return {"ok": False, "error": "no session; POST /api/eval/start first"}
    if _letters is None:
        return {"ok": False, "error": "eval module not configured"}
    scored = _score(req)
    if not scored:
        return {"ok": False, "error": "empty gaze posterior"}
    with _lock:
        return _commit_both(req, scored)


@router.post("/api/eval/keystroke")
def eval_keystroke(req: DecideRequest):
    """Classical mode: score a decision the keyboard already made.

    Fire-and-forget from the client's point of view — it does not await this,
    so nothing here can delay or alter a keystroke.
    """
    return eval_decide(req)


class FinishRequest(BaseModel):
    """The client's final entry-rate figures, pushed once when the trial ends."""
    elapsed_s: float = 0.0
    chars_entered: int = 0


@router.post("/api/eval/finish")
def eval_finish(req: FinishRequest):
    """Freeze the session's entry rate at the client's final measurement.

    Only the client can time this: it owns the wall clock the participant typed
    against and the text field the characters landed in. Polling the server
    mid-session samples it at whatever instant the request arrived, which is why
    the on-screen figure and the saved JSON could disagree. One push at the end
    makes them the same number by construction.
    """
    if _session is None:
        return {"ok": False, "error": "no session"}
    with _lock:
        _session.elapsed_s = req.elapsed_s
        _session.chars_entered = req.chars_entered
        _session.finished = True
        _write_files(_session)
        return {"ok": True, "speed": _session.speed()}


@router.get("/api/eval/series")
def eval_series():
    """The live chart's lines, plus the alternate y-fields."""
    if _session is None:
        return {"ok": False, "error": "no session"}
    with _lock:
        return {"ok": True, "trial": _session.trial, "rule": _session.rule,
                "x_field": "keystroke_index",
                "series": {n: {"x": a.x,
                               "cumulative_accuracy_pct": a.y,
                               "cumulative_prompt_rate_pct": a.prompt_y,
                               "cumulative_mean_latency_ms": a.lat_y}
                           for n, a in _session.arms.items()}}


@router.get("/api/eval/summary")
def eval_summary():
    if _session is None:
        return {"ok": False, "error": "no session"}
    with _lock:
        return {"ok": True, "trial": _session.trial, "rule": _session.rule,
                "phase_scale": _session.phase_scale, "theta": THETA,
                "lm_beta": LM_BETA, "lm_floor": LM_FLOOR,
                "seconds": round(time.time() - _session.started, 1),
                "speed": _session.speed(),
                "arms": {n: a.summary() for n, a in _session.arms.items()}}


@router.post("/api/eval/reset")
def eval_reset():
    global _session
    with _lock:
        _session = None
    return {"ok": True}


def _write_files(s: Session) -> None:
    """Rewrite the logs after every keystroke, so an aborted run keeps its data."""
    for name, arm in s.arms.items():
        base = os.path.join(s.outdir, "live_commit_log_%s" % name)
        with open(base + ".csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            w.writeheader()
            w.writerows(arm.rows)
        with open(base + "_raw.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=RAW_COLUMNS)
            w.writeheader()
            w.writerows(arm.raw)
    with open(os.path.join(s.outdir, "live_commit_summary.json"), "w") as fh:
        json.dump({"trial": s.trial, "rule_driving_commits": s.rule,
                   "phase_scale": s.phase_scale, "theta": THETA,
                   "lm_beta": LM_BETA, "lm_floor": LM_FLOOR,
                   "seconds": round(time.time() - s.started, 1),
                   "speed": s.speed(),
                   "arms": {n: a.summary() for n, a in s.arms.items()},
                   "series": {n: {"x": a.x,
                                  "cumulative_accuracy_pct": a.y,
                                  "cumulative_mean_latency_ms": a.lat_y}
                              for n, a in s.arms.items()}},
                  fh, indent=2)
