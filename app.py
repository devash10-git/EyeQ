import os

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from decoder import HybridDecoder
from letters import LetterModel

NUM_CLUSTERS = int(os.environ.get("NUM_CLUSTERS", "9"))
VOCAB_PATH = os.environ.get("VOCAB_PATH", f"vocab-{NUM_CLUSTERS}.json")
MODEL_DIR = os.environ.get("MODEL_DIR") or None

app = FastAPI(title="Eye Gaze Keyboard")
decoder = HybridDecoder(vocab_path=VOCAB_PATH, model_dir=MODEL_DIR, num_clusters=NUM_CLUSTERS)
letters = LetterModel(decoder.unigrams)

# Quantum-kernel eye-movement classifier. Optional: PennyLane is a heavy
# dependency and the base install must keep working without it, so a missing
# model or import degrades to the client's local threshold rule rather than
# failing startup.
_movement_model = None
_movement_error = "not loaded"
if os.environ.get("DISABLE_QSVM") != "1":
    try:
        from quantum.movement_qsvm import MovementQSVM

        _movement_model = MovementQSVM.load()
        _movement_error = ""
    except Exception as _exc:  # noqa: BLE001 - startup must never hard-fail
        _movement_error = f"{type(_exc).__name__}: {_exc}"


# QAOA typing-sequence optimizer (see quantum/typing_qaoa.py). Same guarded
# pattern as the QSVM: heavy quantum deps must never break the base install.
_typing_qaoa = None
_typing_qaoa_error = "not loaded"
if os.environ.get("DISABLE_QAOA") != "1":
    try:
        from quantum.typing_qaoa import TypingQAOA

        _typing_qaoa = TypingQAOA(decoder.unigrams)
        _typing_qaoa_error = ""
    except Exception as _exc:  # noqa: BLE001 - startup must never hard-fail
        _typing_qaoa_error = f"{type(_exc).__name__}: {_exc}"


class DecodeRequest(BaseModel):
    clusters: str  # e.g. "51663 531 421"


@app.post("/api/decode")
def decode(req: DecodeRequest):
    result = decoder.decode(req.clusters)
    return result.to_dict()


@app.get("/api/suggest")
def suggest(prefix: str = "", n: int = 6):
    # word autocomplete for the individual-letter QWERTY keyboard
    return {"words": decoder.suggest(prefix, n)}


@app.get("/api/letter_prior")
def letter_prior(prefix: str = ""):
    # P(next letter | word typed so far); "$" = end-of-word (SPACE)
    return {"prefix": prefix, "prior": letters.letter_prior(prefix)}


@app.get("/api/correct")
def correct(word: str = "", n: int = 3):
    # gaze-aware noisy-channel correction for an out-of-vocab word
    return letters.correct(word, n)


class MovementRequest(BaseModel):
    ear: float          # eye aspect ratio (openness)
    velocity: float     # normalised inter-frame gaze motion
    stability: float    # 1 - normalised recent variance
    history: float      # EMA of recent displacement


@app.post("/api/movement")
def movement(req: MovementRequest):
    """Classify eye-movement state with the quantum-kernel SVM.

    Replaces the hand-tuned threshold rule in the tracker (0.70 accuracy) with
    a fidelity-kernel SVM measured at 1.00 on held-out synthetic kinematics.
    Returns ``usable=True`` only for a clean fixation, so the caller can refuse
    to commit a keystroke sampled during a saccade or blink.

    Degrades safely: if the model file is absent the endpoint reports
    ``available=False`` and the client keeps using its local threshold rule.
    """
    if _movement_model is None:
        return {"available": False, "reason": _movement_error}
    try:
        features = [req.ear, req.velocity, req.stability, req.history]
        res = _movement_model.classify_one(features)
        # Graded reliability from the QNN's movement distribution pushed through
        # the quantum Bayesian network (blink/saccade -> unreliable -> mis-key).
        # The QSVM verdict above is binary; this is the continuous companion the
        # client uses to temper its posterior, which measurably cuts wrong
        # commits further (0.47 -> 0.33 per 100 fixations at sigma=0.02).
        # Kinematics only, so it needs no per-user calibration.
        if _gaze_stack is not None:
            try:
                res["reliability"] = float(
                    _gaze_stack.reliab.reliability([features])[0])
            except Exception:
                pass
        return {"available": True, **res}
    except Exception as exc:  # never let a classifier fault break typing
        return {"available": False, "reason": str(exc)}


# Quantum gaze->letter stack (quantum/gaze_letter.py). Same guarded pattern.
# Loaded for two consumers: the graded reliability that /api/movement returns,
# and the /api/gaze_letter research endpoint. It no longer steers typing.
_gaze_stack = None
_gaze_stack_error = "not loaded"
if os.environ.get("DISABLE_QGAZE") != "1":
    try:
        from quantum.gaze_letter import GazeLetterStack, LETTERS, vectors_from_client

        _gaze_stack = GazeLetterStack.load()
        _gaze_stack_error = ""
    except Exception as _exc:  # noqa: BLE001 - startup must never hard-fail
        _gaze_stack_error = f"{type(_exc).__name__}: {_exc}"


class GazeLetterRequest(BaseModel):
    iris_x: float       # horizontal iris position within the eye opening [0,1]
    iris_y: float       # vertical iris position within the eyelids       [0,1]
    yaw: float = 0.0    # head yaw   = (nose_x - 0.5) * 2
    pitch: float = 0.0  # head pitch = (nose_y - 0.5) * 2
    ear: float = 0.75
    velocity: float = 0.0
    stability: float = 0.9
    history: float = 0.0


@app.post("/api/gaze_letter")
def gaze_letter(req: GazeLetterRequest):
    """Decode the intended letter from one fixation with the quantum stack.

    Runs all seven stages (QAOA-selected features -> QSVM region -> VQC slot,
    quantum-kernel prototype matching, QNN + Bayesian-network reliability,
    QAOA consensus).

    **Research surface only — the keyboard does not call this.** It used to,
    behind ENABLE_QGAZE=1, multiplying this posterior into each letter's score.
    Measured on 660 real recorded fixations that cost 0.7 pp of letter accuracy
    (26.8% -> 26.1%), improved no session, and flipped six correct commits to
    wrong against one the other way. Two causes: the top-5 truncation below
    floors every other letter at 1e-3 regardless of what the gaze said, and
    across 240 fixations the stack reported ``channels_used=['geom']`` 100% of
    the time — its own fusion discarded every quantum channel and returned the
    classical answer, for 68 ms per call. Reproduce with
    ``eval/qgaze_influence.py``.
    """
    if _gaze_stack is None:
        return {"available": False, "reason": _gaze_stack_error}
    try:
        X, kin = vectors_from_client(req.model_dump())
        pred = _gaze_stack.predict_one(X, kin)
        top = sorted(enumerate(pred.posterior), key=lambda kv: -kv[1])[:5]
        return {
            "available": True,
            "letter": pred.letter,
            "candidates": [(c, round(float(p), 4)) for c, p in pred.candidates],
            "posterior": [(LETTERS[i], round(float(p), 4)) for i, p in top],
            "method": pred.method,
            "matched_exact": pred.matched_exact,
            "channels_used": pred.channels_used,
            "reliability": round(float(pred.reliability), 4),
            "latency_ms": round(pred.latency_ms, 1),
        }
    except Exception as exc:  # a decoder fault must never break typing
        return {"available": False, "reason": str(exc)}


class QaoaWordRequest(BaseModel):
    #: per typed letter, candidate [letter, prob] pairs sorted by prob
    positions: list


@app.post("/api/qaoa_word")
def qaoa_word(req: QaoaWordRequest):
    """Jointly re-decode a finished word with QAOA over its letter posteriors.

    Called asynchronously by the keyboard after SPACE — never on the frame or
    dwell path. Certain letters are frozen; each uncertain run is solved by a
    fixed-angle QAOA circuit trained offline (see quantum/typing_qaoa.py), with
    an exact chain-MAP fallback so the answer can never be worse than greedy.
    """
    if _typing_qaoa is None:
        return {"available": False, "reason": _typing_qaoa_error}
    try:
        positions = []
        for p in list(req.positions)[:12]:
            cands = []
            for c in list(p)[:2]:
                ch = str(c[0]).lower()
                if len(ch) == 1 and "a" <= ch <= "z":
                    cands.append((ch, max(float(c[1]), 1e-9)))
            if not cands:
                return {"available": False, "reason": "position with no letter candidates"}
            positions.append(cands)
        if not positions:
            return {"available": False, "reason": "empty word"}
        res = _typing_qaoa.decode_word(positions)
        return {
            "available": True,
            "word": res.word,
            "greedy": res.greedy,
            "changed": res.changed,
            "latency_ms": round(res.latency_ms, 1),
            "segments": [
                {"method": s.method, "qubits": s.qubits, "letters": s.letters,
                 "matched_exact": s.qaoa_matched_exact, "mass": s.qaoa_mass}
                for s in res.segments
            ],
        }
    except Exception as exc:  # a decoder fault must never break typing
        return {"available": False, "reason": str(exc)}


@app.get("/api/diag")
def diag(camera: str = "?", pred: int = 0, calib: int = 0, mouse: str = "?", err: str = ""):
    # client heartbeat: state also shows in the access log line via the query string
    return {"ok": True}


@app.get("/api/info")
def info():
    return {
        "num_clusters": NUM_CLUSTERS,
        "model_loaded": decoder.model is not None,
        "vocab_words": len(decoder.unigrams),
        # The stack is loaded for the reliability signal that /api/movement
        # returns; it no longer influences letter choice. ENABLE_QGAZE and the
        # matching "gaze_letter_active" flag are gone — see the comment on the
        # endpoint below for the measurement that removed them.
        "gaze_letter_loaded": _gaze_stack is not None,
        "eval_layer": _eval_error or "ready",
    }


# Real-time dual-arm evaluation (eval/live_eval.py). Strictly additive: it adds
# /api/eval/* routes that score each real commit under both selection rules and
# writes the CSVs. The keyboard's own decision path never calls into it, so the
# deployed model behaves identically whether or not this import succeeds.
try:
    from eval.live_eval import configure as _eval_configure, router as _eval_router

    _eval_configure(letters)
    app.include_router(_eval_router)
    _eval_error = ""
except Exception as _exc:  # noqa: BLE001 - evaluation must never break typing
    _eval_error = f"{type(_exc).__name__}: {_exc}"


@app.get("/")
def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8321")))
