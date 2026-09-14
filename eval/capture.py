"""Recording server for real gaze-calibration sessions.

Serves ``eval/capture.html`` and writes each recorded session to
``eval/sessions/``. Runs standalone on its own port so it cannot disturb the
keyboard server, and it holds no state beyond the files it writes.

    python3 eval/capture.py            # http://127.0.0.1:8322/capture

Why this exists: the project has no measured gaze accuracy in any unit, on any
data. Every quantum result recorded so far was fitted and evaluated against the
same synthetic forward model, which cannot establish real-world error. This is
the only source of numbers that constraint C3 permits for accuracy claims.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

HERE = Path(__file__).resolve().parent
SESSIONS = HERE / "sessions"
SESSIONS.mkdir(exist_ok=True)

app = FastAPI(title="Gaze calibration capture")


@app.get("/capture")
def page() -> FileResponse:
    return FileResponse(HERE / "capture.html")


@app.post("/capture/save")
async def save(request: Request) -> JSONResponse:
    body = await request.json()

    # A session that cannot be traced back to a real recording is worthless for
    # accuracy claims, so the fields that make it traceable are required rather
    # than defaulted.
    missing = [k for k in ("samples", "screen", "viewing_distance_cm", "px_per_degree")
               if k not in body]
    if missing:
        return JSONResponse({"error": f"missing fields: {missing}"}, status_code=400)
    if not body["samples"]:
        return JSONResponse({"error": "no samples in session"}, status_code=400)
    body.setdefault("data_source", "recorded_human")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    who = re.sub(r"[^A-Za-z0-9_-]", "", str(body.get("participant") or "anon"))[:24] or "anon"
    path = SESSIONS / f"session_{stamp}_{who}.json"
    path.write_text(json.dumps(body))

    n_fit = sum(1 for s in body["samples"] if s.get("split") == "fit")
    n_hold = len(body["samples"]) - n_fit
    return JSONResponse({
        "path": str(path.relative_to(HERE.parent)),
        "samples": len(body["samples"]),
        "fit": n_fit,
        "holdout": n_hold,
    })


@app.get("/capture/sessions")
def sessions() -> dict:
    out = []
    for p in sorted(SESSIONS.glob("session_*.json")):
        try:
            d = json.loads(p.read_text())
            out.append({"file": p.name, "samples": len(d.get("samples", [])),
                        "participant": d.get("participant"),
                        "recorded_utc": d.get("recorded_utc")})
        except Exception as exc:  # a corrupt file must not hide the good ones
            out.append({"file": p.name, "error": str(exc)})
    return {"sessions": out}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("CAPTURE_PORT", "8322"))
    print(f"capture page:  http://127.0.0.1:{port}/capture")
    print(f"writing to:    {SESSIONS}")
    uvicorn.run(app, host="127.0.0.1", port=port)
