---
name: verify
description: Build/launch/drive recipe to verify the eye-gaze keyboard's camera + typing features headlessly with a fake webcam.
---

# Verifying the eye-gaze keyboard

## Launch

```bash
python3 app.py &          # serves http://127.0.0.1:8321 (deps: fastapi, uvicorn)
curl -s http://127.0.0.1:8321/api/info   # sanity: {"num_clusters":9,...}
```

## Fake webcam with a real face (MediaPipe genuinely tracks it)

```bash
curl -sL -o portrait.jpg https://storage.googleapis.com/mediapipe-assets/portrait.jpg
ffmpeg -y -loop 1 -i portrait.jpg -t 2 -r 15 \
  -vf "crop=560:420:130:0,scale=640:480,format=yuv420p" face.y4m
```

Crop so eye-outer distance ≥ 0.10 of frame width and nose x lands in 0.30–0.70,
or the tracker reports "small"/"offcenter" instead of "ok". Chrome loops the y4m.

## Drive with Playwright (pip install playwright; uses system Chrome, no browser download)

```python
browser = await p.chromium.launch(channel="chrome", headless=True, args=[
    "--use-fake-ui-for-media-stream",          # auto-grant camera (browser-wide!)
    "--use-fake-device-for-media-stream",
    "--use-file-for-fake-video-capture=face.y4m"])
```

The page exposes test hooks: `window.__gaze.state()` (text, cameraState,
faceState, predictionCount, calibrated, banner…), `.commitLetter/.commitAction`,
`.setPoint`, `.getTracker()`.

Flows worth driving: wait `state().faceState === 'ok'` (CDN+model load ≈
10–30 s first run) → calibrate all 7 spots (click `#calib-XX`, then click its
dot 5×; dots sit at 50px-inset corners/midpoints of the viewport) → gaze
dwell-types on its own with a static face → mouse mode dwell → suggestions →
reload (calibration persists in localStorage `iris-calib-v1`).

## Gotchas

- `--use-fake-ui-for-media-stream` is browser-level: test the permission-denied
  path in a **separate launch without it** (Playwright then auto-denies).
- Cursor is EMA-smoothed (0.35): synthetic `mouse.move` needs `steps=` plus a
  few 1px settle moves at the target, or the app cursor rests one key short.
- A parked cursor keeps dwell-typing the key under it (by design) — park the
  mouse off-keyboard (e.g. 640,100) between test phases.
- Simulate a dead MediaPipe CDN with `page.route("**://cdn.jsdelivr.net/**", abort)`;
  the app must fall back to mouse mode and release the camera (video track "ended").
