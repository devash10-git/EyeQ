# Fake-webcam fixture

`face.y4m` is a looping portrait that MediaPipe genuinely tracks, used by
`tests/e2e/calibration_flow.js` so the calibration flow can be driven without a
real camera. It is 13 MB and regenerated rather than committed:

```bash
curl -sL -o portrait.jpg https://storage.googleapis.com/mediapipe-assets/portrait.jpg
ffmpeg -y -loop 1 -i portrait.jpg -t 2 -r 15 \
  -vf "crop=560:420:130:0,scale=640:480,format=yuv420p" face.y4m
```

Crop so the eye-outer distance is at least 0.10 of frame width and the nose x
lands in 0.30–0.70, or the tracker reports "small"/"offcenter" instead of "ok".

The face is static, so every calibration dot yields near-identical features.
This fixture tests the **flow**, never accuracy. Accuracy requires
`eval/capture.py` with a real person.
