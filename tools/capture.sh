#!/usr/bin/env bash
# Open the gaze capture page with the camera already permitted.
#
# Chrome is launched with --use-fake-ui-for-media-stream, which auto-accepts the
# getUserMedia prompt and hands the page the REAL camera. No dialog, no clicking
# "Allow" — the capture starts as soon as your face is tracked.
#
# A throwaway profile is used on purpose: that flag is browser-wide, and you do
# not want every site in your normal profile silently granted the camera.
#
#   tools/capture.sh            # starts the server if needed, opens the page
#   tools/capture.sh --keep     # leave the temp profile behind for inspection
#
# One thing this cannot do for you: macOS itself must have granted Chrome camera
# access. If the preview stays black, open
#   System Settings -> Privacy & Security -> Camera
# and enable Google Chrome. That switch is OS-level and needs your password.
set -euo pipefail
cd "$(dirname "$0")/.."

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PORT="${CAPTURE_PORT:-8322}"
URL="http://127.0.0.1:${PORT}/capture"
PROFILE="$(mktemp -d -t gazecapture)"

[ -x "$CHROME" ] || { echo "Google Chrome not found at $CHROME"; exit 1; }

if ! curl -sf -o /dev/null "$URL"; then
  echo "starting capture server on :${PORT} ..."
  CAPTURE_PORT="$PORT" python3 eval/capture.py >/tmp/gaze_capture_server.log 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 40); do curl -sf -o /dev/null "$URL" && break; sleep 0.25; done
  curl -sf -o /dev/null "$URL" || { echo "server did not come up; see /tmp/gaze_capture_server.log"; exit 1; }
  trap '[ -n "${SERVER_PID:-}" ] && kill "$SERVER_PID" 2>/dev/null || true' EXIT
fi

echo "opening Chrome with the camera pre-granted ..."
echo "  recording takes about 4 minutes: 21 targets, 5 fixations each"
echo "  results land in eval/sessions/ ; then run:  python3 eval/gaze_bench.py"
"$CHROME" \
  --user-data-dir="$PROFILE" \
  --use-fake-ui-for-media-stream \
  --no-first-run --no-default-browser-check \
  --start-maximized \
  "$URL"

if [ "${1:-}" = "--keep" ]; then echo "profile kept at $PROFILE"; else rm -rf "$PROFILE"; fi
