#!/usr/bin/env bash
# Auto-poweroff the brev latentsync box after sustained idle to stop GPU billing.
#
# Triggered every minute by `latentsync-idle-watchdog.timer`. Powers off iff:
#   1. uptime >= MIN_UPTIME_S  (don't shut down during boot before /readyz responds)
#   2. /readyz returns status=idle  (no job running, queue empty)
#   3. condition (2) held for the last IDLE_STREAK_S seconds
#
# Disable: sudo systemctl disable --now latentsync-idle-watchdog.timer
# Inspect: journalctl -u latentsync-idle-watchdog.service --no-pager -n 50

set -euo pipefail

READYZ_URL="${READYZ_URL:-http://127.0.0.1:8014/readyz}"
STATE_FILE="${STATE_FILE:-/var/lib/latentsync-idle-watchdog/last_active}"
IDLE_STREAK_S="${IDLE_STREAK_S:-1800}"  # 30 min
MIN_UPTIME_S="${MIN_UPTIME_S:-600}"     # 10 min

mkdir -p "$(dirname "$STATE_FILE")"

uptime_s=$(awk '{print int($1)}' /proc/uptime)
if [ "$uptime_s" -lt "$MIN_UPTIME_S" ]; then
    echo "uptime=${uptime_s}s < MIN_UPTIME_S=${MIN_UPTIME_S}s — skip"
    exit 0
fi

body=$(curl -sS --max-time 5 "$READYZ_URL" 2>/dev/null || true)
if [ -z "$body" ]; then
    echo "readyz unreachable — server still loading or down, reset idle streak"
    touch "$STATE_FILE"
    exit 0
fi

# Parse {"status":"idle|busy|loading","queue_depth":N}
status=$(printf '%s' "$body" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','?'))" 2>/dev/null || echo "?")
queue=$(printf '%s' "$body" | python3 -c "import sys,json; print(json.load(sys.stdin).get('queue_depth',-1))" 2>/dev/null || echo "-1")

if [ "$status" != "idle" ] || [ "$queue" != "0" ]; then
    echo "readyz status=$status queue=$queue — active, reset streak"
    touch "$STATE_FILE"
    exit 0
fi

if [ ! -f "$STATE_FILE" ]; then
    echo "no prior state — initialise"
    touch "$STATE_FILE"
    exit 0
fi

last_active=$(stat -c "%Y" "$STATE_FILE")
now=$(date +%s)
streak=$((now - last_active))

if [ "$streak" -ge "$IDLE_STREAK_S" ]; then
    echo "idle ${streak}s >= ${IDLE_STREAK_S}s — powering off"
    /sbin/poweroff
else
    echo "idle ${streak}s (need ${IDLE_STREAK_S}s) — wait"
fi
