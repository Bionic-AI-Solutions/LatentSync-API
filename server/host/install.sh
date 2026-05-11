#!/usr/bin/env bash
# Install the LatentSync idle watchdog on a brev host (or any Ubuntu/systemd box).
# Polls the LatentSync container's /readyz; after sustained idle, runs poweroff
# so a brev instance stops the GPU-billed VM.
#
# Run from the repo root:
#   sudo server/host/install.sh
#
# Uninstall:
#   sudo systemctl disable --now latentsync-idle-watchdog.timer
#   sudo rm -f /usr/local/bin/latentsync-idle-watchdog.sh \
#              /etc/systemd/system/latentsync-idle-watchdog.{service,timer}
#   sudo systemctl daemon-reload

set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "must run as root (try: sudo $0)" >&2
    exit 1
fi

HERE="$(cd "$(dirname "$0")" && pwd)"

install -m 755 "$HERE/latentsync-idle-watchdog.sh" /usr/local/bin/latentsync-idle-watchdog.sh
install -m 644 "$HERE/latentsync-idle-watchdog.service" /etc/systemd/system/latentsync-idle-watchdog.service
install -m 644 "$HERE/latentsync-idle-watchdog.timer" /etc/systemd/system/latentsync-idle-watchdog.timer

systemctl daemon-reload
systemctl enable --now latentsync-idle-watchdog.timer
systemctl status --no-pager latentsync-idle-watchdog.timer | head -8
