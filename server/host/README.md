# Host-side artifacts

Files in this directory run on the **brev host** (or any Ubuntu/systemd box that
hosts the `ai-latentsync` container), not inside the container itself.

## `latentsync-idle-watchdog`

Polls `http://127.0.0.1:8014/readyz` every minute. After a configurable streak
of consecutive `status: idle, queue_depth: 0` reports, runs `poweroff` so a
brev VM stops the GPU-billed instance. The brev control plane detects the
shutdown and marks the instance stopped — no more GPU charges until someone
runs `brev start <instance>`.

**Install** (from repo root):
```bash
sudo server/host/install.sh
```

**Disable temporarily**:
```bash
sudo systemctl stop latentsync-idle-watchdog.timer
```

**Logs**:
```bash
sudo journalctl -u latentsync-idle-watchdog.service --no-pager -n 50
```

**Tunables** (override via systemd env or edit `/usr/local/bin/latentsync-idle-watchdog.sh`):

| var | default | purpose |
|---|---|---|
| `READYZ_URL` | `http://127.0.0.1:8014/readyz` | endpoint to probe |
| `IDLE_STREAK_S` | `1800` (30 min) | sustained-idle threshold before poweroff |
| `MIN_UPTIME_S` | `600` (10 min) | post-boot grace, avoids shutdown loops |
| `STATE_FILE` | `/var/lib/latentsync-idle-watchdog/last_active` | tracks last "active" moment |

The state file is touched on every "active" or "unreachable" probe; the script
shuts down only when `now - state_file_mtime >= IDLE_STREAK_S` and `/readyz`
currently reports idle. A new submit anywhere during the streak resets the
counter via the next probe (it'll see `status: busy`).
