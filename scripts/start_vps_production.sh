#!/usr/bin/env bash
# MegaX VPS production — FastAPI GUI on :18555.
#
# Usage:
#   ./scripts/start_vps_production.sh {start|stop|restart|status}
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GUI_HOST="${MEGAX_GUI_HOST:-0.0.0.0}"
GUI_PORT="${MEGAX_GUI_PORT:-18555}"
PID_FILE="${ROOT}/megax-gui.pid"
LOG_FILE="${ROOT}/logs/gui.log"
STOP_TIMEOUT_SEC="${MEGAX_STOP_TIMEOUT_SEC:-15}"

log() { printf '%s\n' "$*"; }
err() { printf 'Error: %s\n' "$*" >&2; }

read_pid() {
  [[ -f "$PID_FILE" ]] && tr -d '[:space:]' < "$PID_FILE"
}

is_alive() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

find_gui_pids() {
  pgrep -f "[Pp]ython.*megax-gui" 2>/dev/null || true
}

any_running() {
  local pid pids
  pid="$(read_pid || true)"
  is_alive "$pid" && return 0
  pids="$(find_gui_pids)"
  [[ -n "$pids" ]]
}

cmd_stop() {
  log "Stopping MegaX GUI..."
  local pid pids waited=0
  pid="$(read_pid || true)"
  if is_alive "$pid"; then
    kill -TERM "$pid" 2>/dev/null || true
  fi
  while read -r p; do
    [[ -z "$p" ]] && continue
    kill -TERM "$p" 2>/dev/null || true
  done < <(find_gui_pids)
  while any_running && [[ "$waited" -lt "$STOP_TIMEOUT_SEC" ]]; do
    sleep 1
    waited=$((waited + 1))
  done
  if any_running; then
    while read -r p; do kill -KILL "$p" 2>/dev/null || true; done < <(find_gui_pids)
  fi
  rm -f "$PID_FILE"
  log "MegaX GUI stopped."
}

cmd_start() {
  if [[ ! -x "$ROOT/.venv/bin/megax-gui" ]]; then
    err "Missing .venv — run: python3 -m venv .venv && .venv/bin/pip install -e ."
    exit 1
  fi
  if any_running; then
    log "MegaX GUI already running — restarting."
    cmd_stop
  fi
  mkdir -p "$(dirname "$LOG_FILE")"
  log "[start] megax-gui :${GUI_PORT}"
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
  nohup megax-gui --host "$GUI_HOST" --port "$GUI_PORT" >> "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  sleep 2
  if ! any_running; then
    err "megax-gui failed to start — see $LOG_FILE"
    tail -5 "$LOG_FILE" >&2 || true
    exit 1
  fi
  cmd_status
}

cmd_status() {
  log "MegaX production (repo: $ROOT)"
  if any_running; then
    log "  megax-gui :${GUI_PORT}  RUNNING  pid(s)=$(find_gui_pids | tr '\n' ' ')"
  else
    log "  megax-gui :${GUI_PORT}  STOPPED"
  fi
  [[ -f "$LOG_FILE" ]] && tail -n 1 "$LOG_FILE" | sed 's/^/  last: /' || true
}

cmd_restart() { cmd_stop; cmd_start; }

case "${1:-}" in
  start) cmd_start ;;
  stop) cmd_stop ;;
  restart) cmd_restart ;;
  status) cmd_status ;;
  *) sed -n '2,5p' "$0" | sed 's/^# //'; exit 1 ;;
esac
