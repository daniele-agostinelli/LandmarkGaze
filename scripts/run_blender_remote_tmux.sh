#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${SESSION:-blender_remote}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/blender_remote}"
mkdir -p "$LOG_DIR"
touch "$LOG_DIR/session.log"

if ! command -v tmux >/dev/null 2>&1; then
  echo "Error: tmux not found. Install tmux and retry."
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Error: tmux session '$SESSION' already exists."
  echo "Close it with: tmux kill-session -t $SESSION"
  exit 1
fi

tmux new-session -d -s "$SESSION" "cd \"$ROOT_DIR\" && bash \"$ROOT_DIR/scripts/run_blender_remote_batch_7gpu.sh\" > \"$LOG_DIR/session.log\" 2>&1"
tmux new-window -t "$SESSION" -n "logs" "cd \"$ROOT_DIR\" && tail -F \"$LOG_DIR/session.log\""

echo "Started tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
echo "Logs dir: $LOG_DIR"
