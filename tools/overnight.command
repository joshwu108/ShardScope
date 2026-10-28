#!/bin/bash
# Keep the wake assertion scoped to Claude's process lifetime.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROMPT="$ROOT/planning/overnight.command.prompt"

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--check" ) ]]; then
    echo "Usage: bash tools/overnight.command [--check]" >&2
    exit 2
fi

cd "$ROOT"
for executable in claude caffeinate git tee; do
    if ! command -v "$executable" >/dev/null 2>&1; then
        echo "error: required command is missing: $executable" >&2
        exit 2
    fi
done
if [[ ! -s "$PROMPT" ]]; then
    echo "error: missing local prompt: $PROMPT" >&2
    echo "The prompt lives in ignored planning/; it is not included in a fresh clone." >&2
    exit 2
fi
if [[ "$(git rev-parse --show-toplevel)" != "$ROOT" ]]; then
    echo "error: the launcher must belong to the ShardScope checkout." >&2
    exit 2
fi

if [[ "${1:-}" == "--check" ]]; then
    echo "Ready: $ROOT"
    echo "Prompt: $PROMPT"
    echo "Command: caffeinate -dims claude --dangerously-skip-permissions --print --verbose --output-format stream-json"
    echo "Check only: Claude was not started. Authentication and project setup are checked during the run."
    exit 0
fi

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
    source "$ROOT/.venv/bin/activate"
fi

RUN_DIR="$(mktemp -d "$ROOT/planning/overnight.XXXXXX")"
LOG="$RUN_DIR/session.jsonl"
echo "Working directory: $ROOT"
echo "Session log: $LOG"
echo "Permission checks are bypassed for this run, as requested."
echo "Keep this terminal open. The wake assertion ends when Claude exits."

# Preserve Claude's failure status even though tee is the last pipeline command.
set +e
caffeinate -dims claude \
    --dangerously-skip-permissions \
    --print \
    --verbose \
    --output-format stream-json \
    < "$PROMPT" 2>&1 | tee "$LOG"
statuses=("${PIPESTATUS[@]}")
set -e
claude_status="${statuses[0]}"
log_status="${statuses[1]}"
printf '%s\n' "$claude_status" > "$RUN_DIR/exit-code.txt"
echo "Claude exited with status $claude_status. Log: $LOG"
if [[ "$claude_status" -ne 0 ]]; then
    exit "$claude_status"
fi
exit "$log_status"
