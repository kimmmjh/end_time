#!/bin/bash

set -uo pipefail

ROOT="$HOME/end_time"
L_SELECTION="${1:-all}"
MATCHING_MODE="${2:-standard}"
SHOTS="${SHOTS:-262144}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
SEED="${SEED:-12345}"

case "$L_SELECTION" in
    all)
        L_VALUES=(5 7 9)
        ;;
    5|7|9)
        L_VALUES=("$L_SELECTION")
        ;;
    *)
        echo "Usage: bash $0 [all|5|7|9] [standard|correlated]" >&2
        exit 2
        ;;
esac

case "$MATCHING_MODE" in
    standard)
        CORRELATION_ARGS=()
        ;;
    correlated)
        CORRELATION_ARGS=(--enable_correlations)
        ;;
    *)
        echo "Usage: bash $0 [all|5|7|9] [standard|correlated]" >&2
        exit 2
        ;;
esac

if [[ ! "$SHOTS" =~ ^[1-9][0-9]*$ ]] ||
   [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] ||
   [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
    echo "Error: SHOTS and BATCH_SIZE must be positive integers; SEED must be non-negative." >&2
    exit 2
fi

if [[ ! -f "$HOME/env/bin/activate" ]]; then
    echo "Error: virtual environment not found: $HOME/env/bin/activate" >&2
    exit 1
fi

source "$HOME/env/bin/activate" || exit 1
cd "$ROOT" || exit 1

SCRIPT_PID=$$
RESDIR="$ROOT/resdir_${SCRIPT_PID}"

if [[ -e "$RESDIR" ]]; then
    echo "Error: directory already exists: $RESDIR"
    exit 1
fi

mkdir "$RESDIR" || exit 1

OUTPUT="$RESDIR/circuit_pymatching_${MATCHING_MODE}.csv"
LOG="$RESDIR/log_pymatching.txt"
CURRENT_PYTHON_PID=""

params=(
    --L "${L_VALUES[@]}"
    --p 0.008 0.009 0.010 0.011 0.012
    --shots "$SHOTS"
    --batch_size "$BATCH_SIZE"
    --seed "$SEED"
    --output "$OUTPUT"
    "${CORRELATION_ARGS[@]}"
)

{
    printf 'python3 -u %q ' "$ROOT/scripts/circuit_pymatching_threshold.py"
    printf '%q ' "${params[@]}"
    echo
} > "$RESDIR/command.txt"

cleanup() {
    echo
    echo "Stopping PyMatching benchmark..."

    if [[ -n "${CURRENT_PYTHON_PID:-}" ]] &&
       kill -0 "$CURRENT_PYTHON_PID" 2>/dev/null; then

        echo "Stopping Python PID: $CURRENT_PYTHON_PID"
        kill -TERM "$CURRENT_PYTHON_PID" 2>/dev/null
        wait "$CURRENT_PYTHON_PID" 2>/dev/null
    fi

    {
        echo "Interrupted: $(date)"
        echo "Python PID: ${CURRENT_PYTHON_PID:-none}"
    } > "$RESDIR/interrupted.txt"

    echo "Stopped."
    exit 130
}

trap cleanup INT TERM

echo "Run script PID: $SCRIPT_PID"
echo "Results directory: $RESDIR"
echo "Lattice sizes: ${L_VALUES[*]}"
echo "Matching mode: $MATCHING_MODE"
echo "Shots per point: $SHOTS"

python3 -u "$ROOT/scripts/circuit_pymatching_threshold.py" \
    "${params[@]}" > "$LOG" 2>&1 &
CURRENT_PYTHON_PID=$!

echo "Python PID: $CURRENT_PYTHON_PID"
wait "$CURRENT_PYTHON_PID"
exit_code=$?
CURRENT_PYTHON_PID=""

cat "$LOG"
echo "$exit_code" > "$RESDIR/exit_code.txt"

if ((exit_code != 0)); then
    {
        echo "PyMatching benchmark failed"
        echo "Exit code: $exit_code"
        echo "Failed: $(date)"
    } > "$RESDIR/failed.txt"
    exit "$exit_code"
fi

{
    echo "PyMatching benchmark completed"
    echo "Matching mode: $MATCHING_MODE"
    echo "Finished: $(date)"
} > "$RESDIR/completed.txt"

echo
echo "PyMatching benchmark finished."
echo "CSV: $OUTPUT"
