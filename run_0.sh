#!/bin/bash

set -uo pipefail

ROOT="$HOME/end_time"
GPU_ID=0  # Physical GPU used by this script.

EPOCHS=300
EVAL_BATCHES=256
FINAL_EVAL_BATCHES=2048
LEARNING_RATE=0.0003

# Fresh pure-ConvGRU runs that fill the unresolved threshold-curve gaps.
# run_1.sh contains the complementary p values.
EXPERIMENT_L=(13 15 13 15)
EXPERIMENT_P=(0.016 0.011 0.018 0.013)
EXPERIMENT_SEED=(130016 150011 130018 150013)
TOTAL=${#EXPERIMENT_L[@]}

if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "Error: GPU_ID must be a non-negative integer." >&2
    exit 2
fi
if [[ ! -f "$HOME/env/bin/activate" ]]; then
    echo "Error: virtual environment not found: $HOME/env/bin/activate" >&2
    exit 1
fi
if [[ ! -f "$ROOT/main.py" ]]; then
    echo "Error: repository main.py not found under ROOT=$ROOT" >&2
    exit 1
fi
if ((${#EXPERIMENT_P[@]} != TOTAL || ${#EXPERIMENT_SEED[@]} != TOTAL)); then
    echo "Error: experiment arrays have different lengths." >&2
    exit 2
fi

source "$HOME/env/bin/activate" || exit 1
cd "$ROOT" || exit 1

SCRIPT_PID=$$
RESDIR="$ROOT/resdir_${SCRIPT_PID}"
if [[ -e "$RESDIR" ]]; then
    echo "Error: directory already exists: $RESDIR" >&2
    exit 1
fi
mkdir "$RESDIR" || exit 1

CURRENT_PYTHON_PID=""
CURRENT_EXPERIMENT=""

cleanup() {
    trap - INT TERM
    echo
    echo "Stopping run_0.sh..."
    if [[ -n "${CURRENT_PYTHON_PID:-}" ]] &&
       kill -0 "$CURRENT_PYTHON_PID" 2>/dev/null; then
        echo "Stopping Python PID: $CURRENT_PYTHON_PID"
        kill -TERM "$CURRENT_PYTHON_PID" 2>/dev/null || true
        wait "$CURRENT_PYTHON_PID" 2>/dev/null || true
    fi
    {
        echo "Interrupted: $(date)"
        echo "Current experiment: ${CURRENT_EXPERIMENT:-none}"
        echo "Python PID: ${CURRENT_PYTHON_PID:-none}"
    } > "$RESDIR/interrupted.txt"
    exit 130
}
trap cleanup INT TERM

run_experiment() {
    local index="$1"
    local L="${EXPERIMENT_L[$index]}"
    local p="${EXPERIMENT_P[$index]}"
    local seed="${EXPERIMENT_SEED[$index]}"
    local expdir="$RESDIR/exp_${index}"
    local batch_size
    local batches
    local exit_code

    case "$L" in
        13)
            batch_size=32
            batches=1024
            ;;
        15)
            batch_size=16
            batches=2048
            ;;
        *)
            echo "Error: unsupported lattice size: $L" >&2
            return 2
            ;;
    esac

    local -a params=(
        --architecture=convgru
        --gru_channels=96
        --gru_layers=2
        --gru_kernel_size=3
        --noise_model=phenomenological
        --rounds="$L"
        --measurement_error_rate="$p"
        --p="$p"
        --loss_fn=ce
        --epochs="$EPOCHS"
        --batch_size="$batch_size"
        --batches="$batches"
        --eval_batches="$EVAL_BATCHES"
        --eval_every=5
        --final_eval_batches="$FINAL_EVAL_BATCHES"
        --L="$L"
        --channels 96 96 96
        --depths 4 4 4
        --lr="$LEARNING_RATE"
        --amp_dtype=none
        --seed="$seed"
        --save_model
    )

    mkdir "$expdir" || return 1
    {
        printf 'CUDA_VISIBLE_DEVICES=%q python3 -u %q ' "$GPU_ID" "$ROOT/main.py"
        printf '%q ' "${params[@]}"
        echo
    } > "$expdir/command.txt" || return 1
    {
        echo "Started: $(date)"
        echo "Runner: run_0.sh"
        echo "GPU: $GPU_ID"
        echo "Architecture: pure convgru (no MWPM)"
        echo "Training mode: fresh"
        echo "L: $L"
        echo "p=q: $p"
        echo "Maximum LR: $LEARNING_RATE"
        echo "Epochs: $EPOCHS"
        echo "Seed: $seed"
    } > "$expdir/started.txt" || return 1

    CURRENT_EXPERIMENT="$index"
    (
        cd "$expdir" || exit 1
        exec env CUDA_VISIBLE_DEVICES="$GPU_ID" \
            python3 -u "$ROOT/main.py" "${params[@]}"
    ) > "$expdir/log.txt" 2>&1 &
    CURRENT_PYTHON_PID=$!

    echo "Experiment $index/$((TOTAL - 1)) started on GPU $GPU_ID "\
         "(PID $CURRENT_PYTHON_PID): L=$L p=q=$p"
    wait "$CURRENT_PYTHON_PID"
    exit_code=$?

    echo "$exit_code" > "$expdir/exit_code.txt"
    {
        echo "Finished: $(date)"
        echo "Exit code: $exit_code"
        echo "GPU: $GPU_ID"
    } > "$expdir/finished.txt"
    CURRENT_PYTHON_PID=""
    CURRENT_EXPERIMENT=""

    if ((exit_code != 0)); then
        {
            echo "Experiment $index failed"
            echo "Exit code: $exit_code"
            echo "Failed: $(date)"
        } > "$RESDIR/failed.txt"
        return "$exit_code"
    fi

    echo "Experiment $index finished successfully."
}

echo "Runner: run_0.sh"
echo "Run script PID: $SCRIPT_PID"
echo "Results directory: $RESDIR"
echo "Physical GPU: $GPU_ID"
echo "Architecture: pure convgru (no MWPM)"
for index in "${!EXPERIMENT_L[@]}"; do
    echo "  exp_$index: L=${EXPERIMENT_L[$index]} p=q=${EXPERIMENT_P[$index]}"
done

for index in "${!EXPERIMENT_L[@]}"; do
    run_experiment "$index"
    exit_code=$?
    if ((exit_code != 0)); then
        echo "Stopping after experiment $index failed with exit code $exit_code." >&2
        exit "$exit_code"
    fi
done

{
    echo "All $TOTAL experiments completed"
    echo "Runner: run_0.sh"
    echo "GPU: $GPU_ID"
    echo "Architecture: pure convgru (no MWPM)"
    echo "Finished: $(date)"
} > "$RESDIR/completed.txt"

echo
echo "run_0.sh finished successfully."
echo "Results directory: $RESDIR"
