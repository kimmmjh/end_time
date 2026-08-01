#!/bin/bash

set -uo pipefail

ROOT="$HOME/end_time"
GPU_ID=1  # Change this number to the physical GPU you want this script to use.

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

source "$HOME/env/bin/activate" || exit 1
cd "$ROOT" || exit 1

SCRIPT_PID=$$
RESDIR="$ROOT/resdir_${SCRIPT_PID}"
if [[ -e "$RESDIR" ]]; then
    echo "Error: directory already exists: $RESDIR" >&2
    exit 1
fi
mkdir "$RESDIR" || exit 1

# Since rounds=L, runtime scales roughly with L^3. This partition has estimated
# cost 3030 versus 2955 in run_0.sh, and the two lists cover the grid exactly.
EXPERIMENT_L=(5 5 5 5 7 9 9 9)
EXPERIMENT_P=(0.008 0.009 0.011 0.012 0.010 0.008 0.010 0.012)
EXPERIMENT_SEED=(700508 700509 700511 700512 700710 700908 700910 700912)
TOTAL=${#EXPERIMENT_L[@]}

CURRENT_PYTHON_PID=""
CURRENT_EXPERIMENT=""

cleanup() {
    trap - INT TERM
    echo
    echo "Stopping run_1.sh..."
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
    local exit_code
    local -a params=(
        --architecture=convgru_weighted_mwpm
        --gru_channels=96
        --gru_layers=2
        --gru_kernel_size=3
        --noise_model=circuit
        --rounds="$L"
        --measurement_error_rate="$p"
        --p="$p"
        --loss_fn=edge_bce
        --epochs=300
        --batch_size=32
        --batches=2048
        --eval_batches=128
        --eval_every=5
        --final_eval_batches=512
        --L="$L"
        --channels 96 96 96
        --depths 4 4 4
        --lr=0.0003
        --edge_hidden_channels=192
        --edge_delta_scale=6
        --edge_chunk_size=1024
        --edge_entropy_weight=0
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
        echo "Runner: run_1.sh"
        echo "GPU: $GPU_ID"
        echo "L: $L"
        echo "p=q: $p"
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
         "(PID $CURRENT_PYTHON_PID): L=$L p=q=$p seed=$seed"
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

echo "Runner: run_1.sh"
echo "Run script PID: $SCRIPT_PID"
echo "Results directory: $RESDIR"
echo "Physical GPU: $GPU_ID"
echo "Experiment points: $TOTAL"

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
    echo "Runner: run_1.sh"
    echo "GPU: $GPU_ID"
    echo "Finished: $(date)"
} > "$RESDIR/completed.txt"

echo
echo "run_1.sh finished successfully."
echo "Results directory: $RESDIR"
