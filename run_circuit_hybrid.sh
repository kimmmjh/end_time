#!/bin/bash

set -uo pipefail

ROOT="$HOME/end_time"
GPU_ID="${GPU_ID:-0}"
L="${1:-5}"

if [[ ! "$L" =~ ^(5|7|9)$ ]]; then
    echo "Usage: bash $0 [5|7|9]" >&2
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

experiments=(
    "--architecture=convgru_mwpm --gru_channels=96 --gru_layers=2 --gru_kernel_size=3 --noise_model=circuit --rounds=$L --measurement_error_rate=0.008 --p=0.008 --loss_fn=ce --epochs=300 --batch_size=64 --batches=1024 --eval_batches=512 --hybrid_calibration_batches=256 --final_eval_batches=2048 --L=$L --channels 96 96 96 --depths 4 4 4 --lr=0.0003 --amp_dtype=none --seed=5108 --save_model"
    "--architecture=convgru_mwpm --gru_channels=96 --gru_layers=2 --gru_kernel_size=3 --noise_model=circuit --rounds=$L --measurement_error_rate=0.009 --p=0.009 --loss_fn=ce --epochs=300 --batch_size=64 --batches=1024 --eval_batches=512 --hybrid_calibration_batches=256 --final_eval_batches=2048 --L=$L --channels 96 96 96 --depths 4 4 4 --lr=0.0003 --amp_dtype=none --seed=5109 --save_model"
    "--architecture=convgru_mwpm --gru_channels=96 --gru_layers=2 --gru_kernel_size=3 --noise_model=circuit --rounds=$L --measurement_error_rate=0.010 --p=0.010 --loss_fn=ce --epochs=300 --batch_size=64 --batches=1024 --eval_batches=512 --hybrid_calibration_batches=256 --final_eval_batches=2048 --L=$L --channels 96 96 96 --depths 4 4 4 --lr=0.0003 --amp_dtype=none --seed=5110 --save_model"
    "--architecture=convgru_mwpm --gru_channels=96 --gru_layers=2 --gru_kernel_size=3 --noise_model=circuit --rounds=$L --measurement_error_rate=0.011 --p=0.011 --loss_fn=ce --epochs=300 --batch_size=64 --batches=1024 --eval_batches=512 --hybrid_calibration_batches=256 --final_eval_batches=2048 --L=$L --channels 96 96 96 --depths 4 4 4 --lr=0.0003 --amp_dtype=none --seed=5111 --save_model"
    "--architecture=convgru_mwpm --gru_channels=96 --gru_layers=2 --gru_kernel_size=3 --noise_model=circuit --rounds=$L --measurement_error_rate=0.012 --p=0.012 --loss_fn=ce --epochs=300 --batch_size=64 --batches=1024 --eval_batches=512 --hybrid_calibration_batches=256 --final_eval_batches=2048 --L=$L --channels 96 96 96 --depths 4 4 4 --lr=0.0003 --amp_dtype=none --seed=5112 --save_model"
)

CURRENT_PYTHON_PID=""

cleanup() {
    echo
    echo "Stopping run script..."

    if [[ -n "${CURRENT_PYTHON_PID:-}" ]] &&
       kill -0 "$CURRENT_PYTHON_PID" 2>/dev/null; then

        echo "Stopping Python PID: $CURRENT_PYTHON_PID"
        kill -TERM "$CURRENT_PYTHON_PID" 2>/dev/null
        wait "$CURRENT_PYTHON_PID" 2>/dev/null
    fi

    {
        echo "Interrupted: $(date)"
        echo "Last experiment PID: ${CURRENT_PYTHON_PID:-none}"
    } > "$RESDIR/interrupted.txt"

    echo "Stopped."
    exit 130
}

# Ctrl+C also terminates the currently running Python process.
trap cleanup INT TERM

run_experiment() {
    local index="$1"
    local params="$2"

    local expdir="$RESDIR/exp_${index}"
    mkdir "$expdir" || return 1

    echo "$params" > "$expdir/command.txt" || return 1

    cd "$expdir" || return 1

    CUDA_VISIBLE_DEVICES="$GPU_ID" \
        python3 -u "$ROOT/main.py" $params &

    CURRENT_PYTHON_PID=$!

    echo "Python PID: $CURRENT_PYTHON_PID"
    echo "Experiment $index started with Python PID $CURRENT_PYTHON_PID"

    wait "$CURRENT_PYTHON_PID"
    local exit_code=$?

    {
        echo "Finished: $(date)"
        echo "Exit code: $exit_code"
    } > finished.txt
    echo "$exit_code" > exit_code.txt

    CURRENT_PYTHON_PID=""
    cd "$ROOT" || exit 1

    echo "Experiment $index finished with exit code $exit_code"
    return "$exit_code"
}

echo "Run script PID: $SCRIPT_PID"
echo "Results directory: $RESDIR"
echo "Using physical GPU: $GPU_ID"
echo "Lattice size: $L"

for i in "${!experiments[@]}"; do
    echo
    echo "Starting experiment $i on GPU $GPU_ID"
    run_experiment "$i" "${experiments[$i]}"
    exit_code=$?
    if ((exit_code != 0)); then
        {
            echo "Experiment $i failed"
            echo "Exit code: $exit_code"
            echo "Failed: $(date)"
        } > "$RESDIR/failed.txt"
        echo "Stopping after experiment $i failed with exit code $exit_code." >&2
        exit "$exit_code"
    fi
done

{
    echo "All experiments completed"
    echo "Finished: $(date)"
} > "$RESDIR/completed.txt"

echo
echo "All experiments finished."
echo "Results directory: $RESDIR"
