#!/bin/bash

set -uo pipefail

ROOT="${ROOT:-$HOME/end_time}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/env/bin/activate}"
read -r -a GPUS <<< "${GPU_IDS:-0 1}"

if ((${#GPUS[@]} != 2)); then
    echo "Error: GPU_IDS must contain exactly two GPU IDs, for example GPU_IDS='0 1'." >&2
    exit 2
fi
if [[ ! "${GPUS[0]}" =~ ^[0-9]+$ || ! "${GPUS[1]}" =~ ^[0-9]+$ ||
      "${GPUS[0]}" == "${GPUS[1]}" ]]; then
    echo "Error: GPU_IDS must contain two distinct non-negative integers." >&2
    exit 2
fi
if [[ ! -f "$VENV_ACTIVATE" ]]; then
    echo "Error: virtual environment not found: $VENV_ACTIVATE" >&2
    exit 1
fi
if [[ ! -f "$ROOT/main.py" ]]; then
    echo "Error: repository main.py not found under ROOT=$ROOT" >&2
    exit 1
fi

source "$VENV_ACTIVATE" || exit 1
cd "$ROOT" || exit 1

SCRIPT_PID=$$
RESDIR="$ROOT/resdir_${SCRIPT_PID}"
if [[ -e "$RESDIR" ]]; then
    echo "Error: directory already exists: $RESDIR" >&2
    exit 1
fi
mkdir "$RESDIR" || exit 1

L_VALUES=(5 7 9)
P_VALUES=(0.008 0.009 0.010 0.011 0.012)
P_CODES=(8 9 10 11 12)
EXPERIMENT_L=()
EXPERIMENT_P=()
EXPERIMENT_SEED=()
for L in "${L_VALUES[@]}"; do
    for p_index in "${!P_VALUES[@]}"; do
        EXPERIMENT_L+=("$L")
        EXPERIMENT_P+=("${P_VALUES[$p_index]}")
        # Stable and unique across every (L, p) point in this grid.
        EXPERIMENT_SEED+=("$((700000 + L * 100 + P_CODES[$p_index]))")
    done
done
TOTAL=${#EXPERIMENT_L[@]}

RUNNING_PIDS=("" "")
RUNNING_INDEXES=("" "")
next_index=0

terminate_running() {
    local slot
    local pid
    local index
    local gpu
    local expdir
    local exit_code
    local -a term_sent=(0 0)
    for slot in "${!RUNNING_PIDS[@]}"; do
        pid="${RUNNING_PIDS[$slot]}"
        [[ -n "$pid" ]] || continue
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
            term_sent[$slot]=1
        fi
    done
    for slot in "${!RUNNING_PIDS[@]}"; do
        pid="${RUNNING_PIDS[$slot]}"
        [[ -n "$pid" ]] || continue
        wait "$pid" 2>/dev/null
        exit_code=$?
        index="${RUNNING_INDEXES[$slot]}"
        gpu="${GPUS[$slot]}"
        expdir="$RESDIR/exp_${index}"
        echo "$exit_code" > "$expdir/exit_code.txt"
        {
            echo "Finished: $(date)"
            echo "Exit code: $exit_code"
            echo "GPU: $gpu"
            echo "SIGTERM forwarded by runner: ${term_sent[$slot]}"
        } > "$expdir/finished.txt"
        RUNNING_PIDS[$slot]=""
        RUNNING_INDEXES[$slot]=""
    done
}

cleanup() {
    trap - INT TERM
    echo
    echo "Stopping full run and forwarding the signal to active experiments..."
    terminate_running
    {
        echo "Interrupted: $(date)"
        echo "Next unlaunched experiment: $next_index"
        echo "All active experiment processes received SIGTERM."
    } > "$RESDIR/interrupted.txt"
    exit 130
}
trap cleanup INT TERM

launch_experiment() {
    local index="$1"
    local slot="$2"
    local gpu="${GPUS[$slot]}"
    local L="${EXPERIMENT_L[$index]}"
    local p="${EXPERIMENT_P[$index]}"
    local seed="${EXPERIMENT_SEED[$index]}"
    local expdir="$RESDIR/exp_${index}"
    local -a params=(
        --architecture=convgru_mwpm
        --gru_channels=96
        --gru_layers=2
        --gru_kernel_size=3
        --noise_model=circuit
        --rounds="$L"
        --measurement_error_rate="$p"
        --p="$p"
        --loss_fn=ce
        --epochs=300
        --batch_size=64
        --batches=1024
        --eval_batches=512
        --hybrid_calibration_batches=256
        --final_eval_batches=2048
        --L="$L"
        --channels 96 96 96
        --depths 4 4 4
        --lr=0.0003
        --amp_dtype=none
        --seed="$seed"
        --save_model
    )

    mkdir "$expdir" || return 1
    {
        printf 'CUDA_VISIBLE_DEVICES=%q python3 -u %q ' "$gpu" "$ROOT/main.py"
        printf '%q ' "${params[@]}"
        echo
    } > "$expdir/command.txt" || return 1
    {
        echo "Started: $(date)"
        echo "GPU: $gpu"
        echo "L: $L"
        echo "p=q: $p"
        echo "Seed: $seed"
    } > "$expdir/started.txt" || return 1

    (
        cd "$expdir" || exit 1
        exec env CUDA_VISIBLE_DEVICES="$gpu" \
            python3 -u "$ROOT/main.py" "${params[@]}"
    ) > "$expdir/log.txt" 2>&1 &

    local pid=$!
    RUNNING_PIDS[$slot]="$pid"
    RUNNING_INDEXES[$slot]="$index"
    echo "Started experiment $index/$((TOTAL - 1)) on GPU $gpu (PID $pid): L=$L p=q=$p seed=$seed"
}

echo "Full runner PID: $SCRIPT_PID"
echo "Results directory: $RESDIR"
echo "GPUs: ${GPUS[*]}"
echo "Experiment points: $TOTAL"

running=0
for slot in "${!GPUS[@]}"; do
    if ! launch_experiment "$next_index" "$slot"; then
        terminate_running
        echo "Failed to launch experiment $next_index." >&2
        exit 1
    fi
    next_index=$((next_index + 1))
    running=$((running + 1))
done

while ((running > 0)); do
    finished_slot=-1
    while ((finished_slot < 0)); do
        for slot in "${!RUNNING_PIDS[@]}"; do
            pid="${RUNNING_PIDS[$slot]}"
            [[ -n "$pid" ]] || continue
            if ! kill -0 "$pid" 2>/dev/null; then
                finished_slot="$slot"
                break
            fi
        done
        if ((finished_slot < 0)); then
            sleep 1
        fi
    done

    finished_pid="${RUNNING_PIDS[$finished_slot]}"
    index="${RUNNING_INDEXES[$finished_slot]}"
    gpu="${GPUS[$finished_slot]}"
    wait "$finished_pid"
    exit_code=$?
    expdir="$RESDIR/exp_${index}"
    echo "$exit_code" > "$expdir/exit_code.txt"
    {
        echo "Finished: $(date)"
        echo "Exit code: $exit_code"
        echo "GPU: $gpu"
    } > "$expdir/finished.txt"
    RUNNING_PIDS[$finished_slot]=""
    RUNNING_INDEXES[$finished_slot]=""
    running=$((running - 1))

    if ((exit_code != 0)); then
        {
            echo "Experiment $index failed"
            echo "Exit code: $exit_code"
            echo "Failed: $(date)"
        } > "$RESDIR/failed.txt"
        echo "Experiment $index failed on GPU $gpu; stopping all remaining work." >&2
        terminate_running
        exit "$exit_code"
    fi

    echo "Experiment $index completed successfully on GPU $gpu."
    if ((next_index < TOTAL)); then
        if ! launch_experiment "$next_index" "$finished_slot"; then
            {
                echo "Failed to launch experiment $next_index"
                echo "Failed: $(date)"
            } > "$RESDIR/failed.txt"
            terminate_running
            exit 1
        fi
        next_index=$((next_index + 1))
        running=$((running + 1))
    fi
done

{
    echo "All $TOTAL full-grid experiments completed"
    echo "Finished: $(date)"
} > "$RESDIR/completed.txt"

echo
echo "Full ConvGRU+MWPM grid finished successfully."
echo "Results directory: $RESDIR"
