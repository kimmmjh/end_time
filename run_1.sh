#!/bin/bash

set -uo pipefail

ROOT="$HOME/end_time"
GPU_ID=1  # Change this number to the physical GPU you want this script to use.

# Pure-ConvGRU checkpoints. L13/p=.02 has 600 completed epochs; L15/p=.015
# has 300. The complementary branches live in run_0.sh, and together the two
# scripts cover every target/LR pair once while keeping GPU runtimes balanced.
L13_CHECKPOINT="$ROOT/resdir_1332513/exp_2/outputs/2026-07-21/23-49-20-036141_phenomenological_convgru_L13_r13_p0.02_q0.02_lr0.0003_bs32_b1024_eb1024_feb1024_ch96-96-96_d4-4-4_gru96x2_gk3_resume/model.pt"
L15_CHECKPOINT="$ROOT/resdir_1309506/exp_4/outputs/2026-07-22/01-56-12-401483_phenomenological_convgru_L15_r15_p0.015_q0.015_lr0.0003_bs16_b2048_eb2048_feb2048_ch96-96-96_d4-4-4_gru96x2_gk3/model.pt"

EVAL_BATCHES=256
FINAL_EVAL_BATCHES=2048
ADDITIONAL_EPOCHS=300

# The same seed is used for all LR branches of a target, making their newly
# generated training/evaluation streams directly comparable.
EXPERIMENT_L=(13 15 13)
EXPERIMENT_P=(0.020 0.015 0.020)
EXPERIMENT_LR=(0.0001 0.0003 0.001)
EXPERIMENT_SEED=(130020 150015 130020)
EXPERIMENT_CHECKPOINT=("$L13_CHECKPOINT" "$L15_CHECKPOINT" "$L13_CHECKPOINT")
TOTAL=${#EXPERIMENT_LR[@]}

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
if ((${#EXPERIMENT_L[@]} != TOTAL ||
     ${#EXPERIMENT_P[@]} != TOTAL ||
     ${#EXPERIMENT_SEED[@]} != TOTAL ||
     ${#EXPERIMENT_CHECKPOINT[@]} != TOTAL)); then
    echo "Error: experiment arrays have different lengths." >&2
    exit 2
fi
for checkpoint in "${EXPERIMENT_CHECKPOINT[@]}"; do
    if [[ ! -f "$checkpoint" ]]; then
        echo "Error: resume checkpoint not found:" >&2
        echo "  $checkpoint" >&2
        exit 1
    fi
done

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
    local lr="${EXPERIMENT_LR[$index]}"
    local seed="${EXPERIMENT_SEED[$index]}"
    local checkpoint="${EXPERIMENT_CHECKPOINT[$index]}"
    local expdir="$RESDIR/exp_${index}"
    local exit_code
    local batch_size
    local batches
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
            echo "Error: unsupported lattice size in experiment $index: $L" >&2
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
        --epochs="$ADDITIONAL_EPOCHS"
        --batch_size="$batch_size"
        --batches="$batches"
        --eval_batches="$EVAL_BATCHES"
        --eval_every=5
        --final_eval_batches="$FINAL_EVAL_BATCHES"
        --L="$L"
        --channels 96 96 96
        --depths 4 4 4
        --lr="$lr"
        --amp_dtype=none
        --seed="$seed"
        --save_model
        --load_model="$checkpoint"
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
        echo "Architecture: pure convgru (no MWPM)"
        echo "L: $L"
        echo "p=q: $p"
        echo "Maximum LR: $lr"
        echo "Additional epochs: $ADDITIONAL_EPOCHS"
        echo "Seed shared across this target's LR branches: $seed"
        echo "Checkpoint: $checkpoint"
    } > "$expdir/started.txt" || return 1

    CURRENT_EXPERIMENT="$index"
    (
        cd "$expdir" || exit 1
        exec env CUDA_VISIBLE_DEVICES="$GPU_ID" \
            python3 -u "$ROOT/main.py" "${params[@]}"
    ) > "$expdir/log.txt" 2>&1 &
    CURRENT_PYTHON_PID=$!

    echo "Experiment $index/$((TOTAL - 1)) started on GPU $GPU_ID "\
         "(PID $CURRENT_PYTHON_PID): L=$L p=q=$p lr=$lr"
    wait "$CURRENT_PYTHON_PID"
    exit_code=$?

    echo "$exit_code" > "$expdir/exit_code.txt"
    {
        echo "Finished: $(date)"
        echo "Exit code: $exit_code"
        echo "GPU: $GPU_ID"
        echo "Maximum LR: $lr"
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
echo "Architecture: pure convgru (no MWPM)"
echo "Experiment points: $TOTAL"
for index in "${!EXPERIMENT_L[@]}"; do
    echo "  exp_$index: L=${EXPERIMENT_L[$index]} "\
         "p=q=${EXPERIMENT_P[$index]} lr=${EXPERIMENT_LR[$index]}"
done

for index in "${!EXPERIMENT_LR[@]}"; do
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
    echo "Architecture: pure convgru (no MWPM)"
    echo "Finished: $(date)"
} > "$RESDIR/completed.txt"

echo
echo "run_1.sh finished successfully."
echo "Results directory: $RESDIR"
