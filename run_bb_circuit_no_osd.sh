#!/bin/bash

set -uo pipefail

# Usage:
#   bash run_bb_circuit_no_osd.sh GPU_ID [bb72|bb144] [low|high|all] \
#       [legacy|standard|si1000]
#
# Split one code across two GPUs with, for example:
#   bash run_bb_circuit_no_osd.sh 0 bb72 low
#   bash run_bb_circuit_no_osd.sh 1 bb72 high
# Run the paper's Table-II channel with:
#   bash run_bb_circuit_no_osd.sh 0 bb72 all standard

ROOT="${ROOT:-$HOME/end_time}"
GPU_ID="${1:-${GPU_ID:-0}}"
CODE="${2:-bb72}"
P_GRID="${3:-all}"
NOISE_PROFILE="${4:-${BB_CIRCUIT_NOISE_MODEL:-legacy}}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/env/bin/activate}"

EPOCHS="${EPOCHS:-100}"
BATCHES="${BATCHES:-128}"
EVAL_EVERY="${EVAL_EVERY:-10}"
LEARNING_RATE="${LEARNING_RATE:-0.0003}"

if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "Usage: bash $0 GPU_ID [bb72|bb144] [low|high|all] [legacy|standard|si1000]" >&2
    echo "Error: GPU_ID must be a non-negative integer." >&2
    exit 2
fi

case "$CODE" in
    bb72)
        ROUNDS=6
        BATCH_SIZE=16
        EVAL_BATCHES=64
        FINAL_EVAL_BATCHES=256
        SEED_PREFIX=7201
        ;;
    bb144)
        ROUNDS=12
        BATCH_SIZE=8
        EVAL_BATCHES=128
        FINAL_EVAL_BATCHES=512
        SEED_PREFIX=14401
        ;;
    *)
        echo "Usage: bash $0 GPU_ID [bb72|bb144] [low|high|all]" >&2
        echo "Error: unsupported BB code: $CODE" >&2
        exit 2
        ;;
esac

case "$NOISE_PROFILE" in
    legacy|standard|si1000)
        ;;
    *)
        echo "Usage: bash $0 GPU_ID [bb72|bb144] [low|high|all] [legacy|standard|si1000]" >&2
        echo "Error: unsupported BB circuit noise model: $NOISE_PROFILE" >&2
        exit 2
        ;;
esac

case "$P_GRID" in
    low)
        EXPERIMENT_P=(0.001 0.002 0.003 0.004)
        EXPERIMENT_SUFFIX=(001 002 003 004)
        ;;
    high)
        EXPERIMENT_P=(0.005 0.006 0.008 0.010)
        EXPERIMENT_SUFFIX=(005 006 008 010)
        ;;
    all)
        EXPERIMENT_P=(0.001 0.002 0.003 0.004 0.005 0.006 0.008 0.010)
        EXPERIMENT_SUFFIX=(001 002 003 004 005 006 008 010)
        ;;
    *)
        echo "Usage: bash $0 GPU_ID [bb72|bb144] [low|high|all]" >&2
        echo "Error: p grid must be low, high, or all." >&2
        exit 2
        ;;
esac

TOTAL=${#EXPERIMENT_P[@]}
if ((${#EXPERIMENT_SUFFIX[@]} != TOTAL)); then
    echo "Error: experiment arrays have different lengths." >&2
    exit 2
fi
for value in "$EPOCHS" "$BATCHES" "$EVAL_EVERY"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: EPOCHS, BATCHES, and EVAL_EVERY must be positive integers." >&2
        exit 2
    fi
done
if [[ ! "$LEARNING_RATE" =~ ^[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$ ]]; then
    echo "Error: LEARNING_RATE must be a positive number." >&2
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

CURRENT_PYTHON_PID=""
CURRENT_EXPERIMENT=""

cleanup() {
    trap - INT TERM
    echo
    echo "Stopping raw BB circuit Neural-BP runner..."
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
    local p="${EXPERIMENT_P[$index]}"
    local suffix="${EXPERIMENT_SUFFIX[$index]}"
    local seed="${SEED_PREFIX}${suffix}"
    local expdir="$RESDIR/exp_${index}"
    local exit_code
    local measurement_description
    local idle_description
    local -a profile_args=(--bb_circuit_noise_model="$NOISE_PROFILE")
    case "$NOISE_PROFILE" in
        legacy)
            profile_args+=(--measurement_error_rate="$p" --bb_idle_error_rate=0)
            measurement_description="p"
            idle_description="0 (legacy data-only idle disabled)"
            ;;
        standard)
            measurement_description="p"
            idle_description="p on every inactive qubit/tick"
            ;;
        si1000)
            measurement_description="5p"
            idle_description="p/10 gate idle + 2p resonator idle on M/R ticks"
            ;;
    esac
    local -a params=(
        --architecture=bb_neural_bp
        --code="$CODE"
        --noise_model=circuit
        --rounds="$ROUNDS"
        --p="$p"
        "${profile_args[@]}"
        --loss_fn=bb_coset
        --epochs="$EPOCHS"
        --batch_size="$BATCH_SIZE"
        --batches="$BATCHES"
        --eval_batches="$EVAL_BATCHES"
        --eval_every="$EVAL_EVERY"
        --final_eval_batches="$FINAL_EVAL_BATCHES"
        --bp_iterations=12
        --bp_residual_hidden_dim=32
        --bp_orbit_embedding_dim=8
        --bp_parameter_sharing=orbit
        --bp_normalisation=0.625
        --bp_residual_scale=2.0
        --bp_max_relaxation_delta=0.5
        --bp_deep_supervision_weight=0.2
        --bp_gradient_clip=1.0
        --bb_syndrome_loss_weight=1.0
        --bb_logical_loss_weight=1.0
        --bb_pauli_loss_weight=0.1
        --bb_weight_decay=0.0001
        --bb_osd_eval_shots=0
        --bb_osd_method=OSD_0
        --bb_osd_order=0
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
        echo "Runner: run_bb_circuit_no_osd.sh"
        echo "GPU: $GPU_ID"
        echo "Code: $CODE"
        echo "Circuit noise model: $NOISE_PROFILE"
        echo "Noisy rounds: $ROUNDS"
        echo "Base p: $p"
        echo "Measurement error: $measurement_description"
        echo "Idle error: $idle_description"
        echo "Decoder: raw Neural BP2 (no OSD)"
        echo "Checkpoint selection: raw neural paired gain versus vanilla BP2"
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
         "(PID $CURRENT_PYTHON_PID): code=$CODE model=$NOISE_PROFILE p=$p, OSD disabled"
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

{
    echo "Started: $(date)"
    echo "Runner PID: $SCRIPT_PID"
    echo "GPU: $GPU_ID"
    echo "Code: $CODE"
    echo "p grid: $P_GRID"
    echo "Circuit noise model: $NOISE_PROFILE"
    echo "OSD eval shots: 0"
} > "$RESDIR/job_metadata.txt"

echo "Runner: run_bb_circuit_no_osd.sh"
echo "Run script PID: $SCRIPT_PID"
echo "Results directory: $RESDIR"
echo "Physical GPU: $GPU_ID"
echo "Code: $CODE | noisy rounds: $ROUNDS | p grid: $P_GRID | noise: $NOISE_PROFILE"
echo "Decoder: raw Neural BP2; OSD disabled for evaluation and selection"
for index in "${!EXPERIMENT_P[@]}"; do
    echo "  exp_$index: p=q=${EXPERIMENT_P[$index]}"
done

for index in "${!EXPERIMENT_P[@]}"; do
    run_experiment "$index"
    exit_code=$?
    if ((exit_code != 0)); then
        echo "Stopping after experiment $index failed with exit code $exit_code." >&2
        exit "$exit_code"
    fi
done

{
    echo "All $TOTAL experiments completed"
    echo "Finished: $(date)"
    echo "GPU: $GPU_ID"
    echo "Code: $CODE"
    echo "p grid: $P_GRID"
    echo "Decoder: raw Neural BP2 (no OSD)"
} > "$RESDIR/completed.txt"

echo
echo "All raw BB circuit Neural-BP experiments finished successfully."
echo "Results directory: $RESDIR"
