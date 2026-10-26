#!/bin/bash
#SBATCH --job-name=bbdem_d46812_standard
#SBATCH --account=m5328_g
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --array=0-7
#SBATCH --mail-user=ds08152@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

# Array tasks: 0/1 = BB72/BB144 depth 4; 2/3 = depth 6;
#              4/5 = depth 8; 6/7 = depth 12.
# Each task runs p=.001/.002/.004/.006 on four GPUs (32 experiments total).
# END/FAIL email applies to the whole array, not each experiment.
# Preview: BB_DRY_RUN=1 bash run_bb_dem_depth46812.slurm
# BB144 only: sbatch --array=1,3,5,7 run_bb_dem_depth46812.slurm
if [[ "${BB_DRY_RUN:-0}" == 1 && -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    for task in 0 1 2 3 4 5 6 7; do
        SLURM_ARRAY_TASK_ID="$task" bash "${BASH_SOURCE[0]}"
    done
    exit 0
fi
case "${SLURM_ARRAY_TASK_ID:-}" in
    0|1|2|3|4|5|6|7) ;;
    *) echo "Expected Slurm array task 0..7 (use sbatch or BB_DRY_RUN=1)." >&2; exit 2 ;;
esac
if [[ "${BB_DRY_RUN:-0}" == 1 ]]; then
    ROOT="${BB_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
else
    : "${SLURM_JOB_ID:?Submit this script with sbatch}"
    module load python/3.10
    source "$PSCRATCH/envs/nde/bin/activate"
    ROOT="${BB_REPO_ROOT:-$HOME/end_time}"
fi
cd "$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bb_dem_d46812_mpl_${SLURM_JOB_ID:-preview}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/bb_dem_d46812_cache_${SLURM_JOB_ID:-preview}"

depths=(4 6 8 12)
depth="${depths[$((SLURM_ARRAY_TASK_ID / 2))]}"
if ((SLURM_ARRAY_TASK_ID % 2 == 0)); then code=bb72; rounds=6; seed_base=7202000;
else code=bb144; rounds=12; seed_base=14402000; fi
batch_size="${BB_DEM_BATCH_SIZE:-4}"
edge_chunk="${BB_DEM_EDGE_CHUNK_SIZE:-4096}"
osd_shots="${BB_DEM_OSD_SHOTS:-0}"
for value in "$batch_size" "$edge_chunk" "$osd_shots"; do
    if [[ ! "$value" =~ ^(0|[1-9][0-9]*)$ ]]; then
        echo "Batch/chunk/OSD overrides must be nonnegative decimal integers." >&2; exit 2
    fi
done
if ((batch_size < 1 || batch_size > 512 || 512 % batch_size != 0 || edge_chunk < 1)); then
    echo "Batch size must be a positive divisor of 512; edge chunk must be positive." >&2; exit 2
fi
# Keep shot budgets fixed when changing batch size for GPU memory.
train_batches=$((512 / batch_size))
eval_batches=$((1024 / batch_size))
final_batches=$((8192 / batch_size))
BB_CAMPAIGN="bb_dem_cnn_depth46812_e800_standard"
BB_PYTHON_ENTRYPOINT="$ROOT/main.py"
# Train on latent DEM samples; validate/test on independent circuit samples.
# Standard circuit noise, noisy rounds=d, perfect reference/closing cycles.
# 800 epochs x 512 shots; validate every 5 epochs on 1024 shots; final 8192.
# Raw success selects best weights. Residual OSD is opt-in on a paired subset
# because the existing dense CPU solver is expensive for these large DEMs.
common="--architecture=bb_dem_cnn --noise_model=circuit --bb_circuit_noise_model=standard --rounds=$rounds --loss_fn=bb_coset --epochs=800 --batch_size=$batch_size --batches=$train_batches --eval_batches=$eval_batches --eval_every=5 --final_eval_batches=$final_batches --lr=0.0003 --amp_dtype=none --save_model --bb_cnn_width=64 --bb_cnn_gradient_clip=1.0 --bb_syndrome_loss_weight=1.0 --bb_logical_loss_weight=1.0 --bb_pauli_loss_weight=0.1 --bb_weight_decay=0.0001 --edge_chunk_size=$edge_chunk --bb_osd_eval_shots=$osd_shots"
experiments=()
BB_EXPERIMENT_LABELS=()
for rate in 1 2 4 6; do
    p="0.00${rate}"
    seed=$((seed_base + rate))
    experiments+=("$common --code=$code --bb_cnn_depth=$depth --p=$p --seed=$seed")
    BB_EXPERIMENT_LABELS+=("${code}_dem_depth${depth}_e800_p${p}_r${rounds}_seed${seed}_osd${osd_shots}")
done

# Launch/logging code is embedded so sbatch captures it with this script.
_BB_STEP_PIDS=()
_BB_RESULT_DIRECTORY=""

_bb_stop_steps() {
    local pid

    trap - INT TERM
    echo
    echo "Stopping BB Slurm steps..."

    for pid in "${_BB_STEP_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    for pid in "${_BB_STEP_PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done

    if [[ -n "$_BB_RESULT_DIRECTORY" ]]; then
        {
            echo "Interrupted: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
            echo "Job ID: ${SLURM_JOB_ID:-unknown}"
        } > "$_BB_RESULT_DIRECTORY/interrupted.txt"
    fi

    exit 130
}

run_bb_experiments() {
    local experiment_count="$#"
    local cpus_per_task="${SLURM_CPUS_PER_TASK:-16}"
    local gpus_per_task="${BB_GPUS_PER_TASK:-1}"
    local entrypoint
    local index params pid exit_code label
    local failures=0
    local -a argv=()
    local -a step_resources=(
        --cpus-per-task="$cpus_per_task"
        --mem=32G
    )

    if [[ -z "${ROOT:-}" ]]; then
        echo "Error: ROOT must be set before calling run_bb_experiments." >&2
        return 2
    fi
    entrypoint="$BB_PYTHON_ENTRYPOINT"
    if ((experiment_count < 1 || experiment_count > 4)); then
        echo "Error: expected 1-4 experiments, got $experiment_count." >&2
        return 2
    fi
    if [[ ! "$gpus_per_task" =~ ^[0-9]+$ ]]; then
        echo "Error: BB_GPUS_PER_TASK must be a non-negative integer." >&2
        return 2
    fi
    if ((gpus_per_task > 0)); then
        step_resources+=(--gpus-per-task="$gpus_per_task")
    fi
    if [[ ! -f "$entrypoint" ]]; then
        echo "Error: Python entrypoint not found: $entrypoint" >&2
        return 1
    fi

    if [[ "${BB_DRY_RUN:-0}" == 1 ]]; then
        for params in "$@"; do
            argv=()
            read -r -a argv <<< "$params"
            printf 'python -u %q ' "$entrypoint"
            printf '%q ' "${argv[@]}"
            echo
        done
        return 0
    fi
    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
        echo "Error: this launcher must run inside a Slurm allocation." >&2
        return 2
    fi

    _BB_RESULT_DIRECTORY="$ROOT/resdir_${SLURM_JOB_ID}"
    if [[ -e "$_BB_RESULT_DIRECTORY" ]]; then
        echo "Error: result directory already exists: $_BB_RESULT_DIRECTORY" >&2
        return 1
    fi
    mkdir "$_BB_RESULT_DIRECTORY" || return 1

    {
        echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "Job ID: $SLURM_JOB_ID"
        echo "Job name: ${SLURM_JOB_NAME:-unknown}"
        echo "Node list: ${SLURM_NODELIST:-unknown}"
        echo "Experiments: $experiment_count"
        echo "Campaign: ${BB_CAMPAIGN:-unspecified}"
        echo "Array job/task: ${SLURM_ARRAY_JOB_ID:-none}/${SLURM_ARRAY_TASK_ID}"
        echo "Code: $code; depth: $depth; noisy rounds: $rounds; circuit noise: standard"
        echo "Batch: $batch_size; edge chunk: $edge_chunk; gradient checkpointing: on"
        echo "Training shots/epoch: 512; validation: 1024; final: 8192"
        echo "Residual OSD paired subset requested per evaluation: $osd_shots (0=disabled)"
        echo "Schedule: fresh 800 epochs, cosine LR 3e-4; raw detector-and-observable success best selection"
    } > "$_BB_RESULT_DIRECTORY/job_metadata.txt"
    cp "$0" "$_BB_RESULT_DIRECTORY/submitted_script.slurm" 2>/dev/null || true
    cp "$entrypoint" "$_BB_RESULT_DIRECTORY/entrypoint_snapshot.py" \
        2>/dev/null || true
    # The active decoder may be uncommitted during a research campaign.
    # Preserve its implementation as well as the exact shell command.
    mkdir -p "$_BB_RESULT_DIRECTORY/source_snapshot/models" \
        "$_BB_RESULT_DIRECTORY/source_snapshot/src"
    for params in models/_bb_dem_cnn.py src/_bb_dem_cnn_experiment.py \
        src/_direct_osd.py src/_bb_circuit_loss.py src/_bb_circuit_metrics.py \
        src/_bb_circuit_experiment.py src/bb_circuit_data.py src/bb_dem.py \
        src/bb_stim_utils.py src/bb_code.py main.py; do
        cp "$ROOT/$params" "$_BB_RESULT_DIRECTORY/source_snapshot/$params" || return 1
    done
    if command -v git >/dev/null 2>&1 && \
       git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git -C "$ROOT" rev-parse HEAD > "$_BB_RESULT_DIRECTORY/git_commit.txt"
        git -C "$ROOT" status --short > "$_BB_RESULT_DIRECTORY/git_status.txt"
    fi

    cd "$_BB_RESULT_DIRECTORY" || return 1
    export SLURM_CPU_BIND="cores"
    _BB_STEP_PIDS=()
    trap _bb_stop_steps INT TERM

    index=0
    printf 'experiment_index\tlabel\targuments\n' > experiments.tsv
    for params in "$@"; do
        argv=()
        read -r -a argv <<< "$params"
        label="${BB_EXPERIMENT_LABELS[$index]:-experiment_$index}"
        printf '%s\t%s\t%s\n' "$index" "$label" "$params" >> experiments.tsv

        {
            printf 'python -u %q ' "$entrypoint"
            printf '%q ' "${argv[@]}"
            echo
        } > "command_exp_${index}.txt"

        echo "Starting experiment $index ($label): $params"
        srun --exclusive --exact --nodes=1 --ntasks=1 \
            "${step_resources[@]}" \
            python -u "$entrypoint" "${argv[@]}" \
            > "log_exp_${index}.txt" 2>&1 &
        _BB_STEP_PIDS+=("$!")
        index=$((index + 1))
    done

    for index in "${!_BB_STEP_PIDS[@]}"; do
        pid="${_BB_STEP_PIDS[$index]}"
        if wait "$pid"; then
            exit_code=0
        else
            exit_code=$?
            failures=$((failures + 1))
        fi

        echo "$exit_code" > "exit_code_exp_${index}.txt"
        {
            echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
            echo "Exit code: $exit_code"
        } > "finished_exp_${index}.txt"
        echo "Experiment $index finished with exit code $exit_code."
    done

    trap - INT TERM
    if ((failures > 0)); then
        {
            echo "Failed experiments: $failures/$experiment_count"
            echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        } > failed.txt
        echo "Error: $failures BB experiment(s) failed." >&2
        return 1
    fi

    {
        echo "All $experiment_count experiments completed"
        echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } > completed.txt
    echo "All BB experiments completed: $_BB_RESULT_DIRECTORY"
}

run_bb_experiments "${experiments[@]}"
