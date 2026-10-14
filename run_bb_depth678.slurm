#!/bin/bash
#SBATCH --job-name=bbcnn_d678_syndrome_osd0
#SBATCH --account=m5328_g
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --array=0-5
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

# Array tasks: 0=BB72 d6, 1=BB144 d6, 2=BB72 d7, 3=BB144 d7,
#              4=BB72 d8, 5=BB144 d8. Each trains four noise rates on four GPUs.
# Preview every command: BB_DRY_RUN=1 bash run_bb_depth678.slurm
# Submit only BB144: sbatch --array=1,3,5 run_bb_depth678.slurm
if [[ "${BB_DRY_RUN:-0}" == 1 && -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    for task in 0 1 2 3 4 5; do
        SLURM_ARRAY_TASK_ID="$task" bash "${BASH_SOURCE[0]}"
    done
    exit 0
fi
case "${SLURM_ARRAY_TASK_ID:-}" in
    0|1|2|3|4|5) ;;
    *) echo "Expected Slurm array task 0..5 (use sbatch or BB_DRY_RUN=1)." >&2; exit 2 ;;
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
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bb_d678_mpl_${SLURM_JOB_ID:-preview}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/bb_d678_cache_${SLURM_JOB_ID:-preview}"

depth=$((6 + SLURM_ARRAY_TASK_ID / 2))
if ((SLURM_ARRAY_TASK_ID % 2 == 0)); then code=bb72; seed_base=7201000;
else code=bb144; seed_base=14401000; fi
BB_CAMPAIGN="bb_tanner_cnn_depth678_e800_syndrome_osd0"
BB_PYTHON_ENTRYPOINT="$ROOT/main.py"
# Fresh 800-epoch cosine schedule. Validation: 4096 shots every five epochs.
# Final: 65536 shots. OSD is used for evaluation only, including validation;
# raw CNN LER still selects the best checkpoint. No source checkpoint needed.
common="--architecture=bb_tanner_cnn --noise_model=capacity --bb_channel=depolarizing --rounds=1 --measurement_error_rate=0 --bb_idle_error_rate=0 --loss_fn=bb_coset --epochs=800 --batch_size=64 --batches=128 --eval_batches=64 --eval_every=5 --final_eval_batches=1024 --lr=0.0003 --amp_dtype=none --save_model --bb_cnn_width=64 --bb_cnn_gradient_clip=1.0 --bb_syndrome_loss_weight=1.0 --bb_logical_loss_weight=1.0 --bb_pauli_loss_weight=0.1 --bb_weight_decay=0.0001 --bb_cnn_osd=syndrome"
experiments=()
BB_EXPERIMENT_LABELS=()
for rate in 2 4 6 8; do
    p="0.0${rate}"
    seed=$((seed_base + rate * 10))
    experiments+=("$common --code=$code --bb_cnn_depth=$depth --p=$p --seed=$seed")
    BB_EXPERIMENT_LABELS+=("${code}_depth${depth}_e800_p${p}_seed${seed}_syndrome_osd0")
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
        --mem=16G
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
        echo "Code: $code; depth: $depth; OSD: syndrome_osd0_v1"
        echo "Schedule: fresh 800 epochs, cosine LR 3e-4; raw-LER best selection"
    } > "$_BB_RESULT_DIRECTORY/job_metadata.txt"
    cp "$0" "$_BB_RESULT_DIRECTORY/submitted_script.slurm" 2>/dev/null || true
    cp "$entrypoint" "$_BB_RESULT_DIRECTORY/entrypoint_snapshot.py" \
        2>/dev/null || true
    # The active decoder may be uncommitted during a research campaign.
    # Preserve its implementation as well as the exact shell command.
    mkdir -p "$_BB_RESULT_DIRECTORY/source_snapshot/models" \
        "$_BB_RESULT_DIRECTORY/source_snapshot/src"
    for params in models/_bb_tanner_cnn.py src/_bb_tanner_cnn_experiment.py \
        src/_direct_osd.py src/_bb_loss.py src/_bb_metrics.py \
        src/bb_data_generator.py src/bb_code.py main.py; do
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
