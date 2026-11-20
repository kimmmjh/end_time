#!/bin/bash
#SBATCH --job-name=bbcnn_more_1
#SBATCH --account=m5328_g
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# Preview locally: BB_DRY_RUN=1 bash run_bb_1.slurm
if [[ "${BB_DRY_RUN:-0}" == 1 ]]; then
    ROOT="${BB_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
else
    module load python/3.10
    source "$PSCRATCH/envs/nde/bin/activate"
    ROOT="${BB_REPO_ROOT:-$HOME/end_time}"
fi
cd "$ROOT"
export BB_REPO_ROOT="$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

# Continue selected depth-4/5 models from latest model.pt, with AdamW/RNG/best
# preserved. New cosine phase starts at 1e-4. Only exp_0 in job 0 is recovery.
# All source checkpoints are validated before any srun starts.
# Raw-validation selection; old/new best share 65536 fresh final shots.
# No source results are overwritten. Details: docs/bb_tanner_cnn_more_training.md
BB_CAMPAIGN="bb_tanner_cnn_depth45_resume800_2026_10"
BB_PYTHON_ENTRYPOINT="$ROOT/scripts/continue_bb_tanner_cnn.py"

experiments=(
    "--source-job=59122449 --code=bb72 --depth=5 --p=0.04 --seed=7201040 --expected-epochs=400 --target-epochs=800 --lr=0.0001"
    "--source-job=59122449 --code=bb72 --depth=5 --p=0.06 --seed=7201060 --expected-epochs=400 --target-epochs=800 --lr=0.0001"
    "--source-job=59122448 --code=bb144 --depth=4 --p=0.04 --seed=14401040 --expected-epochs=400 --target-epochs=800 --lr=0.0001"
    "--source-job=59122448 --code=bb144 --depth=4 --p=0.06 --seed=14401060 --expected-epochs=400 --target-epochs=800 --lr=0.0001"
)

BB_EXPERIMENT_LABELS=(
    "bb72_d5_p0.04_seed7201040_400to800"
    "bb72_d5_p0.06_seed7201060_400to800"
    "bb144_d4_p0.04_seed14401040_400to800"
    "bb144_d4_p0.06_seed14401060_400to800"
)

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

    # Validate all four source checkpoints before starting any training step.
    for params in "$@"; do
        read -r -a argv <<< "$params"
        python -u "$entrypoint" "${argv[@]}" --dry-run
    done

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
        echo "Resume source jobs: see experiments.tsv"
        echo "Epochs: 400 -> 800; recovery only: 114 -> 400; LR restart 1e-4"
        echo "Resume source override: ${BB_CNN_SOURCE_ROOT:-automatic}"
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
