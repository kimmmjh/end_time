#!/bin/bash
#SBATCH --job-name=bbc_base144_high
#SBATCH --account=m5328
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# Preview locally: BB_DRY_RUN=1 bash run_bb_3.slurm
if [[ "${BB_DRY_RUN:-0}" == 1 ]]; then
    ROOT="${BB_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
else
    module load python/3.10
    source "$PSCRATCH/envs/nde/bin/activate"
    ROOT="${BB_REPO_ROOT:-$HOME/end_time}"
fi
cd "$ROOT"
# Standalone launcher: no repository .sh helpers are required.
# Python sources are still read from ROOT when the job starts.
bb_baseline_experiment_args() {
    if (($# != 4)); then
        echo "Usage: bb_baseline_experiment_args code p seed output_directory" >&2
        return 2
    fi
    local code="$1" p="$2" seed="$3" output_directory="$4"
    local rounds resume_arg=""
    case "$code" in
        bb72) rounds=6 ;;
        bb144) rounds=12 ;;
        *) echo "Unknown code: $code" >&2; return 2 ;;
    esac
    if [[ -n "${BB_BASELINE_RESUME_ROOT:-}" ]]; then
        output_directory="$BB_BASELINE_RESUME_ROOT/$output_directory"
        resume_arg="--resume"
    fi
    printf '%s\n' "--code=$code --p=$p --seed=$seed --rounds=$rounds --shots=${BB_BASELINE_SHOTS:-100000} --max-iterations=${BB_BASELINE_ITERATIONS:-1000} --detector-inputs x xz --workers=${BB_BASELINE_WORKERS:-32} --chunk-size=128 $resume_arg --out=$output_directory"
}

BB_CAMPAIGN="bb_library_bp_osd_2026_09_27"
BB_PYTHON_ENTRYPOINT="$ROOT/scripts/evaluate_bb_circuit_baselines.py"

# Paired library BP, BP+OSD-0 and BP+OSD-CS3; X-only and joint X/Z input.
# Paper eight-tick schedule, noisy idles, logical-X memory / logical-Z block LER.
# CPU processes, original DEM priors, max 1000 BP iterations, no rescaling.
# Experiment matrix and resumption: docs/bb_baseline_campaign.md

experiments=(
    "$(bb_baseline_experiment_args bb144 0.005 14401005 exp_0)"
    "$(bb_baseline_experiment_args bb144 0.006 14401006 exp_1)"
    "$(bb_baseline_experiment_args bb144 0.007 14401007 exp_2)"
    "$(bb_baseline_experiment_args bb144 0.008 14401008 exp_3)"
)

BB_EXPERIMENT_LABELS=(
    "bb144_baseline_p0.005_seed14401005"
    "bb144_baseline_p0.006_seed14401006"
    "bb144_baseline_p0.007_seed14401007"
    "bb144_baseline_p0.008_seed14401008"
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
    local cpus_per_task="${SLURM_CPUS_PER_TASK:-32}"
    local entrypoint
    local index params pid exit_code label
    local failures=0
    local -a argv=()
    local -a step_resources=(
        --cpus-per-task="$cpus_per_task"
        --mem=48G
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
    } > "$_BB_RESULT_DIRECTORY/job_metadata.txt"
    cp "$0" "$_BB_RESULT_DIRECTORY/submitted_script.slurm" 2>/dev/null || true
    cp "$entrypoint" "$_BB_RESULT_DIRECTORY/entrypoint_snapshot.py" \
        2>/dev/null || true
    # The active decoder may be uncommitted during a research campaign.
    # Preserve its implementation as well as the exact shell command.
    mkdir -p "$_BB_RESULT_DIRECTORY/source_snapshot/src"
    for params in src/bb_paper_baselines.py src/bb_code.py; do
        cp "$ROOT/$params" "$_BB_RESULT_DIRECTORY/source_snapshot/$params" || return 1
    done
    if command -v git >/dev/null 2>&1 && \
       git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git -C "$ROOT" rev-parse HEAD > "$_BB_RESULT_DIRECTORY/git_commit.txt"
        git -C "$ROOT" status --short > "$_BB_RESULT_DIRECTORY/git_status.txt"
    fi

    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1 MPLCONFIGDIR="$_BB_RESULT_DIRECTORY/mpl_cache"
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
        srun --exclusive --nodes=1 --ntasks=1 \
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
