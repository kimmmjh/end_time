#!/bin/bash
# Shared four-GPU runner for the two Nov 2 launchers.
set -euo pipefail
mode="$1"
submitted_script="$2"
ROOT="$3"
cd "$ROOT"
ROOT="$PWD"
if [[ "${BB_DRY_RUN:-0}" == 1 ]]; then
    python_bin="${BB_PYTHON:-python3}"
else
    : "${SLURM_JOB_ID:?Submit with sbatch}"
    : "${SLURM_ARRAY_TASK_ID:?Submit the array launcher}"
    module load python/3.10
    source "$PSCRATCH/envs/nde/bin/activate"
    python_bin="${BB_PYTHON:-python}"
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bb_followup_mpl_${SLURM_JOB_ID:-preview}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/bb_followup_cache_${SLURM_JOB_ID:-preview}"
helper="$ROOT/scripts/run_bb_dem_followup.py"
manifest="${BB_FOLLOWUP_MANIFEST:-$ROOT/scripts/manifests/bb_dem_followup_2026_11_02.json}"
args=(--root "$ROOT" --source-root "${BB_DEM_SOURCE_ROOT:-$ROOT}" --manifest "$manifest" --mode "$mode"
      --target-epochs "${BB_DEM_TARGET_EPOCHS:-1000}" --learning-rate "${BB_DEM_RESTART_LR:-0.00003}"
      --shots "${BB_HIER_SHOTS:-8192}" --seed "${BB_HIER_SEED:-20261101}"
      --confidence "${BB_HIER_CONFIDENCE:-0.9}")
count=$("$python_bin" "$helper" "${args[@]}" --count)
if [[ "${BB_DRY_RUN:-0}" == 1 && -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    for ((worker=0; worker<count; worker++)); do
        "$python_bin" "$helper" "${args[@]}" --worker "$worker" --dry-run
    done
    exit 0
fi
task="${SLURM_ARRAY_TASK_ID:?Expected an array task}"
if [[ ! "$task" =~ ^[0-9]+$ ]] || ((task * 4 >= count)); then
    echo "Array task outside campaign range: $task" >&2
    exit 1
fi
worker_ids=()
for ((worker=task*4; worker<(task+1)*4 && worker<count; worker++)); do
    worker_ids+=("$worker")
done
if [[ "${BB_DRY_RUN:-0}" == 1 ]]; then
    for worker in "${worker_ids[@]}"; do
        "$python_bin" "$helper" "${args[@]}" --worker "$worker" --dry-run
    done
    exit 0
fi
for worker in "${worker_ids[@]}"; do
    "$python_bin" "$helper" "${args[@]}" --worker "$worker" --dry-run --require-checkpoint
done
# Keep result directory names uniform; describe the task in job_metadata.txt.
destination="$ROOT/resdir_${SLURM_JOB_ID}"
mkdir "$destination"
cp "$submitted_script" "$destination/submitted_script.slurm"
cp "$manifest" "$destination/source_manifest.json"
for file in main.py scripts/run_bb_dem_followup.sh scripts/run_bb_dem_followup.py \
    scripts/resume_bb_dem_cnn.py scripts/evaluate_bb_hierarchical_dem_cnn.py \
    models/__init__.py models/_bb_dem_cnn.py models/_bb_hierarchical_dem_cnn.py \
    src/_bb_dem_cnn_experiment.py src/_bb_circuit_experiment.py src/_bb_circuit_loss.py \
    src/_bb_circuit_metrics.py src/_direct_osd.py src/bb_circuit_data.py src/bb_dem.py \
    src/bb_stim_utils.py src/bb_code.py; do
    mkdir -p "$destination/source_snapshot/$(dirname "$file")"
    cp "$ROOT/$file" "$destination/source_snapshot/$file"
done
{
    echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Job ID: $SLURM_JOB_ID"
    echo "Array job/task: ${SLURM_ARRAY_JOB_ID:-none}/$task"
    echo "Campaign: bb_dem_followup_2026_11_02; mode: $mode"
    echo "Workers: ${worker_ids[*]}; four GPUs reserved, unused tail slots stay idle"
    echo "Target epochs: ${BB_DEM_TARGET_EPOCHS:-1000}; restart LR: ${BB_DEM_RESTART_LR:-0.00003}"
    echo "Evaluation: best and latest; shots per checkpoint: ${BB_HIER_SHOTS:-8192}"
} > "$destination/job_metadata.txt"
if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git rev-parse HEAD > "$destination/git_commit.txt"
    git status --short > "$destination/git_status.txt"
fi
cd "$destination"
export SLURM_CPU_BIND=cores
pids=()
stop_steps() {
    trap - INT TERM
    for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
    date -u > interrupted.txt
    exit 130
}
trap stop_steps INT TERM
for worker in "${worker_ids[@]}"; do
    worker_args=("${args[@]}" --worker "$worker" --output "$destination/worker_$worker")
    { printf '%q ' "$python_bin" -u "$helper" "${worker_args[@]}"; echo; } > "command_worker_${worker}.txt"
    srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
        --gpus-per-task=1 --mem=32G "$python_bin" -u "$helper" "${worker_args[@]}" \
        > "log_worker_${worker}.txt" 2>&1 &
    pids+=("$!")
done
failures=0
for slot in "${!worker_ids[@]}"; do
    if wait "${pids[$slot]}"; then status=0; else status=$?; failures=$((failures+1)); fi
    echo "$status" > "exit_code_worker_${worker_ids[$slot]}.txt"
done
trap - INT TERM
if ((failures)); then
    echo "Failed workers: $failures/${#worker_ids[@]}" > failed.txt
    exit 1
fi
date -u > completed.txt
