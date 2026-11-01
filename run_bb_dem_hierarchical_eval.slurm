#!/bin/bash
#SBATCH --job-name=bbdem_hier_eval
#SBATCH --account=m5328_g
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --array=0-4,6
#SBATCH --mail-user=ds08152@gmail.com
#SBATCH --mail-type=END,FAIL,ARRAY_TASKS
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
# Original task numbering: 0/1=D4 BB72/144, 2/3=D6 BB72/144,
# 4=D8 BB72, 6=D12 BB72. Only 24 completed runs are in the manifest.
# Each task evaluates four p values on four GPUs. No optimizer/training runs.
# Preview locally: BB_DRY_RUN=1 bash run_bb_dem_hierarchical_eval.slurm
# Small pilot: sbatch --array=0 --export=ALL,BB_HIER_SHOTS=128 run_bb_dem_hierarchical_eval.slurm

if [[ "${BB_DRY_RUN:-0}" == 1 && -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    for task in 0 1 2 3 4 6; do
        SLURM_ARRAY_TASK_ID="$task" bash "${BASH_SOURCE[0]}"
    done
    exit 0
fi
: "${SLURM_ARRAY_TASK_ID:?Expected an array task; use sbatch or BB_DRY_RUN=1}"
if [[ "${BB_DRY_RUN:-0}" == 1 ]]; then
    ROOT="${BB_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
    python_bin="${BB_PYTHON:-python3}"
else
    : "${SLURM_JOB_ID:?Submit with sbatch}"
    module load python/3.10
    source "$PSCRATCH/envs/nde/bin/activate"
    ROOT="${BB_REPO_ROOT:-$HOME/end_time}"
    python_bin="${BB_PYTHON:-python}"
fi
cd "$ROOT"
ROOT="$PWD"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bb_hier_mpl_${SLURM_JOB_ID:-preview}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/bb_hier_cache_${SLURM_JOB_ID:-preview}"
helper="$ROOT/scripts/evaluate_bb_hierarchical_dem_cnn.py"
manifest="${BB_HIER_MANIFEST:-$ROOT/scripts/manifests/bb_dem_hierarchical_800_epochs.json}"
source_root="${BB_DEM_SOURCE_ROOT:-$ROOT}"
case "$manifest" in /*) ;; *) manifest="$ROOT/$manifest" ;; esac
case "$source_root" in /*) ;; *) source_root="$ROOT/$source_root" ;; esac
args=(--root "$ROOT" --source-root "$source_root"
      --manifest "$manifest" --task "$SLURM_ARRAY_TASK_ID"
      --shots "${BB_HIER_SHOTS:-8192}" --batch-size "${BB_HIER_BATCH_SIZE:-4}"
      --edge-chunk-size "${BB_HIER_EDGE_CHUNK_SIZE:-4096}"
      --confidence "${BB_HIER_CONFIDENCE:-0.9}" --seed "${BB_HIER_SEED:-20261101}"
      --weights "${BB_HIER_WEIGHTS:-best}" --device cuda)
if [[ "${BB_HIER_COMPARE_DIRECT_OSD:-0}" == 1 ]]; then args+=(--compare-direct-osd); fi
if [[ "${BB_HIER_SAVE_CORRECTIONS:-0}" == 1 ]]; then args+=(--save-corrections); fi
if [[ "${BB_DRY_RUN:-0}" == 1 ]]; then
    for index in 0 1 2 3; do
        "$python_bin" "$helper" "${args[@]}" --index "$index" --dry-run
    done
    exit 0
fi

# Check every source before starting expensive worker steps. Full config/graph
# validation happens in the evaluator. Source checkpoints are read-only.
for index in 0 1 2 3; do
    "$python_bin" "$helper" "${args[@]}" --index "$index" --dry-run --require-checkpoint
done
destination="$ROOT/resdir_${SLURM_JOB_ID}_hierarchical"
mkdir "$destination"
cp "$0" "$destination/submitted_script.slurm"
cp "$manifest" "$destination/source_manifest.json"
for file in scripts/evaluate_bb_hierarchical_dem_cnn.py models/_bb_hierarchical_dem_cnn.py \
    models/_bb_dem_cnn.py models/__init__.py src/_direct_osd.py src/_bb_circuit_metrics.py \
    src/_bb_circuit_experiment.py src/bb_circuit_data.py src/bb_dem.py src/bb_stim_utils.py src/bb_code.py; do
    mkdir -p "$destination/source_snapshot/$(dirname "$file")"
    cp "$ROOT/$file" "$destination/source_snapshot/$file"
done
if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git rev-parse HEAD > "$destination/git_commit.txt"
    git status --short > "$destination/git_status.txt"
fi
{
    echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Job ID: $SLURM_JOB_ID"
    echo "Array job/task: ${SLURM_ARRAY_JOB_ID:-none}/$SLURM_ARRAY_TASK_ID"
    echo "Campaign: bb_dem_hierarchical_evaluation_only"
    echo "Source root: $source_root"
    echo "Weights: ${BB_HIER_WEIGHTS:-best}; shots: ${BB_HIER_SHOTS:-8192}"
    echo "Confidence: ${BB_HIER_CONFIDENCE:-0.9}; raw and hierarchical use identical fresh circuit shots"
} > "$destination/job_metadata.txt"
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
for index in 0 1 2 3; do
    worker_args=("${args[@]}" --index "$index" --output "$destination/experiment_$index")
    {
        printf '%q ' "$python_bin" -u "$helper" "${worker_args[@]}"
        echo
    } > "command_exp_${index}.txt"
    srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
        --gpus-per-task=1 --mem=32G "$python_bin" -u "$helper" "${worker_args[@]}" \
        > "log_exp_${index}.txt" 2>&1 &
    pids+=("$!")
done
failures=0
for index in 0 1 2 3; do
    if wait "${pids[$index]}"; then status=0; else status=$?; failures=$((failures+1)); fi
    echo "$status" > "exit_code_exp_${index}.txt"
done
trap - INT TERM
if ((failures)); then
    echo "Failed experiments: $failures/4" > failed.txt
    exit 1
fi
{
    echo "All four evaluation workers completed; source weights unchanged"
    echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > completed.txt
