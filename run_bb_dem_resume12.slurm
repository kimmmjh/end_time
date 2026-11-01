#!/bin/bash
#SBATCH --job-name=bbdem_d12_finish800
#SBATCH --account=m5328_g
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mail-user=ds08152@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
# Resume the latest BB144 D12 checkpoints, all four p values, to 800 total epochs.
# Server preview (activate the training environment first):
# BB_DRY_RUN=1 bash run_bb_dem_resume12.slurm
ROOT="${BB_REPO_ROOT:-$HOME/end_time}"
depth=12
source_job="${BB_DEM_SOURCE_JOB:-$ROOT/resdir_60428753}"
if [[ "${BB_DRY_RUN:-0}" != 1 ]]; then
    : "${SLURM_JOB_ID:?Submit with sbatch}"
    module load python/3.10
    source "$PSCRATCH/envs/nde/bin/activate"
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bb_dem_resume_mpl_${SLURM_JOB_ID:-preview}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/bb_dem_resume_cache_${SLURM_JOB_ID:-preview}"
helper="$ROOT/scripts/resume_bb_dem_cnn.py"
args=(--root "$ROOT" --depth "$depth" --source-array "${BB_DEM_SOURCE_ARRAY:-60310800}" --target-epochs 800)
if [[ -n "$source_job" ]]; then args+=(--source-job "$source_job"); fi
if [[ "${BB_DRY_RUN:-0}" == 1 ]]; then
    for index in 0 1 2 3; do python "$helper" "${args[@]}" --index "$index" --dry-run; done
    exit 0
fi

destination="$ROOT/resdir_${SLURM_JOB_ID}"
mkdir "$destination"
cp "$0" "$destination/submitted_script.slurm"
mkdir -p "$destination/source_snapshot/models" "$destination/source_snapshot/src" "$destination/source_snapshot/scripts"
for file in main.py models/_bb_dem_cnn.py src/_bb_dem_cnn_experiment.py src/_direct_osd.py \
    src/_bb_circuit_loss.py src/_bb_circuit_metrics.py src/_bb_circuit_experiment.py \
    src/bb_circuit_data.py src/bb_dem.py src/bb_stim_utils.py src/bb_code.py scripts/resume_bb_dem_cnn.py; do
    cp "$ROOT/$file" "$destination/source_snapshot/$file"
done
{
    echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Job ID: $SLURM_JOB_ID"
    echo "Campaign: bb_dem_cnn_resume_to_800"
    echo "Code: bb144; depth: $depth; target total epochs: 800"
    echo "Source array: ${BB_DEM_SOURCE_ARRAY:-60310800}; source override: ${source_job:-none}"
    echo "Schedule: new cosine from saved optimizer LR; independent sampler phase"
} > "$destination/job_metadata.txt"
cd "$destination"
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
    srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
        --gpus-per-task=1 --mem=32G python -u "$helper" "${args[@]}" --index "$index" \
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
    echo "All 4 resume workers completed (already-completed sources may be skipped)"
    echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > completed.txt
