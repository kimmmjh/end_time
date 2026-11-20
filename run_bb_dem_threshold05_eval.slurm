#!/bin/bash
#SBATCH --job-name=bbdem_threshold05
#SBATCH --account=m5328_g
#SBATCH --qos=regular
#SBATCH --time=08:00:00
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mail-user=ds08152@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm-%x-%j.out

# BB72 latest/best and BB144 latest at q>0.5; CPU OSD; no training/resubmission.
set -euo pipefail
if [[ "${BB_DRY_RUN:-0}" == 1 ]]; then
    ROOT="${BB_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
    python_bin="${BB_PYTHON:-python3}"
else
    ROOT="${BB_REPO_ROOT:-$HOME/end_time}"
    : "${SLURM_JOB_ID:?Submit with sbatch}"
    module load python
    source "$PSCRATCH/envs/nde/bin/activate"
    python_bin="${BB_PYTHON:-python}"
fi
cd "$ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bb_threshold_mpl_${SLURM_JOB_ID:-preview}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/bb_threshold_cache_${SLURM_JOB_ID:-preview}"
output="$ROOT/resdir_${SLURM_JOB_ID:-preview}"
args=(--campaign-root "${BB_SCALE_RUN_DIR:-$ROOT/recovered_60828644_rhhDfV}"
      --shots "${BB_THRESHOLD_SHOTS:-8192}"
      --seed "${BB_THRESHOLD_SEED:-20261103}" --device cuda)
if [[ "${BB_DRY_RUN:-0}" == 1 ]]; then
    for worker in 0 1; do
        "$python_bin" scripts/evaluate_bb_dem_threshold05.py "${args[@]}" \
            --worker "$worker" --output "$output/worker_$worker" --dry-run
    done
    exit 0
fi
if [[ "${SLURM_NTASKS:-1}" -lt 2 ]]; then
    echo 'Need --ntasks-per-node=2 for BB72 and BB144' >&2
    exit 1
fi
mkdir "$output"
cp "${BASH_SOURCE[0]}" "$output/submitted_script.slurm"
for folder in models src scripts; do
    mkdir -p "$output/source_snapshot/$folder"
    cp "$ROOT/$folder/"*.py "$output/source_snapshot/$folder/"
done
cp scripts/manifests/bb_dem_sample_scaling.json "$output/source_manifest.json"
{
    echo "Job ID: $SLURM_JOB_ID"
    echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo 'BB72 latest/best; BB144 latest; threshold 0.5; no training or automatic resubmission'
} > "$output/job_metadata.txt"
pids=()
for worker in 0 1; do
    srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
        --gpus-per-task=1 --mem=32G "$python_bin" -u scripts/evaluate_bb_dem_threshold05.py "${args[@]}" \
        --worker "$worker" --output "$output/worker_$worker" \
        > "$output/log_worker_${worker}.txt" 2>&1 &
    pids[$worker]="$!"
done
failed=0
for worker in 0 1; do
    if wait "${pids[$worker]}"; then status=0; else status=$?; fi
    echo "$status" > "$output/exit_code_worker_${worker}.txt"
    if ((status != 0)); then failed=1; fi
done
if ((failed)); then
    date -u > "$output/failed.txt"
    exit 1
fi
date -u > "$output/completed.txt"
