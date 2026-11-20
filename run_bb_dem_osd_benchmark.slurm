#!/bin/bash
#SBATCH --job-name=bbdem_osd_check
#SBATCH --account=m5328_g
#SBATCH --qos=regular
#SBATCH --time=04:00:00
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mail-user=ds08152@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm-%x-%j.out

# Short, independent correctness/timing experiment. Never trains or resubmits.
# Preview: BB_DRY_RUN=1 bash run_bb_dem_osd_benchmark.slurm
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
ROOT="$PWD"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bb_osd_mpl_${SLURM_JOB_ID:-preview}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/bb_osd_cache_${SLURM_JOB_ID:-preview}"
campaign="${BB_SCALE_RUN_DIR:-$ROOT/recovered_60828644_rhhDfV}"
output="$ROOT/resdir_${SLURM_JOB_ID:-preview}"
case "${BB_OSD_WORKERS:-0,1}" in
    0|1|0,1) IFS=, read -r -a workers <<< "${BB_OSD_WORKERS:-0,1}" ;;
    *) echo 'BB_OSD_WORKERS must be 0, 1, or 0,1' >&2; exit 1 ;;
esac
args=(--campaign-root "$campaign" --shots "${BB_OSD_SHOTS:-128}"
      --batch-size "${BB_OSD_BATCH_SIZE:-4}" --repeats "${BB_OSD_REPEATS:-2}"
      --confidence "${BB_OSD_CONFIDENCE:-0.9}" --weights "${BB_OSD_WEIGHTS:-latest}"
      --device cuda)
if [[ "${BB_DRY_RUN:-0}" == 1 ]]; then
    for worker in "${workers[@]}"; do
        "$python_bin" scripts/benchmark_bb_dem_osd.py "${args[@]}" --worker "$worker" \
            --output "$output/worker_$worker" --dry-run
    done
    exit 0
fi
if [[ "${SLURM_NTASKS:-1}" -lt "${#workers[@]}" ]]; then
    echo "Need --ntasks-per-node=${#workers[@]}" >&2
    exit 1
fi
mkdir "$output"
cp "${BASH_SOURCE[0]}" "$output/submitted_script.slurm"
for folder in models src scripts; do
    mkdir -p "$output/source_snapshot/$folder"
    cp "$ROOT/$folder/"*.py "$output/source_snapshot/$folder/"
done
{
    echo "Job ID: $SLURM_JOB_ID"
    echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Benchmark only; CPU reference / packed CPU / packed CUDA; no training or automatic resubmission"
    echo "Source campaign: $campaign"
    echo "Workers: ${workers[*]}; shots=${BB_OSD_SHOTS:-128}; confidence=${BB_OSD_CONFIDENCE:-0.9}"
} > "$output/job_metadata.txt"
pids=()
for worker in "${workers[@]}"; do
    srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
        --gpus-per-task=1 --mem=32G "$python_bin" -u scripts/benchmark_bb_dem_osd.py \
        "${args[@]}" --worker "$worker" --output "$output/worker_$worker" \
        > "$output/log_worker_${worker}.txt" 2>&1 &
    pids[$worker]="$!"
done
failed=0
for worker in "${workers[@]}"; do
    if wait "${pids[$worker]}"; then status=0; else status=$?; fi
    echo "$status" > "$output/exit_code_worker_$worker.txt"
    if ((status != 0)); then failed=1; fi
done
if ((failed)); then
    date -u > "$output/failed.txt"
    exit 1
fi
date -u > "$output/completed.txt"
