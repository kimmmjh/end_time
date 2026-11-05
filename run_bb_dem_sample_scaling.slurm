#!/bin/bash
#SBATCH --job-name=bbdem_samples2m
#SBATCH --account=m5328_g
#SBATCH --qos=regular
#SBATCH --time=48:00:00
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --signal=B:USR1@1800
#SBATCH --mail-user=ds08152@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
# Preview: BB_DRY_RUN=1 bash run_bb_dem_sample_scaling.slurm
# Resume: sbatch --export=ALL,BB_SCALE_RUN_DIR=$HOME/end_time/resdir_JOB run_bb_dem_sample_scaling.slurm
if [[ "${BB_DRY_RUN:-0}" == 1 ]]; then
    ROOT="${BB_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
    python_bin="${BB_PYTHON:-python3}"
else
    ROOT="${BB_REPO_ROOT:-$HOME/end_time}"
    : "${SLURM_JOB_ID:?Submit with sbatch}"
    module load python/3.10
    source "$PSCRATCH/envs/nde/bin/activate"
    python_bin="${BB_PYTHON:-python}"
fi
cd "$ROOT"
ROOT="$PWD"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bb_scale_mpl_${SLURM_JOB_ID:-preview}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/bb_scale_cache_${SLURM_JOB_ID:-preview}"
job_output="$ROOT/resdir_${SLURM_JOB_ID:-preview}"
campaign="${BB_SCALE_RUN_DIR:-$job_output}"
if [[ -n "${BB_SCALE_RUN_DIR:-}" && ! -d "$campaign" ]]; then
    echo "Resume directory does not exist: $campaign" >&2
    exit 1
fi
if [[ -n "${BB_SCALE_RUN_DIR:-}" ]]; then
    if ! cmp -s "$campaign/source_manifest.json" "$ROOT/scripts/manifests/bb_dem_sample_scaling.json"; then
        echo "Resume directory must contain this campaign's unchanged source_manifest.json" >&2
        exit 1
    fi
fi
helper="$ROOT/scripts/run_bb_dem_sample_scaling.py"
args=(--source-root "${BB_DEM_SOURCE_ROOT:-$ROOT}" --campaign-root "$campaign"
      --stop-file "$job_output/STOP_REQUESTED")
if [[ "${BB_DRY_RUN:-0}" == 1 ]]; then
    for worker in 0 1; do
        "$python_bin" "$helper" "${args[@]}" --worker "$worker" --dry-run
    done
    exit 0
fi
# Check both exact source configurations before reserving worker processes.
for worker in 0 1; do
    "$python_bin" "$helper" "${args[@]}" --worker "$worker" --dry-run --require-checkpoint
done
mkdir "$job_output"
cp "${BASH_SOURCE[0]}" "$job_output/submitted_script.slurm"
cp "$ROOT/scripts/manifests/bb_dem_sample_scaling.json" "$job_output/source_manifest.json"
for folder in models src; do
    mkdir -p "$job_output/source_snapshot/$folder"
    cp "$ROOT/$folder/"*.py "$job_output/source_snapshot/$folder/"
done
mkdir -p "$job_output/source_snapshot/scripts"
for file in run_bb_dem_sample_scaling.py evaluate_bb_hierarchical_dem_cnn.py resume_bb_dem_cnn.py; do
    cp "$ROOT/scripts/$file" "$job_output/source_snapshot/scripts/$file"
done
{
    echo "Job ID: $SLURM_JOB_ID"
    echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Campaign: BB72 D8 and BB144 D6; p=0.002; 2,000,384 samples each"
    echo "Persistent experiment directory: $campaign"
    echo "Two workers use one GPU each; model selection and final evaluation included"
} > "$job_output/job_metadata.txt"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git rev-parse HEAD > "$job_output/git_commit.txt"
    git status --short > "$job_output/git_status.txt"
fi
pids=()
request_stop() { touch "$job_output/STOP_REQUESTED"; }
trap request_stop USR1 TERM INT
export SLURM_CPU_BIND=cores
for worker in 0 1; do
    srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
        --gpus-per-task=1 --mem=32G "$python_bin" -u "$helper" "${args[@]}" --worker "$worker" \
        > "$job_output/log_worker_${worker}.txt" 2>&1 &
    pids+=("$!")
done
failed=0
paused=0
for worker in 0 1; do
    while true; do
        if wait "${pids[$worker]}"; then status=0; else status=$?; fi
        # A signal interrupts bash wait before the srun process exits.
        if ((status > 128)) && kill -0 "${pids[$worker]}" 2>/dev/null; then continue; fi
        break
    done
    echo "$status" > "$job_output/exit_code_worker_${worker}.txt"
    if ((status == 75)); then paused=1; elif ((status != 0)); then failed=1; fi
done
if ((failed)); then
    date -u > "$job_output/failed.txt"
    exit 1
fi
if ((paused)); then
    date -u > "$job_output/interrupted.txt"
    printf 'Resume: sbatch --export=ALL,BB_SCALE_RUN_DIR=%q %q\n' "$campaign" "$ROOT/run_bb_dem_sample_scaling.slurm"
    exit 75
fi
date -u > "$job_output/completed.txt"
