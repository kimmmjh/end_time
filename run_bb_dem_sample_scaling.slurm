#!/bin/bash
#SBATCH --job-name=bbdem_samples2m
#SBATCH --account=m5328_g
#SBATCH --qos=regular
#SBATCH --time=48:00:00
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --signal=B:USR1@1800
#SBATCH --mail-user=ds08152@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
# Preview: BB_DRY_RUN=1 bash run_bb_dem_sample_scaling.slurm
# Resume: sbatch --export=ALL,BB_SCALE_RUN_DIR=$HOME/end_time/resdir_JOB run_bb_dem_sample_scaling.slurm
# Default: BB144 only, finish 3907 then continue with constant LR; chain 48h jobs.
# Stop the chain gracefully: touch "$BB_SCALE_RUN_DIR/STOP_CONTINUOUS"
# One allocation only: export BB_SCALE_AUTO_RESUBMIT=0 before sbatch.
# Original finite experiment: BB_SCALE_CONTINUOUS=0; BB_SCALE_WORKERS=0,1 needs --ntasks-per-node=2.
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
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bb_scale_mpl_${SLURM_JOB_ID:-preview}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/bb_scale_cache_${SLURM_JOB_ID:-preview}"
job_output="$ROOT/resdir_${SLURM_JOB_ID:-preview}"
campaign="${BB_SCALE_RUN_DIR:-$job_output}"
continuous="${BB_SCALE_CONTINUOUS:-1}"
auto_resubmit="${BB_SCALE_AUTO_RESUBMIT:-1}"
worker_spec="${BB_SCALE_WORKERS:-1}"
case "$worker_spec" in
    0|1|0,1) IFS=, read -r -a workers <<< "$worker_spec" ;;
    *) echo 'BB_SCALE_WORKERS must be 0, 1, or 0,1' >&2; exit 1 ;;
esac
if [[ ! "$continuous" =~ ^[01]$ || ! "$auto_resubmit" =~ ^[01]$ ]]; then
    echo 'BB_SCALE_CONTINUOUS and BB_SCALE_AUTO_RESUBMIT must be 0 or 1' >&2
    exit 1
fi
if [[ "${BB_DRY_RUN:-0}" != 1 && "${SLURM_NTASKS:-1}" -lt "${#workers[@]}" ]]; then
    echo "Request --ntasks-per-node=${#workers[@]} for BB_SCALE_WORKERS=$worker_spec" >&2
    exit 1
fi
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
if [[ -f "$campaign/STOP_CONTINUOUS" ]]; then
    echo "Chain stopped by $campaign/STOP_CONTINUOUS; no training or resubmission."
    exit 0
fi
helper="$ROOT/scripts/run_bb_dem_sample_scaling.py"
args=(--source-root "${BB_DEM_SOURCE_ROOT:-$ROOT}" --campaign-root "$campaign"
      --stop-file "$job_output/STOP_REQUESTED")
if [[ "$continuous" == 1 ]]; then
    args+=(--continuous --continuation-lr "${BB_SCALE_CONTINUATION_LR:-1e-5}"
           --continuation-every "${BB_SCALE_CONTINUATION_EVERY:-1000}")
fi
if [[ "${BB_DRY_RUN:-0}" == 1 ]]; then
    for worker in "${workers[@]}"; do
        "$python_bin" "$helper" "${args[@]}" --worker "$worker" --dry-run
    done
    exit 0
fi
# Check the selected exact source configurations before starting workers.
for worker in "${workers[@]}"; do
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
    echo "Campaign: workers=$worker_spec; p=0.002; original target=3907 epochs"
    echo "Persistent experiment directory: $campaign"
    echo "Continuous=$continuous; auto_resubmit=$auto_resubmit; one GPU per selected worker"
    echo "Continuation LR=${BB_SCALE_CONTINUATION_LR:-1e-5}; evaluation interval=${BB_SCALE_CONTINUATION_EVERY:-1000}"
} > "$job_output/job_metadata.txt"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git rev-parse HEAD > "$job_output/git_commit.txt"
    git status --short > "$job_output/git_status.txt"
fi
pids=()
deadline_stop=0
cancelled=0
request_stop() { deadline_stop=1; touch "$job_output/STOP_REQUESTED"; }
cancel_chain() { cancelled=1; touch "$job_output/STOP_REQUESTED" "$campaign/STOP_CONTINUOUS"; }
trap request_stop USR1
trap cancel_chain TERM INT
export SLURM_CPU_BIND=cores
for worker in "${workers[@]}"; do
    srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
        --gpus-per-task=1 --mem=32G "$python_bin" -u "$helper" "${args[@]}" --worker "$worker" \
        > "$job_output/log_worker_${worker}.txt" 2>&1 &
    pids[$worker]="$!"
done
failed=0
paused=0
for worker in "${workers[@]}"; do
    while true; do
        if wait "${pids[$worker]}"; then status=0; else status=$?; fi
        # A signal interrupts bash wait before the srun process exits.
        if ((status > 128)); then
            if kill -0 "${pids[$worker]}" 2>/dev/null; then continue; fi
            # The child may finish between the signal and kill -0. Reap its
            # real exit status rather than mistaking wait's signal for failure.
            if wait "${pids[$worker]}"; then
                status=0
            else
                reaped=$?
                if ((reaped != 127)); then status=$reaped; fi
            fi
        fi
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
    if [[ "$continuous" == 1 && "$auto_resubmit" == 1 && "$deadline_stop" == 1 &&
          "$cancelled" == 0 && ! -f "$campaign/STOP_CONTINUOUS" ]]; then
        export BB_SCALE_RUN_DIR="$campaign" BB_REPO_ROOT="$ROOT"
        export BB_SCALE_WORKERS="$worker_spec" BB_SCALE_CONTINUOUS="$continuous"
        export BB_SCALE_AUTO_RESUBMIT="$auto_resubmit"
        export BB_SCALE_CONTINUATION_LR="${BB_SCALE_CONTINUATION_LR:-1e-5}"
        export BB_SCALE_CONTINUATION_EVERY="${BB_SCALE_CONTINUATION_EVERY:-1000}"
        # Only a successful, graceful pause enables the successor. Never retry errors.
        if next_job=$(sbatch --parsable --dependency="afterok:$SLURM_JOB_ID" --kill-on-invalid-dep=yes \
            --export=ALL --ntasks-per-node="${#workers[@]}" --time=48:00:00 \
            --output="$ROOT/slurm-%x-%j.out" "$job_output/submitted_script.slurm"); then
            printf '%s\n' "$next_job" > "$job_output/next_job_id.txt"
            echo "Next job submitted: $next_job (after successful completion of $SLURM_JOB_ID)"
        else
            echo "Automatic submission failed; checkpoints are saved. Resume manually." >&2
            date -u > "$job_output/resubmit_failed.txt"
            exit 1
        fi
    fi
    # An intentional checkpointed pause is successful, not a training failure.
    exit 0
fi
date -u > "$job_output/completed.txt"
