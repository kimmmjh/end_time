#!/bin/bash
#SBATCH --job-name=bbcnn_latest_standard_osd0
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
# Evaluate the latest completed model per configuration, without training.
# Submit after run_bb_0 with --dependency=afterok:JOB_ID to include its 800-epoch result.
if [[ "${BB_DRY_RUN:-0}" == 1 ]]; then
    ROOT="${BB_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
else
    module load python/3.10
    source "$PSCRATCH/envs/nde/bin/activate"
    ROOT="${BB_REPO_ROOT:-$HOME/end_time}"
fi
cd "$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bb_osd_mpl_${SLURM_JOB_ID:-preview}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/bb_osd_cache_${SLURM_JOB_ID:-preview}"
ENTRY="$ROOT/scripts/evaluate_bb_cnn_osd.py"
RESULT="$ROOT/resdir_${SLURM_JOB_ID:-DRY_RUN}"
# Optional explicit directories are supported as script arguments. Default: root
# resdir_* plus results/bb/code_capacity/depolarizing/tanner_cnn (same as local).
inputs=("$@")
if ((${#inputs[@]} == 0)); then inputs=(--auto-roots); fi
common=("${inputs[@]}" --latest-only --require-complete
        --expected-runs "${BB_OSD_EXPECTED_RUNS:-44}"
        --output-root "$RESULT/standard_osd0" --device cuda:0)
if [[ "${BB_DRY_RUN:-0}" == 1 ]]; then
    for shard in 0 1 2 3; do
        printf 'python -u %q ' "$ENTRY"
        printf '%q ' "${common[@]}" --shard-count 4 --shard-index "$shard"
        echo
    done
    exit 0
fi
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
# Check the full selected set before allocating the four evaluation steps.
python -u "$ENTRY" "${common[@]}" --dry-run
if [[ -e "$RESULT" ]]; then echo "Result directory already exists: $RESULT" >&2; exit 1; fi
mkdir "$RESULT"
cp "$0" "$RESULT/submitted_script.slurm"
mkdir -p "$RESULT/source_snapshot/scripts" "$RESULT/source_snapshot/models" "$RESULT/source_snapshot/src"
for file in scripts/evaluate_bb_cnn_osd.py models/_bb_tanner_cnn.py src/_direct_osd.py src/_bb_metrics.py src/bb_code.py; do
    cp "$ROOT/$file" "$RESULT/source_snapshot/$file"
done
{
    echo "Campaign: latest frozen CNN / raw, residual OSD-0, syndrome OSD-0"
    echo "Training steps: 0; expected latest runs: ${BB_OSD_EXPECTED_RUNS:-44}"
    echo "Job: $SLURM_JOB_ID; started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$RESULT/job_metadata.txt"
pids=()
stop_steps() {
    trap - INT TERM
    for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
    echo "Interrupted" > "$RESULT/interrupted.txt"
    exit 130
}
trap stop_steps INT TERM
for shard in 0 1 2 3; do
    printf '%q ' python -u "$ENTRY" "${common[@]}" --shard-count 4 --shard-index "$shard" > "$RESULT/command_shard_${shard}.txt"
    srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
        --mem=16G --gpus-per-task=1 \
        python -u "$ENTRY" "${common[@]}" --shard-count 4 --shard-index "$shard" \
        > "$RESULT/log_shard_${shard}.txt" 2>&1 &
    pids+=("$!")
done
failures=0
for shard in 0 1 2 3; do
    if wait "${pids[$shard]}"; then code=0; else code=$?; failures=$((failures+1)); fi
    echo "$code" > "$RESULT/exit_code_shard_${shard}.txt"
done
trap - INT TERM
if ((failures)); then echo "$failures evaluation shards failed" > "$RESULT/failed.txt"; exit 1; fi
echo "All four standard-OSD evaluation shards completed" > "$RESULT/completed.txt"
echo "Results: $RESULT/standard_osd0"
