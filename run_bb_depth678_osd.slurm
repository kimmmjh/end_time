#!/bin/bash
#SBATCH --job-name=bbcnn_d678_residual_osd0
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
# Frozen evaluation of the 24 depth 6/7/8 models; no training.
# One job, four GPUs: each worker evaluates six models sequentially.
# Preview: BB_DRY_RUN=1 bash run_bb_depth678_osd.slurm
# Submit: sbatch run_bb_depth678_osd.slurm
# Reuses best weights and all saved final shots; compares both OSD conventions.
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
# Resolve the six original campaign jobs at the server root or in the archive.
# Prefer the original server directory when both copies exist.
if (($#)); then
    echo "This launcher selects the six depth 6/7/8 jobs automatically; no arguments expected." >&2
    exit 2
fi
inputs=()
archive="$ROOT/results/bb/code_capacity/depolarizing/tanner_cnn/800_epochs/syndrome_osd0"
for spec in bb72:59765840 bb144:59795406 bb72:59892117 bb144:59928152 bb72:59948099 bb144:59765839; do
    code="${spec%%:*}"
    job="${spec##*:}"
    if [[ -d "$ROOT/resdir_$job/outputs" ]]; then
        inputs+=("$ROOT/resdir_$job/outputs")
    elif [[ -d "$archive/$code/resdir_$job/outputs" ]]; then
        inputs+=("$archive/$code/resdir_$job/outputs")
    else
        echo "Missing training outputs for $code job $job (checked server root and archive)." >&2
        exit 2
    fi
done
common=("${inputs[@]}" --latest-only --require-complete --expected-runs 24
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
# Verify the exact campaign before allocating the four evaluation steps.
python - "${inputs[@]}" <<'PY_CHECK'
import json
from pathlib import Path
import sys
from scripts.evaluate_bb_cnn_osd import discover_runs
runs, missing = discover_runs([Path(p) for p in sys.argv[1:]], latest_only=True, require_complete=True)
expected = {(c, d, p) for c in ("bb72", "bb144") for d in (6, 7, 8) for p in (.02, .04, .06, .08)}
seen = []
for run in runs:
    h = json.loads((run / "history.json").read_text())
    c = h["config"]
    seen.append((c["code"], c["cnn_depth"], c["error_rate"]))
    if (c["cnn_width"] != 64 or len(h["train"]) != 800 or h["final"]["shots"] != 65536
            or c.get("osd") != "syndrome_osd0_v1"):
        raise SystemExit(f"Unexpected campaign configuration: {run}")
if len(seen) != 24 or set(seen) != expected or missing:
    raise SystemExit(f"Expected all 24 completed depth 6/7/8 runs; selected={seen}, skipped={missing}")
print("Verified BB72/BB144 x depth 6/7/8 x p=.02/.04/.06/.08; 800 epochs, 65536 shots each.")
PY_CHECK
python -u "$ENTRY" "${common[@]}" --dry-run
if [[ -e "$RESULT" ]]; then echo "Result directory already exists: $RESULT" >&2; exit 1; fi
mkdir "$RESULT"
cp "$0" "$RESULT/submitted_script.slurm"
mkdir -p "$RESULT/source_snapshot/scripts" "$RESULT/source_snapshot/models" "$RESULT/source_snapshot/src"
for file in scripts/evaluate_bb_cnn_osd.py models/_bb_tanner_cnn.py src/_direct_osd.py src/_bb_metrics.py src/bb_code.py; do
    cp "$ROOT/$file" "$RESULT/source_snapshot/$file"
done
{
    echo "Campaign: depth 6/7/8 frozen CNN / raw, residual OSD-0, syndrome OSD-0"
    echo "Training steps: 0; expected runs: 24; full saved banks; best_model_state_dict"
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
echo "All four depth 6/7/8 OSD comparison shards completed" > "$RESULT/completed.txt"
echo "Results: $RESULT/standard_osd0"
