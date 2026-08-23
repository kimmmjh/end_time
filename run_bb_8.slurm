#!/bin/bash
#SBATCH --job-name=bb72_classical
#SBATCH --account=m5328
#SBATCH --qos=regular
#SBATCH --time=04:00:00
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --output=slurm-%x-%j.out

set -uo pipefail

module load python/3.10 || exit 1
source "$PSCRATCH/envs/nde/bin/activate" || exit 1
ROOT="$HOME/end_time"
cd "$ROOT" || exit 1

# These decoders are CPU-only. Each p task runs CSS-split BP+OSD-0,
# BP+OSD-CS-7, and BP+LSD-0 on exactly the same sampled error bank.
python -c 'from importlib.metadata import version; assert version("ldpc") == "2.4.1", version("ldpc")' || exit 1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
BB_GPUS_PER_TASK=0
BB_PYTHON_ENTRYPOINT="$ROOT/scripts/bb_classical_baselines.py"
source "$ROOT/scripts/run_bb_slurm_batch.sh" || exit 1

SHOTS="${BB_CLASSICAL_SHOTS:-131072}"
COMMON_ARGS="--code=bb72 --shots=$SHOTS --seed=20260823 --osd_max_iter=72 --lsd_max_iter=72 --bp_method=minimum_sum --osd_ms_scaling_factor=0.625 --lsd_ms_scaling_factor=0.625 --schedule=parallel --omp_threads=1 --warmup_shots=8 --progress_every=10000"

experiments=(
    "$COMMON_ARGS --p=0.04 --output=bb72_p004_classical.csv --save_test_bank=bb72_p004_test_bank.npz"
    "$COMMON_ARGS --p=0.06 --output=bb72_p006_classical.csv --save_test_bank=bb72_p006_test_bank.npz"
    "$COMMON_ARGS --p=0.08 --output=bb72_p008_classical.csv --save_test_bank=bb72_p008_test_bank.npz"
    "$COMMON_ARGS --p=0.10 --output=bb72_p010_classical.csv --save_test_bank=bb72_p010_test_bank.npz"
)

run_bb_experiments "${experiments[@]}"
