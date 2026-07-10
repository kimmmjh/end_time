#!/bin/bash
#SBATCH --job-name=threshold_large_L9
#SBATCH --account=m5328_g
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --array=0-4%4

module load python/3.10
source $PSCRATCH/envs/nde/bin/activate
cd $HOME/end_time

# Large NN: 6.15M parameters versus 2.11M for channels=64/depths=3.
# Per epoch: 32 * 2048 = 65536 training samples.
# Final threshold point: 32 * 4096 = 131072 evaluation samples.
experiments=(
    "--noise_model=phenomenological --measurement_error_rate=0.0100 --p=0.0100 --loss_fn=ce --epochs=300 --batch_size=32 --batches=2048 --eval_batches=512 --final_eval_batches=4096 --L=9 --channels 96 96 96 --depths 4 4 4 --lr=0.0003 --amp_dtype=bf16 --seed=39010 --save_model"
    "--noise_model=phenomenological --measurement_error_rate=0.0125 --p=0.0125 --loss_fn=ce --epochs=300 --batch_size=32 --batches=2048 --eval_batches=512 --final_eval_batches=4096 --L=9 --channels 96 96 96 --depths 4 4 4 --lr=0.0003 --amp_dtype=bf16 --seed=39012 --save_model"
    "--noise_model=phenomenological --measurement_error_rate=0.0150 --p=0.0150 --loss_fn=ce --epochs=300 --batch_size=32 --batches=2048 --eval_batches=512 --final_eval_batches=4096 --L=9 --channels 96 96 96 --depths 4 4 4 --lr=0.0003 --amp_dtype=bf16 --seed=39015 --save_model"
    "--noise_model=phenomenological --measurement_error_rate=0.0175 --p=0.0175 --loss_fn=ce --epochs=300 --batch_size=32 --batches=2048 --eval_batches=512 --final_eval_batches=4096 --L=9 --channels 96 96 96 --depths 4 4 4 --lr=0.0003 --amp_dtype=bf16 --seed=39017 --save_model"
    "--noise_model=phenomenological --measurement_error_rate=0.0200 --p=0.0200 --loss_fn=ce --epochs=300 --batch_size=32 --batches=2048 --eval_batches=512 --final_eval_batches=4096 --L=9 --channels 96 96 96 --depths 4 4 4 --lr=0.0003 --amp_dtype=bf16 --seed=39020 --save_model"
)

currdir=$PWD
dirname="resdir_$SLURM_ARRAY_JOB_ID"
mkdir -p "$dirname"
cd "$dirname"
export SLURM_CPU_BIND="cores"

params="${experiments[$SLURM_ARRAY_TASK_ID]}"
echo "Starting Experiment $SLURM_ARRAY_TASK_ID: $params"
srun --ntasks=1 --mem=32G \
    python "$currdir/main.py" $params \
    > "log_exp_${SLURM_ARRAY_TASK_ID}.txt" 2>&1
