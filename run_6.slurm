#!/bin/bash
#SBATCH --job-name=threshold_large_L15
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

# Per epoch: 8 * 4096 = 32768 training samples.
# Final threshold point: 8 * 16384 = 131072 evaluation samples.
experiments=(
    "--noise_model=phenomenological --measurement_error_rate=0.0100 --p=0.0100 --loss_fn=ce --epochs=300 --batch_size=8 --batches=4096 --eval_batches=2048 --final_eval_batches=16384 --L=15 --channels 96 96 96 --depths 4 4 4 --lr=0.0003 --amp_dtype=bf16 --seed=615010 --save_model"
    "--noise_model=phenomenological --measurement_error_rate=0.0125 --p=0.0125 --loss_fn=ce --epochs=300 --batch_size=8 --batches=4096 --eval_batches=2048 --final_eval_batches=16384 --L=15 --channels 96 96 96 --depths 4 4 4 --lr=0.0003 --amp_dtype=bf16 --seed=615012 --save_model"
    "--noise_model=phenomenological --measurement_error_rate=0.0150 --p=0.0150 --loss_fn=ce --epochs=300 --batch_size=8 --batches=4096 --eval_batches=2048 --final_eval_batches=16384 --L=15 --channels 96 96 96 --depths 4 4 4 --lr=0.0003 --amp_dtype=bf16 --seed=615015 --save_model"
    "--noise_model=phenomenological --measurement_error_rate=0.0175 --p=0.0175 --loss_fn=ce --epochs=300 --batch_size=8 --batches=4096 --eval_batches=2048 --final_eval_batches=16384 --L=15 --channels 96 96 96 --depths 4 4 4 --lr=0.0003 --amp_dtype=bf16 --seed=615017 --save_model"
    "--noise_model=phenomenological --measurement_error_rate=0.0200 --p=0.0200 --loss_fn=ce --epochs=300 --batch_size=8 --batches=4096 --eval_batches=2048 --final_eval_batches=16384 --L=15 --channels 96 96 96 --depths 4 4 4 --lr=0.0003 --amp_dtype=bf16 --seed=615020 --save_model"
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
