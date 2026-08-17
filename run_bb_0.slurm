#!/bin/bash
#SBATCH --job-name=bb72_neural_bp
#SBATCH --account=m5328_g
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1

module load python/3.10
source $PSCRATCH/envs/nde/bin/activate
cd $HOME/end_time

experiments=(
    "--code=bb72 --architecture=bb_neural_bp --noise_model=capacity --measurement_error_rate=0 --loss_fn=bb_coset --bb_channel=depolarizing --p=0.04 --bp_iterations=12 --bp_residual_hidden_dim=64 --bp_parameter_sharing=orbit --epochs=300 --batch_size=64 --batches=512 --eval_batches=256 --eval_every=5 --final_eval_batches=2048 --lr=0.0003 --amp_dtype=none --seed=7204 --save_model"
    "--code=bb72 --architecture=bb_neural_bp --noise_model=capacity --measurement_error_rate=0 --loss_fn=bb_coset --bb_channel=depolarizing --p=0.06 --bp_iterations=12 --bp_residual_hidden_dim=64 --bp_parameter_sharing=orbit --epochs=300 --batch_size=64 --batches=512 --eval_batches=256 --eval_every=5 --final_eval_batches=2048 --lr=0.0003 --amp_dtype=none --seed=7206 --save_model"
    "--code=bb72 --architecture=bb_neural_bp --noise_model=capacity --measurement_error_rate=0 --loss_fn=bb_coset --bb_channel=depolarizing --p=0.08 --bp_iterations=12 --bp_residual_hidden_dim=64 --bp_parameter_sharing=orbit --epochs=300 --batch_size=64 --batches=512 --eval_batches=256 --eval_every=5 --final_eval_batches=2048 --lr=0.0003 --amp_dtype=none --seed=7208 --save_model"
    "--code=bb72 --architecture=bb_neural_bp --noise_model=capacity --measurement_error_rate=0 --loss_fn=bb_coset --bb_channel=depolarizing --p=0.10 --bp_iterations=12 --bp_residual_hidden_dim=64 --bp_parameter_sharing=orbit --epochs=300 --batch_size=64 --batches=512 --eval_batches=256 --eval_every=5 --final_eval_batches=2048 --lr=0.0003 --amp_dtype=none --seed=7210 --save_model"
)

currdir=$PWD
dirname="resdir_$SLURM_JOB_ID"
mkdir -p $dirname
cd $dirname
export SLURM_CPU_BIND="cores"

job_count=0

for params in "${experiments[@]}"; do
    echo "Starting Experiment: $params"

    srun --ntasks=1 --mem=16G \
        python "$currdir/main.py" $params > log_exp_${job_count}.txt 2>&1 &

    sleep 3

    job_count=$((job_count + 1))
done

wait
