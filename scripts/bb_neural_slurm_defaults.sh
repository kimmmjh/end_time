#!/bin/bash

# Canonical settings shared by the BB neural-BP ablation jobs. Keeping model
# and optimization settings separate lets each job state its changed factor
# without relying on duplicate argparse flags.
BB_NEURAL_TRAINING_CORE_ARGS="--architecture=bb_neural_bp --noise_model=capacity --rounds=1 --measurement_error_rate=0 --loss_fn=bb_coset --epochs=300 --batch_size=64 --batches=512 --eval_batches=256 --eval_every=5 --final_eval_batches=2048 --lr=0.0003 --amp_dtype=none --save_model"

BB_NEURAL_TRAINING_ARGS="$BB_NEURAL_TRAINING_CORE_ARGS --bb_channel=depolarizing"

BB_NEURAL_DEFAULT_MODEL_ARGS="--bp_iterations=12 --bp_residual_hidden_dim=64 --bp_parameter_sharing=orbit --bp_residual_scale=2.0 --bp_max_relaxation_delta=0.5 --bp_deep_supervision_weight=0.2 --bp_gradient_clip=1.0 --bb_syndrome_loss_weight=1.0 --bb_logical_loss_weight=1.0 --bb_pauli_loss_weight=0.1 --bb_weight_decay=0.0001"

BB_NEURAL_COMMON_ARGS="$BB_NEURAL_TRAINING_ARGS $BB_NEURAL_DEFAULT_MODEL_ARGS"
