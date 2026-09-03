#!/bin/bash

# Canonical settings shared by the BB neural-BP ablation jobs. Keeping model
# and optimization settings separate lets each job state its changed factor
# without relying on duplicate argparse flags.
BB_NEURAL_TRAINING_CORE_ARGS="--architecture=bb_neural_bp --noise_model=capacity --rounds=1 --measurement_error_rate=0 --loss_fn=bb_coset --epochs=300 --batch_size=64 --batches=512 --eval_batches=256 --eval_every=5 --final_eval_batches=2048 --lr=0.0003 --amp_dtype=none --save_model"

BB_NEURAL_TRAINING_ARGS="$BB_NEURAL_TRAINING_CORE_ARGS --bb_channel=depolarizing"

BB_NEURAL_DEFAULT_MODEL_ARGS="--bp_iterations=12 --bp_residual_hidden_dim=64 --bp_parameter_sharing=orbit --bp_residual_scale=2.0 --bp_max_relaxation_delta=0.5 --bp_deep_supervision_weight=0.2 --bp_gradient_clip=1.0 --bb_syndrome_loss_weight=1.0 --bb_logical_loss_weight=1.0 --bb_pauli_loss_weight=0.1 --bb_weight_decay=0.0001"

BB_NEURAL_COMMON_ARGS="$BB_NEURAL_TRAINING_ARGS $BB_NEURAL_DEFAULT_MODEL_ARGS"

# OSD-enabled circuit reference retained for reproducing the earlier campaign.
# Here --rounds is the number of noisy extraction cycles; the corrected circuit
# adds one perfect closing detector frame. OSD-0 is used during checkpoint
# selection because CS-7 is too expensive to run at every validation point on
# the large space-time DEM graphs.
BB_CIRCUIT_TRAINING_CORE_ARGS="--architecture=bb_neural_bp --noise_model=circuit --bb_circuit_noise_model=legacy --loss_fn=bb_coset --epochs=100 --batches=128 --eval_every=10 --lr=0.0003 --amp_dtype=none --save_model --bb_osd_eval_shots=4096 --bb_osd_method=OSD_0 --bb_osd_order=0"

BB_CIRCUIT_BB72_TRAINING_ARGS="$BB_CIRCUIT_TRAINING_CORE_ARGS --code=bb72 --rounds=6 --batch_size=16 --eval_batches=64 --final_eval_batches=256"
BB_CIRCUIT_BB144_TRAINING_ARGS="$BB_CIRCUIT_TRAINING_CORE_ARGS --code=bb144 --rounds=12 --batch_size=8 --eval_batches=128 --final_eval_batches=512"

BB_CIRCUIT_DEFAULT_MODEL_ARGS="--bp_iterations=12 --bp_residual_hidden_dim=32 --bp_orbit_embedding_dim=8 --bp_parameter_sharing=orbit --bp_normalisation=0.625 --bp_residual_scale=2.0 --bp_max_relaxation_delta=0.5 --bp_deep_supervision_weight=0.2 --bp_gradient_clip=1.0 --bb_syndrome_loss_weight=1.0 --bb_logical_loss_weight=1.0 --bb_pauli_loss_weight=0.1 --bb_weight_decay=0.0001"

BB_CIRCUIT_BB72_ARGS="$BB_CIRCUIT_BB72_TRAINING_ARGS $BB_CIRCUIT_DEFAULT_MODEL_ARGS"
BB_CIRCUIT_BB144_ARGS="$BB_CIRCUIT_BB144_TRAINING_ARGS $BB_CIRCUIT_DEFAULT_MODEL_ARGS"

# Raw circuit-level Neural-BP campaign. Setting the OSD shot budget to zero
# prevents construction or execution of the OSD post-processor. Validation and
# checkpoint selection use raw Neural-BP2 paired gain against raw vanilla BP2.
# These are fresh runs: OSD-selected checkpoints use an incompatible selection
# metric and are intentionally not resumed here.
BB_CIRCUIT_NO_OSD_TRAINING_CORE_ARGS="--architecture=bb_neural_bp --noise_model=circuit --bb_circuit_noise_model=legacy --loss_fn=bb_coset --epochs=100 --batches=128 --eval_every=10 --lr=0.0003 --amp_dtype=none --save_model --bb_osd_eval_shots=0"

BB_CIRCUIT_NO_OSD_BB72_TRAINING_ARGS="$BB_CIRCUIT_NO_OSD_TRAINING_CORE_ARGS --code=bb72 --rounds=6 --batch_size=16 --eval_batches=64 --final_eval_batches=256"
BB_CIRCUIT_NO_OSD_BB144_TRAINING_ARGS="$BB_CIRCUIT_NO_OSD_TRAINING_CORE_ARGS --code=bb144 --rounds=12 --batch_size=8 --eval_batches=128 --final_eval_batches=512"

BB_CIRCUIT_NO_OSD_BB72_ARGS="$BB_CIRCUIT_NO_OSD_BB72_TRAINING_ARGS $BB_CIRCUIT_DEFAULT_MODEL_ARGS"
BB_CIRCUIT_NO_OSD_BB144_ARGS="$BB_CIRCUIT_NO_OSD_BB144_TRAINING_ARGS $BB_CIRCUIT_DEFAULT_MODEL_ARGS"
