#!/bin/bash

# Canonical settings shared by the BB neural-BP ablation jobs. Keeping model
# and optimization settings separate lets each job state its changed factor
# without relying on duplicate argparse flags.
# Ordinary-BP evaluation jobs reuse this shared settings file and launcher.
bb_plain_bp_experiment_args() {
    if (($# != 4)); then
        echo "Usage: bb_plain_bp_experiment_args code p seed output_directory" >&2
        return 2
    fi
    local code="$1" p="$2" seed="$3" output_directory="$4"
    local rounds batch_size
    case "$code" in
        bb72) rounds=6; batch_size=16 ;;
        bb144) rounds=12; batch_size=8 ;;
        *) echo "Unknown BP campaign code: $code" >&2; return 2 ;;
    esac
    printf '%s\n' "--code=$code --p=$p --seed=$seed --rounds=$rounds --shots=${BB_BP_SHOTS:-4096} --batch-size=$batch_size --iteration-caps 12 ${BB_BP_REFERENCE_ITERATIONS:-1000} --normalisation=0.625 --message-clip=30 --device=cuda --threads=1 --out=$output_directory"
}

BB_NEURAL_TRAINING_CORE_ARGS="--architecture=bb_neural_bp --noise_model=capacity --rounds=1 --measurement_error_rate=0 --loss_fn=bb_coset --epochs=300 --batch_size=64 --batches=512 --eval_batches=256 --eval_every=5 --final_eval_batches=2048 --lr=0.0003 --amp_dtype=none --save_model"

BB_NEURAL_TRAINING_ARGS="$BB_NEURAL_TRAINING_CORE_ARGS --bb_channel=depolarizing"

BB_NEURAL_DEFAULT_MODEL_ARGS="--bp_iterations=12 --bp_residual_hidden_dim=64 --bp_parameter_sharing=orbit --bp_residual_scale=2.0 --bp_max_relaxation_delta=0.5 --bp_deep_supervision_weight=0.2 --bp_gradient_clip=1.0 --bb_syndrome_loss_weight=1.0 --bb_logical_loss_weight=1.0 --bb_pauli_loss_weight=0.1 --bb_weight_decay=0.0001"

BB_NEURAL_COMMON_ARGS="$BB_NEURAL_TRAINING_ARGS $BB_NEURAL_DEFAULT_MODEL_ARGS"

# OSD-enabled circuit settings retained from the earlier campaign. New runs
# use first-valid BP stopping and additionally report a max-1000 BP reference;
# archived fixed-iteration results are not reproduced by the new evaluation.
# Here --rounds is the number of noisy extraction cycles; the corrected circuit
# adds one perfect closing detector frame. OSD-0 is used during checkpoint
# selection because CS-7 is too expensive to run at every validation point on
# the large space-time DEM graphs.
BB_CIRCUIT_TRAINING_CORE_ARGS="--architecture=bb_neural_bp --noise_model=circuit --bb_circuit_noise_model=legacy --loss_fn=bb_coset --epochs=100 --batches=128 --eval_every=10 --lr=0.0003 --amp_dtype=none --save_model --bb_bp_reference_iterations=1000 --bb_osd_eval_shots=4096 --bb_osd_method=OSD_0 --bb_osd_order=0"

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
BB_CIRCUIT_NO_OSD_TRAINING_CORE_ARGS="--architecture=bb_neural_bp --noise_model=circuit --bb_circuit_noise_model=legacy --loss_fn=bb_coset --epochs=100 --batches=128 --eval_every=10 --lr=0.0003 --amp_dtype=none --save_model --bb_bp_reference_iterations=1000 --bb_osd_eval_shots=0"

BB_CIRCUIT_NO_OSD_BB72_TRAINING_ARGS="$BB_CIRCUIT_NO_OSD_TRAINING_CORE_ARGS --code=bb72 --rounds=6 --batch_size=16 --eval_batches=64 --final_eval_batches=256"
BB_CIRCUIT_NO_OSD_BB144_TRAINING_ARGS="$BB_CIRCUIT_NO_OSD_TRAINING_CORE_ARGS --code=bb144 --rounds=12 --batch_size=8 --eval_batches=128 --final_eval_batches=512"

BB_CIRCUIT_NO_OSD_BB72_ARGS="$BB_CIRCUIT_NO_OSD_BB72_TRAINING_ARGS $BB_CIRCUIT_DEFAULT_MODEL_ARGS"
BB_CIRCUIT_NO_OSD_BB144_ARGS="$BB_CIRCUIT_NO_OSD_BB144_TRAINING_ARGS $BB_CIRCUIT_DEFAULT_MODEL_ARGS"

# September Relay campaign: retain the archived circuit, optimizer, loss,
# batch sizes and evaluation budgets. Iterations/sharing/Relay settings are
# supplied once by the variant builder below (no repeated argparse options).
BB_RELAY_MODEL_CORE_ARGS="--bp_residual_hidden_dim=32 --bp_orbit_embedding_dim=8 --bp_normalisation=0.625 --bp_residual_scale=2.0 --bp_max_relaxation_delta=0.5 --bp_deep_supervision_weight=0.2 --bp_gradient_clip=1.0 --bb_syndrome_loss_weight=1.0 --bb_logical_loss_weight=1.0 --bb_pauli_loss_weight=0.1 --bb_weight_decay=0.0001"

bb_relay_experiment_args() {
    if (($# != 4)); then
        echo "Usage: bb_relay_experiment_args code variant p seed" >&2
        return 2
    fi
    local code="$1" variant="$2" p="$3" seed="$4"
    local training iterations=12 legs=4 solutions=2 sharing=orbit
    local memory_first=0.125 memory_min=-0.24 memory_max=0.66
    case "$code" in
        bb72) training="$BB_CIRCUIT_NO_OSD_BB72_TRAINING_ARGS" ;;
        bb144) training="$BB_CIRCUIT_NO_OSD_BB144_TRAINING_ARGS" ;;
        *) echo "Unknown Relay campaign code: $code" >&2; return 2 ;;
    esac
    case "$variant" in
        relay) ;;
        legacy_t12) legs=0 ;;
        legacy_t48) legs=0; iterations=48 ;;
        zero_memory) memory_first=0; memory_min=0; memory_max=0 ;;
        # Match the mean of U[-0.24,0.66], keeping first-leg gamma=0.125.
        constant_memory) memory_min=0.21; memory_max=0.21 ;;
        one_leg) legs=1; iterations=48; solutions=1 ;;
        twelve_legs) legs=12; iterations=4 ;;
        first_solution) solutions=1 ;;
        global) sharing=global ;;
        *) echo "Unknown Relay campaign variant: $variant" >&2; return 2 ;;
    esac
    printf '%s %s --bp_iterations=%s --bp_parameter_sharing=%s --bp_relay_legs=%s' \
        "$training" "$BB_RELAY_MODEL_CORE_ARGS" "$iterations" "$sharing" "$legs"
    if ((legs > 0)); then
        printf ' --bp_relay_solutions=%s --bp_relay_memory_strength=%s --bp_relay_memory_min=%s --bp_relay_memory_max=%s' \
            "$solutions" "$memory_first" "$memory_min" "$memory_max"
    fi
    printf ' --p=%s --measurement_error_rate=%s --bb_idle_error_rate=0 --seed=%s\n' \
        "$p" "$p" "$seed"
}
