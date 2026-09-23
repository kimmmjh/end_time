#!/bin/bash

# Code-capacity Tanner CNN campaign. Every run evaluates raw CNN and direct
# OSD-0 on identical shots from one raw-LER-selected checkpoint.
# 8192 training shots/epoch, 4096 validation shots, 65536 final test shots.
BB_TANNER_CNN_TRAINING_ARGS="--architecture=bb_tanner_cnn --noise_model=capacity --bb_channel=depolarizing --rounds=1 --measurement_error_rate=0 --bb_idle_error_rate=0 --loss_fn=bb_coset --epochs=100 --batch_size=64 --batches=128 --eval_batches=64 --eval_every=5 --final_eval_batches=1024 --lr=0.0003 --amp_dtype=none --save_model"
BB_TANNER_CNN_MODEL_ARGS="--bb_cnn_width=64 --bb_cnn_gradient_clip=1.0 --bb_syndrome_loss_weight=1.0 --bb_logical_loss_weight=1.0 --bb_pauli_loss_weight=0.1 --bb_weight_decay=0.0001"

# These graphs are small; avoid oversubscribing CPU tensor/BLAS operations.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

bb_tanner_cnn_experiment_args() {
    if (($# != 4)); then
        echo "Usage: bb_tanner_cnn_experiment_args code depth p seed" >&2
        return 2
    fi
    local code="$1" depth="$2" p="$3" seed="$4"
    case "$code" in
        bb72|bb144) ;;
        *) echo "Unknown Tanner CNN campaign code: $code" >&2; return 2 ;;
    esac
    if [[ ! "$depth" =~ ^[1-9][0-9]*$ ]]; then
        echo "Tanner CNN depth must be a positive integer." >&2
        return 2
    fi
    printf '%s %s --code=%s --bb_cnn_depth=%s --p=%s --seed=%s\n' \
        "$BB_TANNER_CNN_TRAINING_ARGS" "$BB_TANNER_CNN_MODEL_ARGS" \
        "$code" "$depth" "$p" "$seed"
}
