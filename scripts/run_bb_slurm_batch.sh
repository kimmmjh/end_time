#!/bin/bash

# Shared launcher for run_bb_*.slurm. The calling script must define ROOT and
# an experiments array, then call run_bb_experiments "${experiments[@]}".
# BB_PYTHON_ENTRYPOINT may override main.py for inference-only benchmarks.
# BB_DRY_RUN=1 prints expanded commands without an allocation or filesystem
# writes. Optional BB_CAMPAIGN and BB_EXPERIMENT_LABELS describe the experiment.

_BB_STEP_PIDS=()
_BB_RESULT_DIRECTORY=""

_bb_stop_steps() {
    local pid

    trap - INT TERM
    echo
    echo "Stopping BB Slurm steps..."

    for pid in "${_BB_STEP_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    for pid in "${_BB_STEP_PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done

    if [[ -n "$_BB_RESULT_DIRECTORY" ]]; then
        {
            echo "Interrupted: $(date --iso-8601=seconds)"
            echo "Job ID: ${SLURM_JOB_ID:-unknown}"
        } > "$_BB_RESULT_DIRECTORY/interrupted.txt"
    fi

    exit 130
}

run_bb_experiments() {
    local experiment_count="$#"
    local cpus_per_task="${SLURM_CPUS_PER_TASK:-16}"
    local gpus_per_task="${BB_GPUS_PER_TASK:-1}"
    local entrypoint
    local index params pid exit_code label
    local failures=0
    local -a argv=()
    local -a step_resources=(
        --cpus-per-task="$cpus_per_task"
        --mem=16G
    )

    if [[ -z "${ROOT:-}" ]]; then
        echo "Error: ROOT must be set before calling run_bb_experiments." >&2
        return 2
    fi
    entrypoint="${BB_PYTHON_ENTRYPOINT:-$ROOT/main.py}"
    if ((experiment_count < 1 || experiment_count > 4)); then
        echo "Error: expected 1-4 experiments, got $experiment_count." >&2
        return 2
    fi
    if [[ ! "$gpus_per_task" =~ ^[0-9]+$ ]]; then
        echo "Error: BB_GPUS_PER_TASK must be a non-negative integer." >&2
        return 2
    fi
    if ((gpus_per_task > 0)); then
        step_resources+=(--gpus-per-task="$gpus_per_task")
    fi
    if [[ ! -f "$entrypoint" ]]; then
        echo "Error: Python entrypoint not found: $entrypoint" >&2
        return 1
    fi

    if [[ "${BB_DRY_RUN:-0}" == 1 ]]; then
        for params in "$@"; do
            argv=()
            read -r -a argv <<< "$params"
            printf 'python -u %q ' "$entrypoint"
            printf '%q ' "${argv[@]}"
            echo
        done
        return 0
    fi
    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
        echo "Error: this launcher must run inside a Slurm allocation." >&2
        return 2
    fi

    _BB_RESULT_DIRECTORY="$ROOT/resdir_${SLURM_JOB_ID}"
    if [[ -e "$_BB_RESULT_DIRECTORY" ]]; then
        echo "Error: result directory already exists: $_BB_RESULT_DIRECTORY" >&2
        return 1
    fi
    mkdir "$_BB_RESULT_DIRECTORY" || return 1

    {
        echo "Started: $(date --iso-8601=seconds)"
        echo "Job ID: $SLURM_JOB_ID"
        echo "Job name: ${SLURM_JOB_NAME:-unknown}"
        echo "Node list: ${SLURM_NODELIST:-unknown}"
        echo "Experiments: $experiment_count"
        echo "Campaign: ${BB_CAMPAIGN:-unspecified}"
    } > "$_BB_RESULT_DIRECTORY/job_metadata.txt"
    cp "$0" "$_BB_RESULT_DIRECTORY/submitted_script.slurm" 2>/dev/null || true
    cp "$ROOT/scripts/run_bb_slurm_batch.sh" \
        "$_BB_RESULT_DIRECTORY/launcher_snapshot.sh" 2>/dev/null || true
    if [[ -f "$ROOT/scripts/bb_neural_slurm_defaults.sh" ]]; then
        cp "$ROOT/scripts/bb_neural_slurm_defaults.sh" \
            "$_BB_RESULT_DIRECTORY/neural_defaults_snapshot.sh" \
            2>/dev/null || true
    fi
    cp "$entrypoint" "$_BB_RESULT_DIRECTORY/entrypoint_snapshot.py" \
        2>/dev/null || true
    # The active decoder may be uncommitted during a research campaign.
    # Preserve its implementation as well as the exact shell command.
    if [[ "${BB_CAMPAIGN:-}" == "bb_neural_relay_2026_09" || \
          "${BB_CAMPAIGN:-}" == "bb_plain_bp_2026_09" ]]; then
        mkdir -p "$_BB_RESULT_DIRECTORY/source_snapshot/models" \
            "$_BB_RESULT_DIRECTORY/source_snapshot/src"
        for params in models/_equivariant_neural_bp2.py models/_neural_relay_bp2.py \
            src/_bb_circuit_experiment.py src/_bb_circuit_trainer.py \
            src/_bb_circuit_loss.py src/_bb_circuit_metrics.py \
            src/bb_circuit_data.py src/bb_dem.py src/bb_stim_utils.py src/bb_code.py; do
            cp "$ROOT/$params" "$_BB_RESULT_DIRECTORY/source_snapshot/$params" || return 1
        done
    fi
    if [[ "${BB_CAMPAIGN:-}" == "bb_tanner_cnn_2026_09" ]]; then
        cp "$ROOT/scripts/bb_tanner_cnn_slurm_defaults.sh" \
            "$_BB_RESULT_DIRECTORY/tanner_cnn_defaults_snapshot.sh" || return 1
        mkdir -p "$_BB_RESULT_DIRECTORY/source_snapshot/models" \
            "$_BB_RESULT_DIRECTORY/source_snapshot/src"
        for params in models/_bb_tanner_cnn.py src/_bb_tanner_cnn_experiment.py \
            src/_direct_osd.py src/_bb_loss.py src/_bb_metrics.py \
            src/bb_data_generator.py src/bb_code.py; do
            cp "$ROOT/$params" "$_BB_RESULT_DIRECTORY/source_snapshot/$params" || return 1
        done
    fi
    if command -v git >/dev/null 2>&1 && \
       git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git -C "$ROOT" rev-parse HEAD > "$_BB_RESULT_DIRECTORY/git_commit.txt"
        git -C "$ROOT" status --short > "$_BB_RESULT_DIRECTORY/git_status.txt"
    fi

    cd "$_BB_RESULT_DIRECTORY" || return 1
    export SLURM_CPU_BIND="cores"
    _BB_STEP_PIDS=()
    trap _bb_stop_steps INT TERM

    index=0
    printf 'experiment_index\tlabel\targuments\n' > experiments.tsv
    for params in "$@"; do
        argv=()
        read -r -a argv <<< "$params"
        label="${BB_EXPERIMENT_LABELS[$index]:-experiment_$index}"
        printf '%s\t%s\t%s\n' "$index" "$label" "$params" >> experiments.tsv

        {
            printf 'python -u %q ' "$entrypoint"
            printf '%q ' "${argv[@]}"
            echo
        } > "command_exp_${index}.txt"

        echo "Starting experiment $index ($label): $params"
        srun --exclusive --nodes=1 --ntasks=1 \
            "${step_resources[@]}" \
            python -u "$entrypoint" "${argv[@]}" \
            > "log_exp_${index}.txt" 2>&1 &
        _BB_STEP_PIDS+=("$!")
        index=$((index + 1))
    done

    for index in "${!_BB_STEP_PIDS[@]}"; do
        pid="${_BB_STEP_PIDS[$index]}"
        if wait "$pid"; then
            exit_code=0
        else
            exit_code=$?
            failures=$((failures + 1))
        fi

        echo "$exit_code" > "exit_code_exp_${index}.txt"
        {
            echo "Finished: $(date --iso-8601=seconds)"
            echo "Exit code: $exit_code"
        } > "finished_exp_${index}.txt"
        echo "Experiment $index finished with exit code $exit_code."
    done

    trap - INT TERM
    if ((failures > 0)); then
        {
            echo "Failed experiments: $failures/$experiment_count"
            echo "Finished: $(date --iso-8601=seconds)"
        } > failed.txt
        echo "Error: $failures BB experiment(s) failed." >&2
        return 1
    fi

    {
        echo "All $experiment_count experiments completed"
        echo "Finished: $(date --iso-8601=seconds)"
    } > completed.txt
    echo "All BB experiments completed: $_BB_RESULT_DIRECTORY"
}
