"""Construction and launch helper for circuit-level BB experiments."""

from __future__ import annotations

import datetime
import hashlib
import logging
import os
import re
from typing import Any

import numpy as np
import torch

from models._equivariant_neural_bp2 import EquivariantNeuralBP2

from ._bb_circuit_loss import CircuitDegeneracyAwareLoss
from ._bb_circuit_trainer import BBCircuitTrainer
from .bb_circuit_data import BBCircuitGenerator
from .bb_code import BBCodeSpec
from .bb_stim_utils import CIRCUIT_SCHEMA_VERSION


def _graph_fingerprint(graph: Any) -> str:
    """Hash the decoding graph so a checkpoint cannot be reused across codes."""

    digest = hashlib.sha256()
    digest.update(graph.code_name.encode("utf-8"))
    digest.update(
        np.asarray(
            [
                graph.circuit_schema_version,
                graph.rounds,
                graph.detector_frames,
                graph.num_detectors,
                graph.num_mechanisms,
                graph.num_observables,
            ],
            dtype=np.int64,
        ).tobytes()
    )
    digest.update(graph.dem_fingerprint.encode("ascii"))
    digest.update(np.asarray(graph.priors, dtype=np.float64).tobytes())
    digest.update(np.asarray(graph.edge_detector, dtype=np.int64).tobytes())
    digest.update(np.asarray(graph.edge_mechanism, dtype=np.int64).tobytes())
    digest.update(np.asarray(graph.edge_orbit, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _resume_metadata(path: str) -> tuple[dict[str, Any], int]:
    """Read the trusted local checkpoint before constructing sampler streams."""

    if not os.path.isfile(path):
        raise FileNotFoundError(f"BB circuit checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if int(checkpoint.get("format_version", 0)) < 3:
        raise ValueError(
            "This BB circuit checkpoint uses the old round/scoring semantics "
            "and cannot be resumed; start a new corrected experiment."
        )
    config = checkpoint.get("experiment_config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint has no BB circuit experiment_config.")
    return config, int(checkpoint.get("epoch", -1)) + 1


def run_bb_circuit_experiment(args: Any) -> str:
    """Build and train a circuit-level BB neural BP decoder."""

    code = BBCodeSpec.from_name(args.code)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    resume_config: dict[str, Any] | None = None
    phase_start_epoch = 0
    if args.load_model:
        resume_config, phase_start_epoch = _resume_metadata(args.load_model)
        saved_seed = resume_config.get("seed")
        if saved_seed is None:
            raise ValueError("Resume checkpoint does not record its base seed.")
        if args.seed is not None and int(args.seed) != int(saved_seed):
            raise ValueError(
                f"Resume seed mismatch: checkpoint={saved_seed}, requested={args.seed}."
            )
        actual_seed = int(saved_seed)
    elif args.seed is None:
        actual_seed = int(
            np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0]
        )
    else:
        actual_seed = int(args.seed)

    # Stim's compiled samplers do not expose serializable RNG state.  Include
    # the resumed epoch in their seed sequence so a resumed phase is
    # deterministic but does not replay the original training shots.
    train_sequence, eval_sequence, host_sequence = np.random.SeedSequence(
        [actual_seed, phase_start_epoch, CIRCUIT_SCHEMA_VERSION]
    ).spawn(3)
    np.random.seed(int(host_sequence.generate_state(1, dtype=np.uint32)[0]))
    torch.manual_seed(
        int(host_sequence.generate_state(1, dtype=np.uint64)[0] % (2**63 - 1))
    )

    rounds = int(args.rounds) if args.rounds is not None else code.d
    generator_kwargs = {
        "code": code,
        "rounds": rounds,
        "gate_error_rate": args.p,
        "measurement_error_rate": args.measurement_error_rate,
        "idle_error_rate": args.bb_idle_error_rate,
        "batch_size": args.batch_size,
    }
    train_generator = BBCircuitGenerator(
        **generator_kwargs,
        seed=int(train_sequence.generate_state(1, dtype=np.uint64)[0]),
    )
    # Share the decoding graph: it is a deterministic function of the circuit,
    # and rebuilding it for evaluation costs tens of seconds on BB144.
    eval_generator = BBCircuitGenerator(
        **generator_kwargs,
        seed=int(eval_sequence.generate_state(1, dtype=np.uint64)[0]),
        graph=train_generator.graph,
    )
    graph = train_generator.graph

    model = EquivariantNeuralBP2(
        graph,
        iterations=args.bp_iterations,
        hidden_dim=args.bp_residual_hidden_dim,
        orbit_embedding_dim=args.bp_orbit_embedding_dim,
        sharing=args.bp_parameter_sharing,
        normalisation=args.bp_normalisation,
        residual_scale=args.bp_residual_scale,
        max_relaxation_delta=args.bp_max_relaxation_delta,
        gradient_checkpoint=not args.bp_no_gradient_checkpoint,
    ).to(device)
    criterion = CircuitDegeneracyAwareLoss(
        check_matrix=graph.check_matrix,
        observable_matrix=graph.observable_matrix,
        syndrome_weight=args.bb_syndrome_loss_weight,
        logical_weight=args.bb_logical_loss_weight,
        mechanism_weight=args.bb_pauli_loss_weight,
        deep_supervision_weight=args.bp_deep_supervision_weight,
    ).to(device)

    learning_rate = (
        args.lr if args.lr is not None else (1e-4 if args.load_model else 3e-4)
    )
    final_eval_batches = args.final_eval_batches or args.eval_batches
    current_time = datetime.datetime.now()
    run_slug = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        (
            f"circuit_{code.name}_bb_neural_bp2_p{args.p:g}_"
            f"q{args.measurement_error_rate:g}_r{rounds}_f{graph.detector_frames}_"
            f"it{args.bp_iterations}_h{args.bp_residual_hidden_dim}_"
            f"emb{args.bp_orbit_embedding_dim}_"
            f"share{args.bp_parameter_sharing}_"
            f"rs{args.bp_residual_scale:g}_rd{args.bp_max_relaxation_delta:g}_"
            f"ds{args.bp_deep_supervision_weight:g}_lr{learning_rate:g}_"
            f"bs{args.batch_size}_b{args.batches}_eb{args.eval_batches}_"
            f"ee{args.eval_every}_feb{final_eval_batches}_seed{actual_seed}"
            + ("_resume" if args.load_model else "")
        ),
    )
    output_directory = os.path.join(
        os.getcwd(),
        "outputs",
        current_time.strftime("%Y-%m-%d"),
        f"{current_time.strftime('%H-%M-%S-%f')}_{run_slug}",
    )

    experiment_config = {
        "architecture": "bb_neural_bp_circuit",
        "circuit_schema_version": CIRCUIT_SCHEMA_VERSION,
        "code": code.name,
        "n": code.n,
        "k": code.k,
        "d": code.d,
        "rounds": rounds,
        "detector_frames": graph.detector_frames,
        "noise_model": "circuit",
        "gate_error_rate": args.p,
        "measurement_error_rate": args.measurement_error_rate,
        "idle_error_rate": args.bb_idle_error_rate,
        "graph_fingerprint": _graph_fingerprint(graph),
        "num_detectors": graph.num_detectors,
        "num_mechanisms": graph.num_mechanisms,
        "num_edges": graph.num_edges,
        "num_orbits": graph.num_orbits,
        "bp_iterations": args.bp_iterations,
        "bp_residual_hidden_dim": args.bp_residual_hidden_dim,
        "bp_orbit_embedding_dim": args.bp_orbit_embedding_dim,
        "bp_parameter_sharing": args.bp_parameter_sharing,
        "bp_normalisation": args.bp_normalisation,
        "bp_residual_scale": args.bp_residual_scale,
        "bp_max_relaxation_delta": args.bp_max_relaxation_delta,
        "bp_deep_supervision_weight": args.bp_deep_supervision_weight,
        "bb_syndrome_loss_weight": args.bb_syndrome_loss_weight,
        "bb_logical_loss_weight": args.bb_logical_loss_weight,
        "bb_pauli_loss_weight": args.bb_pauli_loss_weight,
        "bb_weight_decay": args.bb_weight_decay,
        "learning_rate": learning_rate,
        "checkpoint_selection_metric": (
            "neural_osd_paired_gain"
            if args.bb_osd_eval_shots > 0
            else "neural_paired_gain"
        ),
        "bb_osd_method": args.bb_osd_method,
        "bb_osd_order": args.bb_osd_order,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "seed": actual_seed,
        "sampler_phase_start_epoch": phase_start_epoch,
    }

    logging.info("Device: %s | Seed: %d", device, actual_seed)
    logging.info("Decoding graph: %s", graph.summary())
    logging.info(
        "Architecture: normalised min-sum BP2 + orbit-shared neural residual | "
        "iterations=%d | hidden=%d | embedding=%d | sharing=%s",
        args.bp_iterations,
        args.bp_residual_hidden_dim,
        args.bp_orbit_embedding_dim,
        args.bp_parameter_sharing,
    )
    logging.info(
        "Plain belief propagation is a weak decoder on a quantum LDPC detector "
        "error model. Enable --bb_osd_eval_shots to also report the "
        "Neural-BP+OSD versus BP+OSD comparison that the classical literature "
        "uses as its baseline."
    )
    logging.info("Output directory: %s", output_directory)

    trainer = BBCircuitTrainer(
        model=model,
        generator=train_generator,
        eval_generator=eval_generator,
        criterion=criterion,
        device=device,
        epochs=args.epochs,
        batches=args.batches,
        eval_batches=args.eval_batches,
        eval_every=args.eval_every,
        final_eval_batches=final_eval_batches,
        learning_rate=learning_rate,
        weight_decay=args.bb_weight_decay,
        gradient_clip=args.bp_gradient_clip,
        output_directory=output_directory,
        experiment_config=experiment_config,
        save_model=args.save_model,
        osd_eval_shots=args.bb_osd_eval_shots,
        osd_method=args.bb_osd_method,
        osd_order=args.bb_osd_order,
        load_model_path=args.load_model,
    )
    trainer.train()
    return output_directory


__all__ = ["run_bb_circuit_experiment"]
