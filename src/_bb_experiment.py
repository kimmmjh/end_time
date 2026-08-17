"""Construction and launch helper for BB code-capacity experiments."""

from __future__ import annotations

import datetime
import hashlib
import logging
import os
import re
from typing import Any

import numpy as np
import torch

from models import EquivariantNeuralBP4

from ._bb_loss import DegeneracyAwareBPLoss
from ._bb_trainer import BBNeuralBPTrainer
from .bb_code import BBCodeSpec
from .bb_data_generator import BBCodeCapacityGenerator


def _graph_fingerprint(code: BBCodeSpec) -> str:
    digest = hashlib.sha256()
    digest.update(code.name.encode("utf-8"))
    digest.update(np.asarray(code.hx, dtype=np.uint8).tobytes())
    digest.update(np.asarray(code.hz, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def run_bb_experiment(args: Any) -> str:
    """Build and train the requested equivariant BB neural BP decoder."""

    code = BBCodeSpec.from_name(args.code)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.seed is None:
        actual_seed = int(
            np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0]
        )
    else:
        actual_seed = int(args.seed)
    seed_sequence = np.random.SeedSequence(actual_seed)
    train_sequence, eval_sequence = seed_sequence.spawn(2)
    train_seed = int(train_sequence.generate_state(1, dtype=np.uint64)[0])
    eval_seed = int(eval_sequence.generate_state(1, dtype=np.uint64)[0])
    np.random.seed(actual_seed % (2**32))
    torch.manual_seed(actual_seed)

    if args.bp_parameter_sharing == "orbit":
        edge_orbits = code.edge_orbit
    elif args.bp_parameter_sharing == "global":
        edge_orbits = np.zeros(code.num_edges, dtype=np.int64)
    elif args.bp_parameter_sharing == "edge":
        edge_orbits = np.arange(code.num_edges, dtype=np.int64)
        logging.warning(
            "Per-edge parameters intentionally break BB translation equivariance; "
            "use this only as an ablation."
        )
    else:  # Defensive: argparse normally prevents this.
        raise ValueError(f"Unknown BP parameter sharing {args.bp_parameter_sharing!r}.")

    model = EquivariantNeuralBP4(
        code.hx,
        code.hz,
        edge_orbits=edge_orbits,
        iterations=args.bp_iterations,
        residual_hidden_dim=args.bp_residual_hidden_dim,
        residual_scale=args.bp_residual_scale,
        max_relaxation_delta=args.bp_max_relaxation_delta,
    ).to(device)
    buffers = code.torch_buffers(device=device)
    criterion = DegeneracyAwareBPLoss(
        hx=buffers["hx"],
        hz=buffers["hz"],
        logicals_x=buffers["logicals_x"],
        logicals_z=buffers["logicals_z"],
        syndrome_weight=args.bb_syndrome_loss_weight,
        logical_weight=args.bb_logical_loss_weight,
        pauli_weight=args.bb_pauli_loss_weight,
        deep_supervision_weight=args.bp_deep_supervision_weight,
    ).to(device)

    generator_kwargs = {
        "code": code,
        "error_rate": args.p,
        "batch_size": args.batch_size,
        "noise_model": args.bb_channel,
        "x_error_rate": args.x_error_rate,
        "z_error_rate": args.z_error_rate,
    }
    train_generator = BBCodeCapacityGenerator(**generator_kwargs, seed=train_seed)
    eval_generator = BBCodeCapacityGenerator(**generator_kwargs, seed=eval_seed)

    learning_rate = (
        args.lr if args.lr is not None else (1e-4 if args.load_model else 3e-4)
    )
    final_eval_batches = args.final_eval_batches or args.eval_batches
    current_time = datetime.datetime.now()
    if args.bb_channel == "depolarizing":
        channel_slug = f"depol_p{args.p:g}"
    else:
        px = args.p if args.x_error_rate is None else args.x_error_rate
        pz = args.p if args.z_error_rate is None else args.z_error_rate
        channel_slug = f"indxz_px{px:g}_pz{pz:g}"
    run_slug = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        (
            f"capacity_{code.name}_bb_neural_bp_{channel_slug}_"
            f"it{args.bp_iterations}_h{args.bp_residual_hidden_dim}_"
            f"share{args.bp_parameter_sharing}_lr{learning_rate:g}_"
            f"bs{args.batch_size}_b{args.batches}_eb{args.eval_batches}_"
            f"ee{args.eval_every}_feb{final_eval_batches}"
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
        "architecture": "bb_neural_bp",
        "code": code.name,
        "n": code.n,
        "k": code.k,
        "d": code.d,
        "graph_fingerprint": _graph_fingerprint(code),
        "noise_model": "capacity",
        "channel": args.bb_channel,
        "error_rate": args.p,
        "x_error_rate": args.x_error_rate,
        "z_error_rate": args.z_error_rate,
        "bp_iterations": args.bp_iterations,
        "bp_residual_hidden_dim": args.bp_residual_hidden_dim,
        "bp_parameter_sharing": args.bp_parameter_sharing,
        "bp_residual_scale": args.bp_residual_scale,
        "bp_max_relaxation_delta": args.bp_max_relaxation_delta,
        "bp_deep_supervision_weight": args.bp_deep_supervision_weight,
        "bb_syndrome_loss_weight": args.bb_syndrome_loss_weight,
        "bb_logical_loss_weight": args.bb_logical_loss_weight,
        "bb_pauli_loss_weight": args.bb_pauli_loss_weight,
        "bb_weight_decay": args.bb_weight_decay,
        "seed": actual_seed,
    }

    if args.amp_dtype != "none":
        logging.warning(
            "BB BP4 uses float32 regardless of --amp_dtype=%s because parity "
            "products/log probabilities are numerically fragile in reduced precision.",
            args.amp_dtype,
        )
    trainer = BBNeuralBPTrainer(
        model=model,
        code=code,
        train_generator=train_generator,
        eval_generator=eval_generator,
        criterion=criterion,
        device=device,
        epochs=args.epochs,
        batches=args.batches,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        eval_every=args.eval_every,
        final_eval_batches=final_eval_batches,
        learning_rate=learning_rate,
        weight_decay=args.bb_weight_decay,
        gradient_clip=args.bp_gradient_clip,
        output_directory=output_directory,
        experiment_config=experiment_config,
        save_model=args.save_model,
        load_model_path=args.load_model,
    )
    logging.info("Device: %s | Seed: %d", device, actual_seed)
    logging.info(
        "Code-capacity path: one perfect syndrome; --L, measurement error, "
        "CNN channels/depths, and GRU arguments are not used."
    )
    logging.info(
        "Architecture: BP4 + orbit-shared neural residual/relaxation | "
        "iterations=%d | hidden=%d | sharing=%s",
        args.bp_iterations,
        args.bp_residual_hidden_dim,
        args.bp_parameter_sharing,
    )
    logging.info(
        "Objective weights: syndrome=%g | logical_surrogate=%g | pauli_aux=%g | "
        "deep_supervision=%g",
        args.bb_syndrome_loss_weight,
        args.bb_logical_loss_weight,
        args.bb_pauli_loss_weight,
        args.bp_deep_supervision_weight,
    )
    logging.info(
        "Channel: %s | probabilities(I,X,Y,Z)=%s",
        args.bb_channel,
        np.array2string(train_generator.channel_probabilities, precision=6),
    )
    logging.info("Output directory: %s", output_directory)
    trainer.train()
    return output_directory


__all__ = ["run_bb_experiment"]
