import os
import torch
import logging
import argparse
import datetime
import re
import numpy as np

from torch import nn
from models import (
    Decoder,
    MatchingResidualDecoder,
    NeuralWeightedMatchingDecoder,
    RecurrentEND2D,
    RecurrentEdgeWeightNetwork,
    RecurrentResidualEND2D,
)
from models.loss_functions import DynamicCELoss, EdgeBCELoss
from models._the_end_3d import TransformedEND3D
from models.pooling_layers import TranslationalEquivariantPooling2D
from src import Trainer
from src._bb_experiment import run_bb_experiment
from src._bb_circuit_experiment import run_bb_circuit_experiment
from src.bb_stim_utils import (
    BB_CIRCUIT_NOISE_MODELS,
    resolve_bb_circuit_noise_profile,
)
from src.stim_utils import generate_toric_memory_circuit
from panqec.codes import Toric2DCode


_BB_DISTANCE = {"bb72": 6, "bb144": 12}


def main() -> None:
    """
    Start the experiments for decoder training.
    """
    parser = argparse.ArgumentParser(
        description="Equivariant neural decoders for toric and BB codes"
    )
    parser.add_argument(
        "--code",
        type=str,
        default="toric",
        choices=["toric", "bb72", "bb144"],
        help=(
            "Code family. bb72 and bb144 select the [[72,12,6]] and "
            "[[144,12,12]] bivariate-bicycle codes."
        ),
    )
    parser.add_argument("--L", type=int, default=5, help="Lattice size (L x L).")
    parser.add_argument("--p", type=float, default=0.01, help="Error rate [0,1).")
    parser.add_argument(
        "--noise_model",
        type=str,
        default="phenomenological",
        choices=["capacity", "phenomenological", "circuit"],
        help="Noise model type.",
    )
    parser.add_argument(
        "--measurement_error_rate",
        type=float,
        default=None,
        help=(
            "Measurement error rate [0,1). Defaults to 0.01 on temporal toric "
            "runs and exactly 0 for BB code capacity. BB standard/SI1000 "
            "circuit profiles derive it from --p."
        ),
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help=(
            "Syndrome rounds. For BB circuit noise this is the number of noisy "
            "extraction cycles; a separate perfect closing frame is added."
        ),
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="cnn3d",
        choices=[
            "cnn3d",
            "convgru",
            "convgru_mwpm",
            "convgru_weighted_mwpm",
            "bb_neural_bp",
        ],
        help=(
            "Decoder architecture. cnn3d treats time as a third spatial axis; "
            "convgru applies a shared equivariant 2D CNN to each round and "
            "recurrently accumulates the rounds; convgru_mwpm lets a Stim DEM "
            "PyMatching decoder do global pairing and trains the equivariant "
            "ConvGRU to predict its residual logical class; "
            "convgru_weighted_mwpm instead predicts shot-dependent sparse DEM "
            "edge probabilities before MWPM."
            " bb_neural_bp uses BP4 on the BB code Tanner graph for capacity "
            "noise and binary BP2 on Stim's detector-error-model graph for "
            "circuit noise, with cyclic edge-orbit parameter sharing."
        ),
    )
    parser.add_argument(
        "--bp_iterations",
        type=int,
        default=12,
        help=(
            "Unrolled BP iterations for --architecture=bb_neural_bp: BP4 for "
            "code capacity and binary BP2 for circuit noise."
        ),
    )
    parser.add_argument(
        "--bp_residual_hidden_dim",
        type=int,
        default=64,
        help="Hidden width of each orbit-shared neural BP residual MLP.",
    )
    parser.add_argument(
        "--bp_parameter_sharing",
        choices=["orbit", "global", "edge"],
        default="orbit",
        help=(
            "Neural BP sharing ablation. orbit is the BB-equivariant default; "
            "global is generic shared BP; edge intentionally breaks equivariance."
        ),
    )
    parser.add_argument(
        "--bp_residual_scale",
        type=float,
        default=2.0,
        help="Maximum tanh residual added to each BP log message.",
    )
    parser.add_argument(
        "--bp_max_relaxation_delta",
        type=float,
        default=0.5,
        help="Range around one available to the learned BP relaxation coefficient.",
    )
    parser.add_argument(
        "--bp_deep_supervision_weight",
        type=float,
        default=0.2,
        help="Weight of intermediate-iteration BB decoding losses.",
    )
    parser.add_argument(
        "--bp_gradient_clip",
        type=float,
        default=1.0,
        help="Gradient norm limit for BB neural BP.",
    )
    parser.add_argument(
        "--bb_channel",
        choices=["depolarizing", "independent_xz"],
        default="depolarizing",
        help="Code-capacity Pauli channel used by a BB experiment.",
    )
    parser.add_argument(
        "--x_error_rate",
        type=float,
        default=None,
        help="Independent X-component rate; defaults to --p.",
    )
    parser.add_argument(
        "--z_error_rate",
        type=float,
        default=None,
        help="Independent Z-component rate; defaults to --p.",
    )
    parser.add_argument("--bb_syndrome_loss_weight", type=float, default=1.0)
    parser.add_argument("--bb_logical_loss_weight", type=float, default=1.0)
    parser.add_argument("--bb_pauli_loss_weight", type=float, default=0.1)
    parser.add_argument("--bb_weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--matching_correlations",
        action="store_true",
        help=(
            "For --architecture=convgru_mwpm, enable PyMatching's correlated "
            "two-pass decoder. The default uses ordinary DEM-based MWPM so the "
            "neural residual model can learn correlations missed by it."
        ),
    )
    parser.add_argument(
        "--causal_edge_gru",
        action="store_true",
        help=(
            "For --architecture=convgru_weighted_mwpm, use only a forward "
            "ConvGRU. By default offline decoding concatenates forward and "
            "reverse ConvGRU features so every edge can use all rounds."
        ),
    )
    parser.add_argument(
        "--edge_hidden_channels",
        type=int,
        default=None,
        help=(
            "Hidden width of the symmetric DEM-edge MLP. Defaults to the "
            "ConvGRU feature width."
        ),
    )
    parser.add_argument(
        "--edge_delta_scale",
        type=float,
        default=6.0,
        help="Maximum absolute neural shift added to each DEM prior logit.",
    )
    parser.add_argument(
        "--edge_chunk_size",
        type=int,
        default=1024,
        help="Number of DEM edges scored together to control GPU memory.",
    )
    parser.add_argument(
        "--edge_entropy_weight",
        type=float,
        default=0.0,
        help="Optional entropy regularization coefficient for edge BCE.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed, including the Stim circuit sampler.",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs."
    )
    parser.add_argument(
        "--batches", type=int, default=128, help="Number of batches per epoch."
    )
    parser.add_argument(
        "--eval_batches",
        type=int,
        default=16,
        help="Number of batches used for evaluation each epoch.",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=1,
        help=(
            "Evaluate every N epochs (the final epoch is always evaluated). "
            "Useful when shot-specific PyMatching reconstruction is expensive."
        ),
    )
    parser.add_argument(
        "--final_eval_batches",
        type=int,
        default=None,
        help=(
            "Number of batches used for the final evaluation. Defaults to "
            "--eval_batches; set this higher for a lower-variance threshold point."
        ),
    )
    parser.add_argument(
        "--hybrid_calibration_batches",
        type=int,
        default=128,
        help=(
            "Fresh batches used to calibrate the selective residual gate for "
            "--architecture=convgru_mwpm. These samples are separate from the "
            "reported evaluation samples. Set to zero to keep the gate disabled "
            "and return the exact MWPM baseline."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument(
        "--loss_fn",
        type=str,
        default=None,
        choices=["ce", "dynamic", "edge_bce", "bb_coset"],
        help=(
            "Loss function type. Defaults to the degeneracy-aware bb_coset "
            "objective for bb_neural_bp and ce otherwise."
        ),
    )
    parser.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=[64, 64, 64],
        help="Number of channels per block.",
    )
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=[3, 3, 3],
        help="Number of layers per block.",
    )
    parser.add_argument(
        "--gru_channels",
        type=int,
        default=None,
        help=(
            "ConvGRU hidden width for --architecture=convgru, "
            "convgru_mwpm, or convgru_weighted_mwpm. "
            "Defaults to the last value in --channels."
        ),
    )
    parser.add_argument(
        "--gru_layers",
        type=int,
        default=1,
        help="Number of stacked ConvGRU layers.",
    )
    parser.add_argument(
        "--gru_kernel_size",
        type=int,
        default=3,
        help="Spatial kernel size used by ConvGRU gates.",
    )
    parser.add_argument(
        "--save_model", action="store_true", help="Save the trained model."
    )
    parser.add_argument(
        "--load_model",
        type=str,
        default=None,
        help=(
            "Checkpoint to resume. Restores model, optimizer, epoch, and plot "
            "history; --epochs specifies how many additional epochs to run."
        ),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Maximum learning rate (defaults: 1e-3 fresh, 1e-4 resumed).",
    )
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "none"],
        help="Mixed precision dtype.",
    )
    parser.add_argument(
        "--bp_orbit_embedding_dim",
        type=int,
        default=8,
        help=(
            "Width of the learned per-orbit embedding used by circuit-level "
            "BB neural BP. Zero ties every Tanner edge to one shared update."
        ),
    )
    parser.add_argument(
        "--bp_normalisation",
        type=float,
        default=0.625,
        help=(
            "Min-sum scaling factor for circuit-level BB neural BP. The "
            "default matches the classical ldpc BP+OSD baselines."
        ),
    )
    parser.add_argument(
        "--bp_no_gradient_checkpoint",
        action="store_true",
        help=(
            "Keep every unrolled BP iteration's activations. Circuit-level "
            "graphs have hundreds of thousands of edges, so this usually "
            "exhausts GPU memory; checkpointing is on by default."
        ),
    )
    parser.add_argument(
        "--bb_circuit_noise_model",
        choices=BB_CIRCUIT_NOISE_MODELS,
        default="legacy",
        help=(
            "BB circuit channel profile: legacy preserves previous runs; "
            "standard implements arXiv:2607.05897 Table II; si1000 implements "
            "its modified SI1000 Table III."
        ),
    )
    parser.add_argument(
        "--bb_idle_error_rate",
        type=float,
        default=None,
        help=(
            "Legacy-only depolarizing rate for data qubits sitting out a CNOT "
            "layer. Defaults to zero. standard and si1000 derive full tick-aware "
            "idle channels from --p and reject incompatible overrides."
        ),
    )
    parser.add_argument(
        "--bb_osd_eval_shots",
        type=int,
        default=0,
        help=(
            "Shots per evaluation additionally decoded through ordered "
            "statistics post-processing, reported as Neural-BP+OSD versus "
            "BP+OSD on the same shots. Plain BP is a weak quantum LDPC "
            "decoder, so this is the comparison the literature uses. When "
            "enabled, best-checkpoint selection uses the paired OSD gain."
        ),
    )
    parser.add_argument(
        "--bb_osd_method",
        type=str,
        default="OSD_CS",
        choices=["OSD_0", "OSD_CS"],
        help="Ordered-statistics variant used by --bb_osd_eval_shots.",
    )
    parser.add_argument(
        "--bb_osd_order",
        type=int,
        default=7,
        help="Combination-sweep depth for --bb_osd_method=OSD_CS.",
    )

    args = parser.parse_args()
    bb_architecture = args.architecture == "bb_neural_bp"
    if bb_architecture and args.code not in {"bb72", "bb144"}:
        parser.error("--architecture=bb_neural_bp requires --code=bb72 or bb144.")
    bb_circuit = bb_architecture and args.noise_model == "circuit"
    if bb_circuit:
        try:
            bb_noise_profile = resolve_bb_circuit_noise_profile(
                args.bb_circuit_noise_model,
                base_error_rate=args.p,
                measurement_error_rate=args.measurement_error_rate,
                idle_error_rate=args.bb_idle_error_rate,
            )
        except ValueError as exc:
            parser.error(str(exc))
        args.bb_circuit_noise_model = bb_noise_profile.name
        # Store the effective rates in the existing fields so logs, metadata,
        # and old analysis code remain explicit. SI1000's additional 2p
        # resonator-idle channel is recorded separately by the experiment.
        args.measurement_error_rate = bb_noise_profile.measurement_error_rate
        args.bb_idle_error_rate = bb_noise_profile.gate_idle_error_rate
    else:
        if args.bb_circuit_noise_model != "legacy":
            parser.error(
                "--bb_circuit_noise_model=standard/si1000 requires "
                "--code=bb72 or bb144 --architecture=bb_neural_bp "
                "--noise_model=circuit."
            )
        if args.measurement_error_rate is None:
            args.measurement_error_rate = 0.0 if bb_architecture else 0.01
        if args.bb_idle_error_rate is None:
            args.bb_idle_error_rate = 0.0
    args.loss_fn = args.loss_fn or ("bb_coset" if bb_architecture else "ce")
    if bb_circuit:
        # Rounds default to the code distance, the usual memory-experiment
        # choice, rather than to the toric lattice size.
        default_rounds = _BB_DISTANCE[args.code]
        rounds = default_rounds if args.rounds is None else args.rounds
    elif bb_architecture:
        rounds = 1 if args.rounds is None else args.rounds
    else:
        rounds = args.L if args.rounds is None else args.rounds
    if rounds < 1:
        parser.error("--rounds must be positive.")
    if not 0.0 <= args.p <= 1.0:
        parser.error("--p must be in [0, 1].")
    if not 0.0 <= args.measurement_error_rate <= 1.0:
        parser.error("--measurement_error_rate must be in [0, 1].")
    if args.eval_batches < 1:
        parser.error("--eval_batches must be positive.")
    if args.epochs < 1:
        parser.error("--epochs must be positive.")
    if args.batches < 1:
        parser.error("--batches must be positive.")
    if args.batch_size < 1:
        parser.error("--batch_size must be positive.")
    if args.eval_every < 1:
        parser.error("--eval_every must be positive.")
    if args.final_eval_batches is not None and args.final_eval_batches < 1:
        parser.error("--final_eval_batches must be positive.")
    if args.hybrid_calibration_batches < 0:
        parser.error("--hybrid_calibration_batches must be non-negative.")
    if not args.channels or len(args.channels) != len(args.depths):
        parser.error("--channels and --depths must be non-empty and have equal length.")
    if any(channel < 1 for channel in args.channels):
        parser.error("--channels values must be positive.")
    if any(depth < 1 for depth in args.depths):
        parser.error("--depths values must be positive.")
    if args.gru_channels is not None and args.gru_channels < 1:
        parser.error("--gru_channels must be positive.")
    if args.gru_layers < 1:
        parser.error("--gru_layers must be positive.")
    if args.gru_kernel_size < 1:
        parser.error("--gru_kernel_size must be positive.")
    if args.edge_hidden_channels is not None and args.edge_hidden_channels < 1:
        parser.error("--edge_hidden_channels must be positive.")
    if args.edge_delta_scale <= 0.0:
        parser.error("--edge_delta_scale must be positive.")
    if args.edge_chunk_size < 1:
        parser.error("--edge_chunk_size must be positive.")
    if args.edge_entropy_weight < 0.0:
        parser.error("--edge_entropy_weight must be non-negative.")
    if args.lr is not None and args.lr <= 0.0:
        parser.error("--lr must be positive.")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be non-negative.")
    if args.x_error_rate is not None and not 0.0 <= args.x_error_rate <= 1.0:
        parser.error("--x_error_rate must be in [0, 1].")
    if args.z_error_rate is not None and not 0.0 <= args.z_error_rate <= 1.0:
        parser.error("--z_error_rate must be in [0, 1].")
    if args.bp_iterations < 1:
        parser.error("--bp_iterations must be positive.")
    if args.bp_residual_hidden_dim < 1:
        parser.error("--bp_residual_hidden_dim must be positive.")
    if args.bp_residual_scale < 0.0:
        parser.error("--bp_residual_scale must be non-negative.")
    if not 0.0 <= args.bp_max_relaxation_delta < 1.0:
        parser.error("--bp_max_relaxation_delta must be in [0, 1).")
    if args.bp_deep_supervision_weight < 0.0:
        parser.error("--bp_deep_supervision_weight must be non-negative.")
    if args.bp_gradient_clip <= 0.0:
        parser.error("--bp_gradient_clip must be positive.")
    if any(
        value < 0.0
        for value in (
            args.bb_syndrome_loss_weight,
            args.bb_logical_loss_weight,
            args.bb_pauli_loss_weight,
            args.bb_weight_decay,
        )
    ):
        parser.error("BB loss weights and --bb_weight_decay must be non-negative.")

    if bb_architecture:
        if args.noise_model == "phenomenological":
            parser.error(
                "BB neural BP supports --noise_model=capacity (four-state BP4 "
                "on the code Tanner graph) or --noise_model=circuit (binary BP "
                "on the Stim detector error model). Phenomenological BB noise "
                "is not implemented."
            )
        if args.noise_model == "circuit":
            if args.loss_fn != "bb_coset":
                parser.error(
                    "--architecture=bb_neural_bp requires --loss_fn=bb_coset."
                )
            if args.matching_correlations:
                parser.error("--matching_correlations does not apply to neural BP.")
            if args.bp_orbit_embedding_dim < 0:
                parser.error("--bp_orbit_embedding_dim must be non-negative.")
            if not 0.0 < args.bp_normalisation <= 1.0:
                parser.error("--bp_normalisation must be in (0, 1].")
            if not 0.0 <= args.bb_idle_error_rate <= 1.0:
                parser.error("--bb_idle_error_rate must be in [0, 1].")
            if (
                args.p == 0.0
                and args.measurement_error_rate == 0.0
                and args.bb_idle_error_rate == 0.0
            ):
                parser.error(
                    "BB circuit training needs at least one non-zero gate, "
                    "measurement, or idle error rate; the all-zero circuit has "
                    "no DEM variables to learn."
                )
            if args.bb_osd_eval_shots < 0:
                parser.error("--bb_osd_eval_shots must be non-negative.")
            if args.bb_osd_order < 0:
                parser.error("--bb_osd_order must be non-negative.")
            if args.amp_dtype != "none":
                parser.error(
                    "BB circuit neural BP currently requires --amp_dtype=none; "
                    "mixed precision is not silently applied to sparse parity loss."
                )
            if args.bb_channel != "depolarizing":
                parser.error(
                    "--bb_channel selects a code-capacity Pauli channel and "
                    "does not apply to circuit-level noise."
                )
            args.rounds = rounds
            run_bb_circuit_experiment(args)
            return
        if rounds != 1:
            parser.error(
                "BB code-capacity decoding has one perfect syndrome: --rounds=1."
            )
        if args.measurement_error_rate != 0.0:
            parser.error(
                "BB code capacity uses perfect checks: "
                "--measurement_error_rate must be 0."
            )
        if args.loss_fn != "bb_coset":
            parser.error("--architecture=bb_neural_bp requires --loss_fn=bb_coset.")
        if args.matching_correlations:
            parser.error("--matching_correlations does not apply to neural BP.")
        run_bb_experiment(args)
        return

    if args.code != "toric":
        parser.error("--code=bb72/bb144 requires --architecture=bb_neural_bp.")
    if args.loss_fn == "bb_coset":
        parser.error("--loss_fn=bb_coset is only valid with bb_neural_bp.")

    matching_architectures = {"convgru_mwpm", "convgru_weighted_mwpm"}
    if args.architecture in matching_architectures and args.noise_model != "circuit":
        parser.error(
            f"--architecture={args.architecture} requires --noise_model=circuit."
        )
    if args.matching_correlations and args.architecture != "convgru_mwpm":
        parser.error(
            "--matching_correlations is only valid with "
            "the legacy --architecture=convgru_mwpm. Per-shot rebuilt weights "
            "cannot retain PyMatching's correlation metadata."
        )
    if args.architecture == "convgru_mwpm" and args.loss_fn != "ce":
        parser.error(
            "--architecture=convgru_mwpm requires --loss_fn=ce. "
            "Inverse-frequency dynamic loss overweights rare residual classes "
            "and can make the hybrid worse than its MWPM fallback."
        )
    if args.architecture == "convgru_weighted_mwpm" and args.loss_fn != "edge_bce":
        parser.error(
            "--architecture=convgru_weighted_mwpm requires " "--loss_fn=edge_bce."
        )
    if args.architecture != "convgru_weighted_mwpm" and args.loss_fn == "edge_bce":
        parser.error(
            "--loss_fn=edge_bce is only valid with "
            "--architecture=convgru_weighted_mwpm."
        )

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    """Initialize the toric stabilizer code."""
    code = Toric2DCode(args.L)

    # in_channels is always 2 for the vertex/face detector sectors.
    in_channels = 2
    if args.architecture == "convgru_weighted_mwpm":
        matching_circuit = generate_toric_memory_circuit(
            code,
            rounds=rounds,
            gate_error_rate=args.p,
            measurement_error_rate=args.measurement_error_rate,
        )
        detector_error_model = matching_circuit.detector_error_model(
            decompose_errors=True
        ).flattened()
        network = RecurrentEdgeWeightNetwork(
            channels=args.channels,
            depths=args.depths,
            lattice_size=args.L,
            in_channels=in_channels,
            gru_channels=args.gru_channels,
            gru_layers=args.gru_layers,
            gru_kernel_size=args.gru_kernel_size,
            bidirectional=not args.causal_edge_gru,
            edge_hidden_channels=args.edge_hidden_channels,
            edge_delta_scale=args.edge_delta_scale,
            edge_chunk_size=args.edge_chunk_size,
        )
        decoder = NeuralWeightedMatchingDecoder(
            edge_network=network,
            detector_error_model=detector_error_model,
            lattice_size=args.L,
            rounds=rounds,
            num_observables=2 * code.k,
        )
    elif args.architecture == "convgru_mwpm":
        network = RecurrentResidualEND2D(
            channels=args.channels,
            depths=args.depths,
            lattice_size=args.L,
            in_channels=in_channels,
            gru_channels=args.gru_channels,
            gru_layers=args.gru_layers,
            gru_kernel_size=args.gru_kernel_size,
        )
        matching_circuit = generate_toric_memory_circuit(
            code,
            rounds=rounds,
            gate_error_rate=args.p,
            measurement_error_rate=args.measurement_error_rate,
        )
        decoder = MatchingResidualDecoder(
            residual_decoder=network,
            circuit=matching_circuit,
            num_observables=2 * code.k,
            enable_correlations=args.matching_correlations,
        )
    elif args.architecture == "convgru":
        pooling = TranslationalEquivariantPooling2D(args.L)
        network = RecurrentEND2D(
            channels=args.channels,
            depths=args.depths,
            lattice_size=args.L,
            in_channels=in_channels,
            gru_channels=args.gru_channels,
            gru_layers=args.gru_layers,
            gru_kernel_size=args.gru_kernel_size,
        )
        decoder = Decoder(network=network, pooling=pooling, ensemble=None)
    else:
        pooling = TranslationalEquivariantPooling2D(args.L)
        network = TransformedEND3D(
            channels=args.channels,
            depths=args.depths,
            lattice_size=args.L,
            in_channels=in_channels,
        )
        decoder = Decoder(network=network, pooling=pooling, ensemble=None)

    decoder.to(device)

    """Instantiate Optimizer, Scheduler and Loss."""
    optimizers, schedulers = [], []

    lr = args.lr if args.lr is not None else (1e-4 if args.load_model else 1e-3)

    optimizers.append(
        opt := torch.optim.AdamW(params=network.parameters(), lr=lr, weight_decay=1e-4)
    )
    schedulers.append(
        torch.optim.lr_scheduler.OneCycleLR(
            optimizer=opt, max_lr=lr, epochs=args.epochs, steps_per_epoch=args.batches
        )
    )

    if args.loss_fn == "ce":
        criterion = nn.CrossEntropyLoss()
    elif args.loss_fn == "dynamic":
        criterion = DynamicCELoss(2 ** (2 * code.k), device)
    else:
        criterion = EdgeBCELoss(entropy_weight=args.edge_entropy_weight)

    """Setup Trainer and start training"""
    logging.info("Start Training")

    curr_time = datetime.datetime.now()
    edge_hidden_channels = args.edge_hidden_channels or (
        (args.gru_channels or args.channels[-1]) * (1 if args.causal_edge_gru else 2)
    )
    run_slug = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        (
            f"{args.noise_model}_{args.architecture}_L{args.L}_r{rounds}_p{args.p:g}_"
            f"q{args.measurement_error_rate:g}_lr{lr:g}_"
            f"bs{args.batch_size}_b{args.batches}_eb{args.eval_batches}_"
            f"ee{args.eval_every}_"
            f"feb{args.final_eval_batches or args.eval_batches}_"
            f"loss{args.loss_fn}_"
            f"ch{'-'.join(map(str, args.channels))}_d{'-'.join(map(str, args.depths))}"
            + (
                f"_gru{args.gru_channels or args.channels[-1]}x{args.gru_layers}"
                f"_gk{args.gru_kernel_size}"
                if args.architecture
                in {"convgru", "convgru_mwpm", "convgru_weighted_mwpm"}
                else ""
            )
            + (
                f"_matching-{'corr' if args.matching_correlations else 'standard'}"
                if args.architecture == "convgru_mwpm"
                else ""
            )
            + (
                f"_gatecal{args.hybrid_calibration_batches}"
                if args.architecture == "convgru_mwpm"
                else ""
            )
            + (
                f"_edge{'causal' if args.causal_edge_gru else 'bidir'}"
                f"_eh{edge_hidden_channels}"
                f"_eds{args.edge_delta_scale:g}_ecs{args.edge_chunk_size}"
                f"_ent{args.edge_entropy_weight:g}"
                if args.architecture == "convgru_weighted_mwpm"
                else ""
            )
            + ("_resume" if args.load_model else "")
        ),
    )
    output_dir = os.path.join(
        os.getcwd(),
        "outputs",
        curr_time.strftime("%Y-%m-%d"),
        f"{curr_time.strftime('%H-%M-%S-%f')}_{run_slug}",
    )
    os.makedirs(output_dir, exist_ok=True)

    if (
        hasattr(network, "conv_in")
        and network.conv_in.__class__.__name__ == "AConvCircular3D"
    ):
        conv_in = network.conv_in
        attention = (
            f"enabled heads={conv_in.number_heads}, "
            f"key_depths={conv_in.key_depths}, "
            f"attn_channels={conv_in.attention_channels}"
        )
    else:
        attention = "disabled"

    trainer = Trainer(
        model=decoder,
        loss_function=criterion,
        optimizers=optimizers,
        schedulers=schedulers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        batches=args.batches,
        eval_batches=args.eval_batches,
        eval_every=args.eval_every,
        final_eval_batches=args.final_eval_batches,
        hybrid_calibration_batches=args.hybrid_calibration_batches,
        amp_dtype=args.amp_dtype,
        lattice_size=args.L,
        channels=args.channels,
        depths=args.depths,
        architecture=args.architecture,
        recurrent=(
            f"ConvGRU channels={args.gru_channels or args.channels[-1]}, "
            f"layers={args.gru_layers}, kernel={args.gru_kernel_size}"
            if args.architecture in {"convgru", "convgru_mwpm", "convgru_weighted_mwpm"}
            else None
        ),
        attention=attention,
        save_model=args.save_model,
        load_model_path=args.load_model,
        save_directory=output_dir,
    )

    logging.info(
        f"Lattice size: {args.L}, Rounds: {rounds}, Error rate: {args.p}, "
        f"Noise Model: {args.noise_model}, Measurement Error: "
        f"{args.measurement_error_rate}, "
        f"{'Additional epochs' if args.load_model else 'Epochs'}: {args.epochs}"
    )
    batch_log = (
        f"Batch size: {args.batch_size}, Training batches: {args.batches}, "
        f"Evaluation batches: {args.eval_batches} every {args.eval_every} epoch(s), "
        f"Final evaluation batches: "
        f"{args.final_eval_batches or args.eval_batches}"
    )
    if args.architecture == "convgru_mwpm":
        batch_log += f", Hybrid calibration batches: {args.hybrid_calibration_batches}"
    logging.info(batch_log)
    logging.info(
        f"Samples - training per epoch: {args.batch_size * args.batches}, "
        f"evaluation per epoch: {args.batch_size * args.eval_batches}, "
        f"final evaluation: "
        f"{args.batch_size * (args.final_eval_batches or args.eval_batches)}"
    )
    logging.info(
        f"Architecture - Type: {args.architecture}, Channels: {args.channels}, "
        f"Depths: {args.depths}"
    )

    # Check if network is using Attention
    if args.architecture in {
        "convgru",
        "convgru_mwpm",
        "convgru_weighted_mwpm",
    }:
        logging.info(
            "Temporal model: ConvGRU | "
            f"Hidden channels: {args.gru_channels or args.channels[-1]} | "
            f"Layers: {args.gru_layers} | Kernel: {args.gru_kernel_size}"
        )
        logging.info("Attention: Disabled (shared 2D circular CNN encoder)")
        if args.architecture == "convgru_mwpm":
            logging.info(
                "Global decoder: Stim DEM PyMatching | "
                f"Correlated matching: {args.matching_correlations} | "
                "Neural target: residual logical class | "
                "Selective gate: calibrated with MWPM fallback"
            )
        elif args.architecture == "convgru_weighted_mwpm":
            logging.info(
                "Global decoder: shot-conditioned standard MWPM | "
                f"Temporal context: {'causal' if args.causal_edge_gru else 'bidirectional'} | "
                "Neural target: fired DEM edge parity | "
                f"Edge delta scale: {args.edge_delta_scale:g} | "
                f"Edge chunk: {args.edge_chunk_size} | "
                f"Entropy weight: {args.edge_entropy_weight:g}"
            )
    elif attention != "disabled":
        logging.info(
            f"Attention: Enabled | Heads: {conv_in.number_heads} | Key Depths: {conv_in.key_depths} | Attn Channels: {conv_in.attention_channels}"
        )
    else:
        logging.info("Attention: Disabled (Pure CNN)")

    """Start training."""
    trainer.train(
        code=code,
        error_rate=args.p,
        noise_model=args.noise_model,
        measurement_error_rate=args.measurement_error_rate,
        rounds=rounds,
        seed=args.seed,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    main()
