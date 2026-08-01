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
    RecurrentEND2D,
    RecurrentResidualEND2D,
)
from models.loss_functions import DynamicCELoss
from models._the_end_3d import TransformedEND3D
from models.pooling_layers import TranslationalEquivariantPooling2D
from src import Trainer
from src.stim_utils import generate_toric_memory_circuit
from panqec.codes import Toric2DCode


def main() -> None:
    """
    Start the experiments for decoder training.
    """
    parser = argparse.ArgumentParser(description="Neural Decoder for Toric Code")
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
        default=0.01,
        help="Measurement error rate [0,1).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Syndrome rounds (defaults to L).",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="cnn3d",
        choices=["cnn3d", "convgru", "convgru_mwpm"],
        help=(
            "Temporal architecture. cnn3d treats time as a third spatial axis; "
            "convgru applies a shared equivariant 2D CNN to each round and "
            "recurrently accumulates the rounds; convgru_mwpm lets a Stim DEM "
            "PyMatching decoder do global pairing and trains the equivariant "
            "ConvGRU to predict its residual logical class."
        ),
    )
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
        default="ce",
        choices=["ce", "dynamic"],
        help="Loss function type.",
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
            "ConvGRU hidden width for --architecture=convgru or "
            "convgru_mwpm. "
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

    args = parser.parse_args()
    rounds = args.L if args.rounds is None else args.rounds
    if rounds < 1:
        parser.error("--rounds must be positive.")
    if not 0.0 <= args.p <= 1.0:
        parser.error("--p must be in [0, 1].")
    if not 0.0 <= args.measurement_error_rate <= 1.0:
        parser.error("--measurement_error_rate must be in [0, 1].")
    if args.eval_batches < 1:
        parser.error("--eval_batches must be positive.")
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
    if args.architecture == "convgru_mwpm" and args.noise_model != "circuit":
        parser.error(
            "--architecture=convgru_mwpm requires --noise_model=circuit."
        )
    if args.matching_correlations and args.architecture != "convgru_mwpm":
        parser.error(
            "--matching_correlations is only valid with "
            "--architecture=convgru_mwpm."
        )
    if args.architecture == "convgru_mwpm" and args.loss_fn != "ce":
        parser.error(
            "--architecture=convgru_mwpm requires --loss_fn=ce. "
            "Inverse-frequency dynamic loss overweights rare residual classes "
            "and can make the hybrid worse than its MWPM fallback."
        )

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    """Initialize the stabilizer Code."""
    code = Toric2DCode(args.L)

    """Make Decoder Model."""
    pooling = TranslationalEquivariantPooling2D(args.L)

    # in_channels is always 2 for the vertex/face detector sectors.
    in_channels = 2
    if args.architecture == "convgru_mwpm":
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
    else:
        criterion = DynamicCELoss(2 ** (2 * code.k), device)

    """Setup Trainer and start training"""
    logging.info("Start Training")

    curr_time = datetime.datetime.now()
    run_slug = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        (
            f"{args.noise_model}_{args.architecture}_L{args.L}_r{rounds}_p{args.p:g}_"
            f"q{args.measurement_error_rate:g}_lr{lr:g}_"
            f"bs{args.batch_size}_b{args.batches}_eb{args.eval_batches}_"
            f"feb{args.final_eval_batches or args.eval_batches}_"
            f"loss{args.loss_fn}_"
            f"ch{'-'.join(map(str, args.channels))}_d{'-'.join(map(str, args.depths))}"
            + (
                f"_gru{args.gru_channels or args.channels[-1]}x{args.gru_layers}"
                f"_gk{args.gru_kernel_size}"
                if args.architecture in {"convgru", "convgru_mwpm"}
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
            if args.architecture in {"convgru", "convgru_mwpm"}
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
    logging.info(
        f"Batch size: {args.batch_size}, Training batches: {args.batches}, "
        f"Evaluation batches: {args.eval_batches}, Final evaluation batches: "
        f"{args.final_eval_batches or args.eval_batches}, Hybrid calibration "
        f"batches: {args.hybrid_calibration_batches}"
    )
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
    if args.architecture in {"convgru", "convgru_mwpm"}:
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
