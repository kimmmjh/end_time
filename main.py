import os
import torch
import logging
import argparse
import datetime

from torch import nn
from models import Decoder
from models.loss_functions import DynamicCELoss
from models._the_end_3d import TransformedEND3D
from models.pooling_layers import TranslationalEquivariantPooling2D
from src import Trainer
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
        "--epochs", type=int, default=100, help="Number of training epochs."
    )
    parser.add_argument(
        "--batches", type=int, default=128, help="Number of batches per epoch."
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
        "--save_model", action="store_true", help="Save the trained model."
    )
    parser.add_argument(
        "--load_model", type=str, default=None, help="Path to load model."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (defaults: 1e-3 from scratch, 1e-4 fine-tuning).",
    )

    args = parser.parse_args()

    """Init variables for later use."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    """Initialize the stabilizer Code."""
    code = Toric2DCode(args.L)

    """Make Decoder Model."""
    pooling = TranslationalEquivariantPooling2D(args.L)

    # Instantiate the 3D Neural Network
    # in_channels is always 2 (X and Z) because Time is treated as a spatial dimension
    in_channels = 2
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
    output_dir = os.path.join(
        os.getcwd(),
        "outputs",
        curr_time.strftime("%Y-%m-%d"),
        curr_time.strftime("%H-%M-%S"),
    )
    os.makedirs(output_dir, exist_ok=True)

    # Convert args to an object similar to Hydras to pass minimally to Trainer without refactoring Trainer just yet
    class ArgsMock:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    trainer_args = ArgsMock(batch_size=args.batch_size, noise_model=args.noise_model)
    trainer_args.default = ArgsMock(epochs=args.epochs, batches=args.batches)

    trainer = Trainer(
        model=decoder,
        loss_function=criterion,
        optimizers=optimizers,
        schedulers=schedulers,
        args=trainer_args,
        save_model=args.save_model,
        load_model_path=args.load_model,
        save_directory=output_dir,
    )

    logging.info(
        f"Lattice size: {args.L}, Error rate: {args.p}, Noise Model: {args.noise_model}, Measurement Error: {args.measurement_error_rate}, Epochs: {args.epochs}"
    )
    logging.info(f"Architecture - Channels: {args.channels}, Depths: {args.depths}")

    # Check if network is using Attention
    if (
        hasattr(network, "conv_in")
        and network.conv_in.__class__.__name__ == "AConvCircular3D"
    ):
        conv_in = network.conv_in
        logging.info(
            f"Attention Mechanism: Enabled | Heads: {conv_in.number_heads} | Key Depths: {conv_in.key_depths} | Attn Channels: {conv_in.attention_channels}"
        )
    else:
        logging.info("Attention Mechanism: Disabled (Pure CNN)")

    """Start training."""
    trainer.train(
        code=code,
        error_rate=args.p,
        noise_model=args.noise_model,
        measurement_error_rate=args.measurement_error_rate,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    main()
