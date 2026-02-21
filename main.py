import os

import torch
import logging
import hydra
import omegaconf

from torch import nn
from hydra.utils import instantiate
from models import Decoder, MViT
from models.loss_functions import DynamicCELoss
from src import Trainer
from panqec.codes import StabilizerCode


@hydra.main(config_path="config", config_name="2d", version_base="1.2")
def main(args) -> None:
    """
    Start the experiments for decoder training.

    Note that we use hydra for experimental setup and configs, however, it is not required to run the experiments.
    Simply remove the hydra part and set the arguments in the file or use another parser method.
    All arguments needed start with 'args.'.

    :param args: The parsed hydra arguments.
    """
    """Init variables for later use."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    L: int = args.default.L  # parse lattice size (code is symmetric and as such follows L x L x ..).
    p: float = args.default.p  # parse the error rate [0,1).

    noise_model = getattr(args.default, "noise_model", "capacity")
    circuit_noise = getattr(args.default, "circuit_noise", False)
    if circuit_noise:
        noise_model = "circuit"
        
    measurement_error_rate = getattr(args.default, "measurement_error_rate", 0.0)
    epochs = args.default.epochs
    logging.info(f"Lattice size: {L}, Error rate: {p}, Noise Model: {noise_model}, Measurement Error: {measurement_error_rate}, Epochs: {epochs}")

    """Initialize the stabilizer Code."""
    code: StabilizerCode = instantiate(args.default.code, L)  # Instantiate the error correcting code using panqec.

    """Make Decoder Model."""
    pooling: nn.Module = instantiate(args.default.pooling, L)  # Instantiate the pooling approach. Pooling layers can be found in 'models/pooling_layers'.
    
    in_channels = 2 * L if noise_model in ["circuit", "phenomenological"] else 2
    net_args = args.net if "net" in args and args.net is not None else {}
    network: nn.Module = instantiate(args.default.network, **net_args, lattice_size=L, in_channels=in_channels)  # Instantiate the network decoder. Decoders can be found in 'models'.
    ensemble = MViT(
        lattice_size=L,
        patch_size=L,
    ) if args.default.network.ensemble else None  # Boolean value determines if the ensemble method is used for decoding.

    decoder = Decoder(network=network, pooling=pooling, ensemble=ensemble)
    decoder.to(device)

    """Instantiate Optimizer, Scheduler and Loss."""
    optimizers, schedulers = [], []

    optimizers.append(opt := torch.optim.AdamW(params=network.parameters(), lr=1e-3, weight_decay=1e-4))
    schedulers.append(torch.optim.lr_scheduler.OneCycleLR(
        optimizer=opt,
        max_lr=0.001, # Reduced from 0.01 for stability (NaN fix)
        epochs=args.default.epochs,  # Define the amount of epochs to train (int).
        steps_per_epoch=args.default.batches  # Define the amount of batches per epoch (int).
    ))

    if args.default.network.ensemble:
        optimizers.append(ens_opt := torch.optim.AdamW(params=ensemble.parameters(), lr=1e-3, weight_decay=1e-4))
        schedulers.append(torch.optim.lr_scheduler.OneCycleLR(
            optimizer=ens_opt,
            max_lr=0.001,
            epochs=args.default.epochs,
            steps_per_epoch=args.default.batches
        ))

    criterion = DynamicCELoss(2**(2*code.k), device) # nn.CrossEntropyLoss()

    """Setup Trainer and start training"""
    logging.info("Start Training")

    save_model = args.save_model if "save_model" in args else False
    load_model = args.load_model if "load_model" in args else None

    # Get Hydra output directory
    try:
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    except (ValueError, IndexError):
        # Fallback if not running with Hydra or if config not set
        output_dir = os.getcwd()

    trainer = Trainer(
        model=decoder,
        loss_function=criterion,
        optimizers=optimizers,
        schedulers=schedulers,
        args=args,
        save_model=save_model,
        load_model_path=load_model,
        save_directory=output_dir
    )
    """Start training."""
    trainer.train(
        code=code,
        error_rate=p,
        noise_model=noise_model,
        measurement_error_rate=measurement_error_rate,
    )


if __name__ == '__main__':
    logger = logging.Logger("default_logger")
    logger.setLevel(logging.INFO)
    main()
