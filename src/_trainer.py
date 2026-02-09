import torch
import os
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import wandb
from src.metrics import WandbMetrics
from typing import Callable, Type
from ._data_generator import DataGenerator
from panqec.codes import StabilizerCode
import logging
import time
from tqdm import tqdm
import matplotlib.pyplot as plt



class Trainer:
    """A trainer that generates batches on the fly."""
    model: nn.Module
    optimizers: list[Optimizer]
    schedulers: list[LRScheduler]
    evaluator: Callable
    criterion: nn.Module
    training_samples: int

    _output: Type[Callable]  # An output function to either print or log progress.

    """Parameters for the training."""
    _batch_size: int
    _num_epochs: int
    _num_batches: int

    """Variables for saving models."""
    _save_model: bool
    _save_directory: str

    def __init__(
            self,
            model: nn.Module,
            loss_function: nn.Module,
            optimizers: list[Optimizer],
            schedulers: list[LRScheduler],
            args,
            verbose: bool = False,
            save_model: bool = False,
            load_model_path: str = None,
            save_directory: str = None,
    ) -> None:
        """
        Initialize the trainer object.

        :param model: The decoder model.
        :param loss_function: The Loss function.
        :param optimizers: The optimizer.
        :param schedulers: The scheduler.
        :param args: Arguments for the Trainer.
        :param verbose: Whether the trainer should print progress or log it.
        :param save_model: If model should be saved.
        :return: The trained decoder and train / validation values.
        """
        self.model = model

        self._output = print if verbose else logging.info
        self.criterion = loss_function
        self.optimizers = optimizers
        self.schedulers = schedulers

        self.scaler = torch.cuda.amp.GradScaler()

        self._num_batches = args.default.batches
        self._num_epochs = args.default.epochs
        self._batch_size = args.batch_size
        self._save_directory = save_directory if save_directory else wandb.run.dir
        self._save_model = save_model
        
        self.history = {'loss': [], 'accuracy': []}
        self.start_epoch = 0
        if load_model_path is not None:
            self.load_model(load_model_path)

    def train(
            self, *,
            code: StabilizerCode,
            error_rate: float,
            circuit_noise: bool = False,
            measurement_error_rate: float = 0.0,
    ) -> None:
        """
        Train the neural decoder on dynamically generated data.

        :param code: The code to train the decoder on.
        :param error_rate: The error rate to train the decoder on.
        """
        """Extract parameters."""
        torch.backends.cudnn.benchmark = True  # Enable cuda to find the best tuner for hardware.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_generator = DataGenerator(
            code=code,
            verbose=False,
            error_rate=error_rate,
            batch_size=self._batch_size,
            circuit_noise=circuit_noise,
            measurement_error_rate=measurement_error_rate,
        )

        """Start Training."""
        for epoch in range(self.start_epoch, self._num_epochs):
            self._output(f"{'=' * 18}")
            self._output(f"Starting Epoch {epoch}.")
            epoch_start = time.time()

            """Train Model"""
            loss, _ = self._process_batches(data_generator, device, self._num_batches)
            epoch_time = time.time() - epoch_start

            """Evaluate model."""
            self._output("Evaluating Model.")
            with torch.no_grad():
                _, (y_pred, y_true) = self._process_batches(data_generator, device, 1, train=False)

            """Record evaluation Metrics."""
            metrics = WandbMetrics.get_metrics(
                y_pred=y_pred,
                y_true=y_true,
                loss=loss,
                learning_rate=self.schedulers[0].optimizer.param_groups[0]['lr'],
                epoch_duration=epoch_time,
            )
            self._output(str(metrics.__dict__))
            wandb.log(metrics.__dict__)
            
            # Update history and save plots
            self.history['loss'].append(float(metrics.loss))
            self.history['accuracy'].append(float(metrics.accuracy))
            self.save_plots(path=self._save_directory)

        """Sve the finished model."""
        if self._save_model:
            self._output("Saving Model.")
            self.save_model(path=self._save_directory, model_name=wandb.run.name, epoch=self._num_epochs-1)

    def _process_batches(
            self,
            data_generator: DataGenerator,
            device: torch.device,
            batches: int,
            train: bool = True,
    ) -> tuple[float, tuple[Tensor, Tensor]]:
        """
        Process epoch and log if it is testing.
        
        :param data_generator: The data generator object.
        :param device: The device to run the loop on.
        :param batches: The amount of batches to train.
        :param train: Whether its training or not.
        :returns: The loss and a tuple of (y_pred, y_true).
        :raises ValueError: If loss is nan.
        """

        loss = 0.
        iterator = range(batches)
        if train:
             iterator = tqdm(iterator, desc="Training")
             
        for _ in iterator:
            X, y = data_generator.generate_batch(use_qmc=train, device=device)
            """Zero out the gradient for all optimizers."""
            for optimizer in self.optimizers:
                optimizer.zero_grad()

            """Forward pass."""
            with torch.autocast("cuda"):
                y_pred = self.model(X)
                loss_c = self.criterion(y_pred, y)

            if train:
                """Record loss."""
                loss += loss_c.item()

                """Backward pass."""
                self.scaler.scale(loss_c).backward()

                """Update weights and step schedulers."""
                for optimizer in self.optimizers:
                    self.scaler.unscale_(optimizer)

                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                for optimizer in self.optimizers:
                    self.scaler.step(optimizer)

                self.scaler.update()
                for scheduler in self.schedulers:
                    scheduler.step()
        return loss / batches, (y_pred, y)

    def save_model(self, path: str = ".", model_name: str = "model", epoch: int = 0) -> None:
        """
        Save the current model and training state.

       :param path: The path to save it to.
       :param model_name: The name of the saved model.
       :param epoch: The current epoch.
        """
        
        # Helper to get state dicts
        optim_states = [opt.state_dict() for opt in self.optimizers]
        sched_states = [sch.state_dict() for sch in self.schedulers]
        
        # If model is DataParallel, access module
        model_state = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model_state,
            'optimizer_states': optim_states,
            'scheduler_states': sched_states,
            'history': self.history,
        }
        
        torch.save(checkpoint, f"{path}/{model_name}.pt")

    def load_model(self, path: str) -> None:
        """
        Load a checkpoint and restore training state.
        :param path: Path to the .pt checkpoint file
        """
        self._output(f"Loading checkpoint from {path}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=device)
        
        # 1. Restore Model
        # Handle case where checkpoint was DataParallel but current model isn't, or vice-versa
        # For simplicity, assume matching architecture
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            # Try to handle 'module.' prefix mismatch
            state_dict = checkpoint['model_state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("module.", "") if k.startswith("module.") else k 
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)
            
        # 2. Restore Optimizers
        if 'optimizer_states' in checkpoint:
            for opt, state in zip(self.optimizers, checkpoint['optimizer_states']):
                opt.load_state_dict(state)
                
        # 3. Restore Schedulers
        if 'scheduler_states' in checkpoint:
            for sch, state in zip(self.schedulers, checkpoint['scheduler_states']):
                sch.load_state_dict(state)
                
        # 4. Restore Epoch
        if 'epoch' in checkpoint:
            self.start_epoch = checkpoint['epoch']
            self._output(f"Resuming from epoch {self.start_epoch}")
            
        # 5. Restore History
        if 'history' in checkpoint:
            self.history = checkpoint['history']

    def save_plots(self, path: str = ".") -> None:
        """
        Save Loss and Accuracy plots to the output directory.
        :param path: Output directory path.
        """
        self._output(f"Saving plots to {path}")
        epochs = range(1, len(self.history['loss']) + 1)
        
        # Plot Loss
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.history['loss'], label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{path}/loss_curve.png")
        plt.close()
        
        # Plot Accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.history['accuracy'], label='Accuracy', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{path}/accuracy_curve.png")
        plt.close()