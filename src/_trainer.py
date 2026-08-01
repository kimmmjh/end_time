import torch
import os
import shutil
import numpy as np
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.metrics import PairedDecoderMetrics, WandbMetrics, paired_decoder_metrics
from typing import Any, Callable, Mapping, Type
from ._data_generator import (
    DataGenerator,
    PhenomenologicalDataGenerator,
    CapacityDataGenerator,
    CircuitLevelDataGenerator,
)
from panqec.codes import StabilizerCode
import logging
import time
import textwrap
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
        batch_size: int,
        epochs: int,
        batches: int,
        eval_batches: int = 16,
        eval_every: int = 1,
        final_eval_batches: int | None = None,
        hybrid_calibration_batches: int | None = None,
        amp_dtype: str = "fp16",
        lattice_size: int | None = None,
        channels: list[int] | None = None,
        depths: list[int] | None = None,
        architecture: str | None = None,
        recurrent: str | None = None,
        attention: str | None = None,
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
        :param batch_size: Number of samples per batch.
        :param epochs: Number of epochs to train.
        :param batches: Number of batches per epoch.
        :param eval_batches: Number of batches used for evaluation each epoch.
        :param eval_every: Evaluate every N epochs; the final epoch is mandatory.
        :param final_eval_batches: Optional larger final-epoch evaluation.
        :param hybrid_calibration_batches: Fresh batches used to calibrate a
            hybrid correction gate before each evaluation. Defaults to
            ``eval_batches`` and is ignored by models without ``calibrate_gate``.
        :param amp_dtype: Mixed precision dtype: "fp16", "bf16", or "none".
        :param lattice_size: Lattice size used for the code.
        :param channels: Model channel widths per stage.
        :param depths: Model residual-block depths per stage.
        :param architecture: Temporal model family.
        :param recurrent: Recurrent-model setting summary.
        :param attention: Attention setting summary.
        :param verbose: Whether the trainer should print progress or log it.
        :param save_model: If model should be saved.
        :return: The trained decoder and train / validation values.
        """
        self.model = model

        # Setup logging to both console and file
        self._save_directory = save_directory
        if save_directory and not os.path.exists(save_directory):
            os.makedirs(save_directory)

        log_file_path = os.path.join(
            save_directory if save_directory else ".", "training_log.txt"
        )
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],
            force=True,
        )
        import sys

        logging.info(f"Executed Command: python {' '.join(sys.argv)}")
        self._output = logging.info if not verbose else print
        self.criterion = loss_function
        self.optimizers = optimizers
        self.schedulers = schedulers

        self._amp_dtype = amp_dtype
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=torch.cuda.is_available() and self._amp_dtype == "fp16"
        )

        self._num_batches = batches
        self._eval_batches = eval_batches
        self._eval_every = eval_every
        if self._eval_every < 1:
            raise ValueError("eval_every must be positive.")
        self._final_eval_batches = (
            eval_batches if final_eval_batches is None else final_eval_batches
        )
        self._hybrid_calibration_batches = (
            eval_batches
            if hybrid_calibration_batches is None
            else hybrid_calibration_batches
        )
        if self._hybrid_calibration_batches < 0:
            raise ValueError("hybrid_calibration_batches must be non-negative.")
        self._num_epochs = epochs
        self._batch_size = batch_size
        self._save_directory = save_directory
        self._save_model = save_model
        self._plot_metadata = self._format_plot_metadata(
            lattice_size=lattice_size,
            channels=channels,
            depths=depths,
            architecture=architecture,
            recurrent=recurrent,
            attention=attention,
        )

        self.history = {
            "epoch": [],
            "loss": [],
            "accuracy": [],
            "mwpm_accuracy": [],
            "net_gain": [],
            "net_gain_standard_error": [],
            "rescued": [],
            "harmed": [],
            "corrections": [],
        }
        self._last_baseline_accuracy: float | None = None
        self._last_eval_baseline_classes: Tensor | None = None
        self._last_eval_residual_logits: Tensor | None = None
        self._last_calibration_metadata: dict[str, Any] | None = None
        self._best_epoch: int | None = None
        self._best_score = float("-inf")
        self._best_accuracy = float("-inf")
        self._best_checkpoint_metadata: dict[str, Any] | None = None
        self.resume_epochs: list[int] = []
        self.start_epoch = 0
        if load_model_path is not None:
            self.load_model(load_model_path)
            self._num_epochs = self.start_epoch + epochs

    def train(
        self,
        *,
        code: StabilizerCode,
        error_rate: float,
        noise_model: str = "capacity",
        measurement_error_rate: float = 0.0,
        rounds: int | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Train the neural decoder on dynamically generated data.

        :param code: The code to train the decoder on.
        :param error_rate: The error rate to train the decoder on.
        """
        """Extract parameters."""
        torch.backends.cudnn.benchmark = (
            True  # Enable cuda to find the best tuner for hardware.
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_seed = seed
        if seed is not None and self.start_epoch > 0:
            # Stim samplers do not expose serializable RNG state.  Starting a
            # resumed run from the original seed would replay the beginning of
            # the previous training stream, so derive a deterministic disjoint
            # stream from the completed-epoch count.
            data_seed = int(
                np.random.SeedSequence(
                    [seed, self.start_epoch, 0x5EED]
                ).generate_state(1, dtype=np.uint64)[0]
            )
            np.random.seed(data_seed % (2**32))
            self._output(
                "Resume data stream uses a derived seed to avoid replaying "
                f"earlier shots: {data_seed}."
            )
        if noise_model == "capacity":
            data_generator = CapacityDataGenerator(
                code=code,
                verbose=False,
                error_rate=error_rate,
                batch_size=self._batch_size,
                measurement_error_rate=measurement_error_rate,
                seed=data_seed,
            )
        elif noise_model == "phenomenological":
            data_generator = PhenomenologicalDataGenerator(
                code=code,
                verbose=False,
                error_rate=error_rate,
                batch_size=self._batch_size,
                measurement_error_rate=measurement_error_rate,
                rounds=rounds,
                seed=data_seed,
            )
        elif noise_model == "circuit":
            data_generator = CircuitLevelDataGenerator(
                code=code,
                verbose=False,
                error_rate=error_rate,
                batch_size=self._batch_size,
                measurement_error_rate=measurement_error_rate,
                rounds=rounds,
                seed=data_seed,
            )
        else:
            raise ValueError(f"Unknown noise model: {noise_model!r}.")

        hybrid_interface = self._hybrid_interface()
        paired_interface = self._paired_interface()

        """Start Training."""
        for epoch in range(self.start_epoch, self._num_epochs):
            self._output(f"{'=' * 18}")
            self._output(f"Starting Epoch {epoch}.")
            epoch_start = time.time()

            """Train Model"""
            self.model.train()
            loss, _ = self._process_batches(data_generator, device, self._num_batches)
            epoch_time = time.time() - epoch_start

            should_evaluate = (
                (epoch + 1) % self._eval_every == 0
                or epoch == self._num_epochs - 1
            )
            if not should_evaluate:
                self._output(
                    f"[Epoch {epoch}] Loss: {loss:.4f} | "
                    f"Evaluation skipped (every {self._eval_every} epochs) | "
                    f"Time: {epoch_time:.2f}s"
                )
                continue

            """Calibrate the hybrid gate on samples disjoint from evaluation."""
            self._output("Evaluating Model.")
            self.model.eval()
            calibration_metadata = None
            if hybrid_interface is not None:
                if self._hybrid_calibration_batches == 0:
                    calibration_metadata = self._disable_hybrid_gate()
                else:
                    calibration_metadata = self._calibrate_hybrid_gate(
                        data_generator=data_generator,
                        device=device,
                        batches=self._hybrid_calibration_batches,
                    )

            """Evaluate on fresh samples after calibration."""
            # A saved hybrid run gets its large held-out evaluation from the
            # selected best checkpoint below.  All other runs retain the
            # legacy behavior of using ``final_eval_batches`` in the last
            # epoch.
            has_selected_best_evaluation = (
                paired_interface is not None and self._save_model
            )
            eval_batches = (
                self._eval_batches
                if has_selected_best_evaluation or epoch < self._num_epochs - 1
                else self._final_eval_batches
            )
            with torch.no_grad():
                _, (y_pred, y_true) = self._process_batches(
                    data_generator, device, eval_batches, train=False
                )

            """Record evaluation Metrics."""
            metrics = WandbMetrics.get_metrics(
                y_pred=y_pred,
                y_true=y_true,
                loss=loss,
                learning_rate=self.schedulers[0].optimizer.param_groups[0]["lr"],
                epoch_duration=epoch_time,
            )
            paired_metrics = self._paired_metrics(y_pred, y_true)
            log_message = (
                f"[Epoch {epoch}] "
                f"Loss: {metrics.loss:.4f} | "
                f"Accuracy: {metrics.accuracy:.4f} "
                f"(±{metrics.accuracy_std:.4f}) | "
                f"Eval Samples: {eval_batches * self._batch_size} | "
                f"Time: {metrics.epoch_duration:.2f}s"
            )
            if paired_metrics is not None:
                log_message += (
                    f" | MWPM Accuracy: {paired_metrics.baseline_accuracy:.4f}"
                    f" | Rescue: {paired_metrics.rescue_rate:.6f}"
                    f" | Harm: {paired_metrics.harm_rate:.6f}"
                    f" | Correction: {paired_metrics.correction_rate:.6f}"
                    f" | Net Gain: {paired_metrics.net_gain:+.6f}"
                    f" | Paired SE: "
                    f"{paired_metrics.paired_standard_error:.6f}"
                )
            self._output(log_message)
            # wandb.log(metrics.__dict__)

            # Update history and save plots
            self._append_history(epoch, metrics, paired_metrics)

            if self._save_model and paired_interface is not None:
                if paired_metrics is None:
                    raise RuntimeError(
                        "A calibratable hybrid model must expose paired MWPM "
                        "predictions during evaluation."
                    )
                self._consider_best_checkpoint(
                    epoch=epoch,
                    accuracy=float(metrics.accuracy),
                    paired_metrics=paired_metrics,
                    calibration_metadata=calibration_metadata,
                )

            # Format info string for plots dynamically
            q_str = (
                f" | q={measurement_error_rate}" if noise_model != "capacity" else ""
            )
            rounds_str = f" | rounds={data_generator.rounds}"
            info_str = (
                f"Noise: {noise_model} | p={error_rate}{q_str}{rounds_str}"
            )
            if self._plot_metadata:
                info_str = f"{info_str}\n{self._plot_metadata}"

            self.save_plots(path=self._save_directory, info_str=info_str)

        """Save the final resumable state, then evaluate the selected best."""
        if self._save_model:
            self._output("Saving final resumable model.")
            self.save_model(
                path=self._save_directory,
                model_name="model",
                epoch=self._num_epochs - 1,
            )
            if paired_interface is not None and self._best_epoch is not None:
                self._evaluate_selected_best(data_generator, device)

    def _hybrid_interface(self) -> nn.Module | None:
        """Return the model object exposing the optional hybrid gate API."""

        interface = (
            self.model.module
            if isinstance(self.model, nn.DataParallel)
            else self.model
        )
        return interface if callable(getattr(interface, "calibrate_gate", None)) else None

    def _paired_interface(self) -> nn.Module | None:
        """Return a model exposing predictions from a same-shot baseline."""

        interface = self._model_state_target()
        supports_baseline = bool(
            getattr(interface, "supports_paired_baseline", False)
        ) or callable(getattr(interface, "calibrate_gate", None))
        return interface if supports_baseline else None

    def _calibrate_hybrid_gate(
        self,
        *,
        data_generator: DataGenerator,
        device: torch.device,
        batches: int,
    ) -> dict[str, Any]:
        """Calibrate on fresh shots without using them for model selection."""

        interface = self._hybrid_interface()
        if interface is None:
            return {}

        with torch.no_grad():
            _, (_, true_classes) = self._process_batches(
                data_generator,
                device,
                batches,
                train=False,
            )
        if self._last_eval_baseline_classes is None:
            raise RuntimeError(
                "A calibratable hybrid model must expose last_baseline_classes."
            )
        if self._last_eval_residual_logits is None:
            raise RuntimeError(
                "A calibratable hybrid model must expose last_residual_logits."
            )

        with torch.no_grad():
            result = interface.calibrate_gate(
                self._last_eval_residual_logits,
                true_classes,
                self._last_eval_baseline_classes,
            )
        if not isinstance(result, Mapping):
            raise TypeError(
                "calibrate_gate must return a mapping of calibration metadata."
            )
        metadata = self._primitive_mapping(result)
        metadata["samples"] = batches * self._batch_size
        self._last_calibration_metadata = metadata
        summary = ", ".join(
            f"{key}={value}" for key, value in sorted(metadata.items())
        )
        self._output(f"Hybrid Calibration: {summary}")
        return metadata

    def _disable_hybrid_gate(self) -> dict[str, Any]:
        """Explicitly retain pure-MWPM fallback when calibration is disabled."""

        interface = self._hybrid_interface()
        if interface is None:
            return {}
        configure_gate = getattr(interface, "configure_gate", None)
        if callable(configure_gate):
            configure_gate(enabled=False)
        else:
            gate_enabled = getattr(interface, "gate_enabled", None)
            if isinstance(gate_enabled, Tensor):
                with torch.no_grad():
                    gate_enabled.fill_(False)
            elif gate_enabled is not None:
                setattr(interface, "gate_enabled", False)
        metadata = {
            "enabled": False,
            "reason": "hybrid_calibration_batches=0",
            "samples": 0,
        }
        self._last_calibration_metadata = metadata
        self._output("Hybrid Calibration: disabled (0 batches)")
        return metadata

    def _paired_metrics(
        self,
        final_logits: Tensor,
        true_classes: Tensor,
    ) -> PairedDecoderMetrics | None:
        if self._last_eval_baseline_classes is None:
            self._last_baseline_accuracy = None
            return None
        metrics = paired_decoder_metrics(
            final_logits,
            true_classes,
            self._last_eval_baseline_classes,
        )
        self._last_baseline_accuracy = metrics.baseline_accuracy
        return metrics

    def _append_history(
        self,
        epoch: int,
        metrics: WandbMetrics,
        paired_metrics: PairedDecoderMetrics | None,
    ) -> None:
        """Append metrics while upgrading a resumed legacy history lazily."""

        previous_points = len(self.history.setdefault("epoch", []))
        optional_values = {
            "mwpm_accuracy": (
                paired_metrics.baseline_accuracy if paired_metrics else None
            ),
            "net_gain": paired_metrics.net_gain if paired_metrics else None,
            "net_gain_standard_error": (
                paired_metrics.paired_standard_error if paired_metrics else None
            ),
            "rescued": paired_metrics.rescued if paired_metrics else None,
            "harmed": paired_metrics.harmed if paired_metrics else None,
            "corrections": paired_metrics.corrections if paired_metrics else None,
        }
        for key in optional_values:
            if key not in self.history:
                self.history[key] = [None] * previous_points

        self.history["epoch"].append(epoch + 1)
        self.history.setdefault("loss", []).append(float(metrics.loss))
        self.history.setdefault("accuracy", []).append(float(metrics.accuracy))
        for key, value in optional_values.items():
            self.history[key].append(value)

    def _consider_best_checkpoint(
        self,
        *,
        epoch: int,
        accuracy: float,
        paired_metrics: PairedDecoderMetrics | None,
        calibration_metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        """Save a candidate selected by paired gain, or accuracy otherwise."""

        selection_metric = "net_gain" if paired_metrics is not None else "accuracy"
        score = paired_metrics.net_gain if paired_metrics is not None else accuracy
        tolerance = 1e-12
        better = score > self._best_score + tolerance
        tied_but_more_accurate = (
            abs(score - self._best_score) <= tolerance
            and accuracy > self._best_accuracy + tolerance
        )
        if not (better or tied_but_more_accurate):
            return False

        self._best_epoch = epoch
        self._best_score = float(score)
        self._best_accuracy = float(accuracy)
        metadata: dict[str, Any] = {
            "epoch": epoch,
            "completed_epochs": epoch + 1,
            "selection_metric": selection_metric,
            "selection_score": float(score),
            "accuracy": float(accuracy),
            "calibration": self._primitive_mapping(calibration_metadata or {}),
        }
        if paired_metrics is not None:
            metadata["paired_metrics"] = paired_metrics.as_dict()
        self._best_checkpoint_metadata = metadata

        self._output(
            f"Saving new best checkpoint at epoch {epoch}: "
            f"{selection_metric}={score:+.6f}."
        )
        self.save_model(
            path=self._save_directory or ".",
            model_name="best_model",
            epoch=epoch,
            checkpoint_metadata={"checkpoint_role": "best"},
        )
        return True

    def _evaluate_selected_best(
        self,
        data_generator: DataGenerator,
        device: torch.device,
    ) -> None:
        """Evaluate the selected checkpoint once on fresh held-out samples."""

        checkpoint_directory = self._save_directory or "."
        best_path = os.path.join(checkpoint_directory, "best_model.pt")
        final_path = os.path.join(checkpoint_directory, "model.pt")
        if not os.path.isfile(best_path):
            raise FileNotFoundError(f"Best checkpoint not found: {best_path}")

        try:
            self._load_model_weights(best_path)
            self.model.eval()
            with torch.no_grad():
                _, (y_pred, y_true) = self._process_batches(
                    data_generator,
                    device,
                    self._final_eval_batches,
                    train=False,
                )
            accuracy = float(
                (y_pred.argmax(dim=1) == y_true).float().mean().item()
            )
            paired_metrics = self._paired_metrics(y_pred, y_true)
            selected: dict[str, Any] = {
                "epoch": self._best_epoch,
                "eval_samples": self._final_eval_batches * self._batch_size,
                "accuracy": accuracy,
            }
            if paired_metrics is not None:
                selected["paired_metrics"] = paired_metrics.as_dict()
                lower_95 = (
                    paired_metrics.net_gain
                    - 1.96 * paired_metrics.paired_standard_error
                )
                selected["net_gain_lower_95"] = lower_95
                interface = self._paired_interface()
                recommendation_name = getattr(
                    interface, "recommendation_name", "hybrid"
                )
                selected["recommended_decoder"] = (
                    recommendation_name if lower_95 > 0.0 else "mwpm"
                )
                log_line = (
                    f"[Selected Best] Epoch: {self._best_epoch}"
                    f" | Accuracy: {accuracy:.6f}"
                    f" | MWPM Accuracy: {paired_metrics.baseline_accuracy:.6f}"
                    f" | Rescue: {paired_metrics.rescue_rate:.6f}"
                    f" | Harm: {paired_metrics.harm_rate:.6f}"
                    f" | Correction: {paired_metrics.correction_rate:.6f}"
                    f" | Net Gain: {paired_metrics.net_gain:+.6f}"
                    f" | Paired SE: {paired_metrics.paired_standard_error:.6f}"
                    f" | Eval Samples: {paired_metrics.num_samples}"
                    f" | Recommended: {selected['recommended_decoder']}"
                )
            else:
                selected["recommended_decoder"] = "model"
                log_line = (
                    f"[Selected Best] Epoch: {self._best_epoch}"
                    f" | Accuracy: {accuracy:.6f}"
                    f" | Eval Samples: {self._final_eval_batches * self._batch_size}"
                    " | Recommended: model"
                )
            self._output(log_line)
            self._update_checkpoint_metadata(
                best_path, "selected_best_evaluation", selected
            )
            self._update_checkpoint_metadata(
                final_path, "selected_best_evaluation", selected
            )
        finally:
            self._load_model_weights(final_path)

    @staticmethod
    def _primitive_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
        """Convert calibration values to objects safe for torch checkpoints."""

        def convert(value: Any) -> Any:
            if isinstance(value, Tensor):
                value = value.detach().cpu()
                return value.item() if value.numel() == 1 else value.tolist()
            if isinstance(value, Mapping):
                return {str(key): convert(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(item) for item in value]
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            return str(value)

        return {str(key): convert(value) for key, value in values.items()}

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

        loss = 0.0
        finite_batches = 0
        skipped_batches = 0
        iterator = range(batches)
        if train:
            iterator = tqdm(iterator, desc="Training", mininterval=10.0)

        all_y_pred = []
        all_y = []
        all_baseline_classes = []
        all_residual_logits = []
        metric_source = self._paired_interface() or self.model
        loss_source = self._model_state_target()
        requires_batch_metadata = bool(
            getattr(loss_source, "requires_batch_metadata", False)
        )

        if not train:
            self._last_eval_baseline_classes = None
            self._last_eval_residual_logits = None

        for _ in iterator:
            batch_metadata = None
            if train and requires_batch_metadata:
                metadata_generator = getattr(
                    data_generator, "generate_batch_with_metadata", None
                )
                if not callable(metadata_generator):
                    raise TypeError(
                        "This model requires batch metadata, but the data "
                        "generator does not provide generate_batch_with_metadata()."
                    )
                X, y, batch_metadata = metadata_generator(device=device)
            else:
                X, y = data_generator.generate_batch(device=device)
            """Zero out the gradient for all optimizers."""
            if train:
                for optimizer in self.optimizers:
                    optimizer.zero_grad()

            """Forward pass."""
            amp_enabled = device.type == "cuda" and self._amp_dtype != "none"
            amp_dtype = torch.bfloat16 if self._amp_dtype == "bf16" else torch.float16
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                y_pred = self.model(X)
                loss_c = None
                if train:
                    loss_input, loss_target = y_pred, y
                    loss_adapter = getattr(loss_source, "loss_inputs", None)
                    if callable(loss_adapter):
                        # Hybrid matching models expose residual logits and labels
                        # for the loss while returning final logical-class logits
                        # for metrics and deployment.
                        if batch_metadata is None:
                            loss_input, loss_target = loss_adapter(y_pred, y)
                        else:
                            loss_input, loss_target = loss_adapter(
                                y_pred,
                                y,
                                batch_metadata=batch_metadata,
                            )
                    loss_c = self.criterion(loss_input, loss_target)

            if train and loss_c is not None and not torch.isfinite(loss_c):
                skipped_batches += 1
                for optimizer in self.optimizers:
                    optimizer.zero_grad(set_to_none=True)
                for scheduler in self.schedulers:
                    scheduler.step()
                continue

            if not train:
                all_y_pred.append(y_pred)
                all_y.append(y)
                baseline_classes = getattr(
                    metric_source, "last_baseline_classes", None
                )
                if baseline_classes is not None:
                    all_baseline_classes.append(
                        baseline_classes.detach().to(device=y.device)
                    )
                residual_logits = getattr(
                    metric_source, "last_residual_logits", None
                )
                if residual_logits is not None:
                    all_residual_logits.append(
                        residual_logits.detach().to(device=y.device)
                    )

            if train:
                assert loss_c is not None
                """Record loss."""
                loss += loss_c.item()
                finite_batches += 1

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

        if not train:
            if all_baseline_classes:
                baseline = torch.cat(all_baseline_classes)
                truth = torch.cat(all_y)
                self._last_eval_baseline_classes = baseline
                self._last_baseline_accuracy = float(
                    (baseline == truth).float().mean().item()
                )
            else:
                self._last_baseline_accuracy = None
            if all_residual_logits:
                self._last_eval_residual_logits = torch.cat(all_residual_logits)
            return loss / batches, (torch.cat(all_y_pred), torch.cat(all_y))

        if skipped_batches:
            self._output(f"Skipped {skipped_batches} non-finite training batches.")

        if finite_batches == 0:
            raise FloatingPointError("All training batches produced non-finite loss.")

        return loss / finite_batches, (y_pred, y)

    @staticmethod
    def _format_plot_metadata(
        *,
        lattice_size: int | None,
        channels: list[int] | None,
        depths: list[int] | None,
        architecture: str | None,
        recurrent: str | None,
        attention: str | None,
    ) -> str:
        parts = []
        if lattice_size is not None:
            parts.append(f"L={lattice_size}")
        if channels is not None:
            parts.append(f"channels={channels}")
        if depths is not None:
            parts.append(f"depths={depths}")
        if architecture is not None:
            parts.append(f"architecture={architecture}")
        if recurrent is not None:
            parts.append(recurrent)
        if attention is not None:
            parts.append(f"attention={attention}")
        return " | ".join(parts)

    def save_model(
        self,
        path: str = ".",
        model_name: str = "model",
        epoch: int = 0,
        checkpoint_metadata: Mapping[str, Any] | None = None,
    ) -> None:
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
        model_state = (
            self.model.module.state_dict()
            if isinstance(self.model, nn.DataParallel)
            else self.model.state_dict()
        )

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model_state,
            "optimizer_states": optim_states,
            "scheduler_states": sched_states,
            "scaler_state_dict": self.scaler.state_dict(),
            "history": self.history,
            "resume_epochs": self.resume_epochs,
            "best_checkpoint": self._best_checkpoint_metadata,
        }
        if checkpoint_metadata:
            checkpoint.update(self._primitive_mapping(checkpoint_metadata))

        destination = os.path.join(path or ".", f"{model_name}.pt")
        temporary = f"{destination}.tmp"
        torch.save(checkpoint, temporary)
        os.replace(temporary, destination)

    def _model_state_target(self) -> nn.Module:
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _restore_model_state(self, state_dict: Mapping[str, Tensor]) -> None:
        """Restore weights while tolerating a legacy DataParallel prefix."""

        target = self._model_state_target()
        try:
            target.load_state_dict(state_dict)
        except RuntimeError:
            normalized = {
                key.replace("module.", "", 1) if key.startswith("module.") else key: value
                for key, value in state_dict.items()
            }
            target.load_state_dict(normalized)

    def _load_model_weights(self, path: str) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=device)
        self._restore_model_state(checkpoint["model_state_dict"])

    @staticmethod
    def _update_checkpoint_metadata(path: str, key: str, value: Any) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        checkpoint[key] = value
        temporary = f"{path}.tmp"
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)

    def load_model(self, path: str) -> None:
        """
        Load a checkpoint and restore training state.
        :param path: Path to the .pt checkpoint file
        """
        self._output(f"Loading checkpoint from {path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=device)

        # Restore model weights.
        self._restore_model_state(checkpoint["model_state_dict"])

        # Restore Adam/optimizer moments while retaining the freshly configured
        # hyperparameters (especially the LR values installed by the new
        # OneCycleLR cycle).
        optimizer_states = checkpoint.get("optimizer_states")
        if optimizer_states is not None:
            if len(optimizer_states) != len(self.optimizers):
                raise ValueError(
                    "Checkpoint optimizer count does not match the current run: "
                    f"{len(optimizer_states)} != {len(self.optimizers)}."
                )
            for optimizer, state in zip(self.optimizers, optimizer_states):
                fresh_settings = [
                    {key: value for key, value in group.items() if key != "params"}
                    for group in optimizer.param_groups
                ]
                optimizer.load_state_dict(state)
                if len(fresh_settings) != len(optimizer.param_groups):
                    raise ValueError(
                        "Checkpoint optimizer parameter-group count does not "
                        "match the current run."
                    )
                for group, settings in zip(optimizer.param_groups, fresh_settings):
                    group.update(settings)
            self._output("Restored optimizer state from checkpoint.")
        else:
            self._output("Checkpoint has no optimizer state; using a fresh optimizer.")

        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler_state:
            self.scaler.load_state_dict(scaler_state)

        checkpoint_history = checkpoint.get("history") or {}
        losses = list(checkpoint_history.get("loss") or [])
        accuracies = list(checkpoint_history.get("accuracy") or [])
        if len(losses) != len(accuracies):
            raise ValueError(
                "Checkpoint history is inconsistent: loss and accuracy lengths differ."
            )
        history_epochs = list(checkpoint_history.get("epoch") or [])
        if not history_epochs:
            history_epochs = list(range(1, len(losses) + 1))
        if len(history_epochs) != len(losses):
            raise ValueError(
                "Checkpoint history is inconsistent: epoch and metric lengths differ."
            )
        self.history = {
            key: list(values)
            for key, values in checkpoint_history.items()
            if isinstance(values, (list, tuple))
        }
        self.history["epoch"] = history_epochs
        self.history["loss"] = losses
        self.history["accuracy"] = accuracies

        # Preserve an earlier best checkpoint across a resumed phase when the
        # checkpoint lives beside it. Legacy checkpoints simply start a fresh
        # best-selection phase.
        best_metadata = checkpoint.get("best_checkpoint")
        source_best = os.path.join(os.path.dirname(path), "best_model.pt")
        if isinstance(best_metadata, Mapping) and os.path.isfile(source_best):
            self._best_checkpoint_metadata = self._primitive_mapping(best_metadata)
            self._best_epoch = int(best_metadata["epoch"])
            self._best_score = float(best_metadata["selection_score"])
            self._best_accuracy = float(best_metadata["accuracy"])
            destination_best = os.path.join(
                self._save_directory or ".", "best_model.pt"
            )
            if os.path.abspath(source_best) != os.path.abspath(destination_best):
                shutil.copy2(source_best, destination_best)

        self.start_epoch = int(checkpoint.get("epoch", len(losses)))
        if self.start_epoch < 0:
            raise ValueError(f"Invalid checkpoint epoch: {self.start_epoch}.")
        self.resume_epochs = [
            int(epoch) for epoch in checkpoint.get("resume_epochs", [])
        ]
        if self.start_epoch > 0 and self.start_epoch not in self.resume_epochs:
            self.resume_epochs.append(self.start_epoch)

        self._output(
            f"Resuming after epoch {self.start_epoch} with "
            f"{len(losses)} historical plot points."
        )
        if checkpoint.get("scheduler_states"):
            self._output(
                "Starting a new learning-rate schedule for the additional epochs; "
                "the completed OneCycleLR state is not reused."
            )

    def save_plots(self, path: str = ".", info_str: str = "") -> None:
        """
        Save Loss and Accuracy plots to the output directory.
        :param path: Output directory path.
        :param info_str: Additional context for the plot titles.
        """
        self._output(f"Saving plots to {path}")
        epochs = self.history.get("epoch") or list(
            range(1, len(self.history["loss"]) + 1)
        )

        wrapped_info = "\n".join(
            wrapped_line
            for line in info_str.splitlines()
            for wrapped_line in textwrap.wrap(
                line,
                width=100,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
        title_suffix = f"\n({wrapped_info})" if wrapped_info else ""

        # Plot Loss
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, self.history["loss"], label="Loss")
        self._plot_resume_markers()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training Loss{title_suffix}", fontsize=11)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{path}/loss_curve.png")
        plt.close()

        # Plot Accuracy
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, self.history["accuracy"], label="Accuracy", color="orange")
        mwpm_history = self.history.get("mwpm_accuracy", [])
        if len(mwpm_history) == len(epochs):
            mwpm_points = [
                (epoch, value)
                for epoch, value in zip(epochs, mwpm_history)
                if value is not None
            ]
            if mwpm_points:
                plt.plot(
                    [point[0] for point in mwpm_points],
                    [point[1] for point in mwpm_points],
                    label="MWPM Accuracy",
                    color="steelblue",
                    linestyle="--",
                )
        self._plot_resume_markers()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Training Accuracy{title_suffix}", fontsize=11)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{path}/accuracy_curve.png")
        plt.close()

        # Plot the paired improvement separately so a tiny hybrid degradation
        # cannot be hidden by two nearly overlapping accuracy curves.
        net_gain_history = self.history.get("net_gain", [])
        standard_errors = self.history.get("net_gain_standard_error", [])
        if (
            len(net_gain_history) == len(epochs)
            and len(standard_errors) == len(epochs)
        ):
            gain_points = [
                (epoch, gain, standard_error)
                for epoch, gain, standard_error in zip(
                    epochs, net_gain_history, standard_errors
                )
                if gain is not None and standard_error is not None
            ]
            if gain_points:
                plt.figure(figsize=(12, 6))
                plt.errorbar(
                    [point[0] for point in gain_points],
                    [point[1] for point in gain_points],
                    yerr=[1.96 * point[2] for point in gain_points],
                    label="Hybrid - MWPM (95% paired interval)",
                    color="purple",
                    marker="o",
                    markersize=3,
                    capsize=2,
                )
                plt.axhline(0.0, color="black", linestyle=":", linewidth=1)
                self._plot_resume_markers()
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy difference")
                plt.title(f"Hybrid Net Gain over MWPM{title_suffix}", fontsize=11)
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"{path}/hybrid_net_gain_curve.png")
                plt.close()

    def _plot_resume_markers(self) -> None:
        """Mark boundaries between checkpointed and newly added epochs."""
        for index, completed_epochs in enumerate(self.resume_epochs):
            plt.axvline(
                completed_epochs + 0.5,
                color="gray",
                linestyle="--",
                alpha=0.65,
                label="Resume" if index == 0 else None,
            )
