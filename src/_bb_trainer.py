"""Dedicated training loop for equivariant BB neural belief propagation.

The toric-code trainer in :mod:`src._trainer` assumes a 16-way logical class
head and lattice-shaped CNN inputs.  BB neural BP instead predicts one Pauli
belief per qubit and must be scored modulo stabilizers, so keeping this path
separate avoids silently reporting the wrong notion of accuracy.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch
from torch import Tensor, nn

from ._bb_loss import DegeneracyAwareBPLoss
from ._bb_metrics import (
    BBAggregateMetrics,
    BBShotOutcomes,
    aggregate_bb_outcomes,
    bb_shot_outcomes,
    paired_success_gain,
)
from .bb_code import BBCodeSpec
from .bb_data_generator import BBCodeCapacityBatch, BBCodeCapacityGenerator


class BBNeuralBPTrainer:
    """Train and evaluate an equivariant BP4 decoder on fresh capacity data."""

    def __init__(
        self,
        *,
        model: nn.Module,
        code: BBCodeSpec,
        train_generator: BBCodeCapacityGenerator,
        eval_generator: BBCodeCapacityGenerator,
        criterion: DegeneracyAwareBPLoss,
        device: torch.device,
        epochs: int,
        batches: int,
        batch_size: int,
        eval_batches: int,
        eval_every: int,
        final_eval_batches: int,
        learning_rate: float,
        weight_decay: float,
        gradient_clip: float,
        output_directory: str | os.PathLike[str],
        experiment_config: dict[str, Any],
        save_model: bool = False,
        load_model_path: str | os.PathLike[str] | None = None,
    ) -> None:
        for name, value in (
            ("epochs", epochs),
            ("batches", batches),
            ("batch_size", batch_size),
            ("eval_batches", eval_batches),
            ("eval_every", eval_every),
            ("final_eval_batches", final_eval_batches),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}.")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative.")
        if gradient_clip <= 0.0:
            raise ValueError("gradient_clip must be positive.")

        self.model = model
        self.code = code
        self.train_generator = train_generator
        self.eval_generator = eval_generator
        self.criterion = criterion
        self.device = device
        self.epochs = int(epochs)
        self.batches = int(batches)
        self.batch_size = int(batch_size)
        self.eval_batches = int(eval_batches)
        self.eval_every = int(eval_every)
        self.final_eval_batches = int(final_eval_batches)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.gradient_clip = float(gradient_clip)
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.experiment_config = copy.deepcopy(experiment_config)
        self.save_model = bool(save_model)

        self._configure_logging()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=weight_decay,
        )
        self.start_epoch = 0
        self.history: dict[str, list[float | int]] = {
            "train_epoch": [],
            "train_loss": [],
            "train_syndrome_loss": [],
            "train_logical_loss": [],
            "train_pauli_loss": [],
            "eval_epoch": [],
            "neural_logical_accuracy": [],
            "vanilla_logical_accuracy": [],
            "neural_syndrome_convergence": [],
            "vanilla_syndrome_convergence": [],
            "paired_gain": [],
            "paired_standard_error": [],
        }
        self.best_epoch: int | None = None
        self.best_accuracy = float("-inf")
        self._best_state: dict[str, Tensor] | None = None

        if load_model_path is not None:
            self._load_checkpoint(Path(load_model_path))
            # A resumed phase is a new OneCycle schedule at the requested LR.
            for group in self.optimizer.param_groups:
                group["lr"] = self.learning_rate
                group["initial_lr"] = self.learning_rate
                group["weight_decay"] = self.weight_decay

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            epochs=self.epochs,
            steps_per_epoch=self.batches,
        )

    def _configure_logging(self) -> None:
        log_path = self.output_directory / "training_log.txt"
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
            force=True,
        )
        logging.info("Executed Command: python %s", " ".join(sys.argv))

    @staticmethod
    def _generator_state(generator: BBCodeCapacityGenerator) -> Any:
        if hasattr(generator, "state_dict"):
            return generator.state_dict()
        return copy.deepcopy(generator._rng.bit_generator.state)  # noqa: SLF001

    @staticmethod
    def _restore_generator_state(
        generator: BBCodeCapacityGenerator, state: Any
    ) -> None:
        if hasattr(generator, "load_state_dict"):
            generator.load_state_dict(state)
        else:
            generator._rng.bit_generator.state = state  # noqa: SLF001

    def _validate_checkpoint_config(self, saved: dict[str, Any]) -> None:
        required = (
            "architecture",
            "code",
            "graph_fingerprint",
            "bp_iterations",
            "bp_residual_hidden_dim",
            "bp_parameter_sharing",
            "bp_residual_scale",
            "bp_max_relaxation_delta",
            "bp_deep_supervision_weight",
            "bb_syndrome_loss_weight",
            "bb_logical_loss_weight",
            "bb_pauli_loss_weight",
            "bb_weight_decay",
            "channel",
            "error_rate",
            "x_error_rate",
            "z_error_rate",
        )
        mismatches = []
        for key in required:
            if saved.get(key) != self.experiment_config.get(key):
                mismatches.append(
                    f"{key}: checkpoint={saved.get(key)!r}, "
                    f"current={self.experiment_config.get(key)!r}"
                )
        if mismatches:
            raise ValueError(
                "Checkpoint is incompatible with this BB graph/model:\n  "
                + "\n  ".join(mismatches)
            )

    def _load_checkpoint(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"BB checkpoint not found: {path}")
        try:
            # PyTorch 2.6 changed the default to weights_only=True.  This is a
            # trusted local training checkpoint and also contains optimizer,
            # history, and NumPy RNG state.
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:  # PyTorch versions predating the weights_only flag.
            checkpoint = torch.load(path, map_location=self.device)
        saved_config = checkpoint.get("experiment_config")
        if not isinstance(saved_config, dict):
            raise ValueError(
                "Checkpoint has no BB experiment_config and cannot be safely resumed."
            )
        self._validate_checkpoint_config(saved_config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = int(checkpoint.get("epoch", -1)) + 1
        loaded_history = checkpoint.get("history")
        if isinstance(loaded_history, dict):
            for key in self.history:
                if key in loaded_history:
                    self.history[key] = list(loaded_history[key])
        self.best_epoch = checkpoint.get("best_epoch")
        self.best_accuracy = float(checkpoint.get("best_accuracy", float("-inf")))
        saved_best_state = checkpoint.get("best_model_state_dict")
        if isinstance(saved_best_state, dict):
            self._best_state = {
                key: value.detach().to(self.device).clone()
                for key, value in saved_best_state.items()
            }
        generator_states = checkpoint.get("generator_states", {})
        if "train" in generator_states:
            self._restore_generator_state(
                self.train_generator, generator_states["train"]
            )
        if "eval" in generator_states:
            self._restore_generator_state(self.eval_generator, generator_states["eval"])
        logging.info(
            "Resumed BB checkpoint %s after epoch %d; running %d additional epochs.",
            path,
            self.start_epoch - 1,
            self.epochs,
        )

    def _checkpoint(self, epoch: int) -> dict[str, Any]:
        return {
            "format_version": 1,
            "epoch": int(epoch),
            "experiment_config": copy.deepcopy(self.experiment_config),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": copy.deepcopy(self.history),
            "best_epoch": self.best_epoch,
            "best_accuracy": self.best_accuracy,
            "best_model_state_dict": self._best_state,
            "generator_states": {
                "train": self._generator_state(self.train_generator),
                "eval": self._generator_state(self.eval_generator),
            },
        }

    def _atomic_torch_save(self, value: Any, path: Path) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(value, temporary)
        os.replace(temporary, path)

    def _save_latest(self, epoch: int) -> None:
        if self.save_model:
            self._atomic_torch_save(
                self._checkpoint(epoch), self.output_directory / "model.pt"
            )

    def _consider_best(self, epoch: int, metrics: BBAggregateMetrics) -> None:
        if metrics.logical_accuracy <= self.best_accuracy:
            return
        self.best_epoch = int(epoch)
        self.best_accuracy = metrics.logical_accuracy
        self._best_state = {
            key: value.detach().clone()
            for key, value in self.model.state_dict().items()
        }
        if self.save_model:
            self._atomic_torch_save(
                self._checkpoint(epoch), self.output_directory / "best_model.pt"
            )

    def _model_prior(self, batch: BBCodeCapacityBatch) -> dict[str, Any]:
        if self.train_generator.noise_model == "depolarizing":
            return {"p": self.train_generator.error_rate}
        log_prior = batch.channel_probabilities.clamp_min(1e-12).log()
        return {"prior_logits": log_prior.unsqueeze(0).expand(self.code.n, -1)}

    def _outcomes(self, logits: Tensor, batch: BBCodeCapacityBatch) -> BBShotOutcomes:
        return bb_shot_outcomes(
            logits,
            batch.syndrome,
            batch.pauli,
            hx=self.criterion.hx,
            hz=self.criterion.hz,
            logicals_x=self.criterion.logicals_x,
            logicals_z=self.criterion.logicals_z,
        )

    def _train_epoch(self) -> dict[str, float]:
        self.model.train()
        sums = {"total": 0.0, "syndrome": 0.0, "logical": 0.0, "pauli": 0.0}
        for _ in range(self.batches):
            batch = self.train_generator.sample(self.batch_size, device=self.device)
            self.optimizer.zero_grad(set_to_none=True)
            final, iteration_logits = self.model(
                batch.syndrome,
                **self._model_prior(batch),
                neural=True,
                return_all=True,
            )
            losses = self.criterion(
                final,
                batch.syndrome,
                batch.pauli,
                iteration_logits=iteration_logits,
            )
            if not torch.isfinite(losses.total):
                raise FloatingPointError(
                    "Non-finite BB loss. Use float32 and reduce the learning rate."
                )
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()
            self.scheduler.step()
            sums["total"] += float(losses.total.detach())
            sums["syndrome"] += float(losses.syndrome.detach())
            sums["logical"] += float(losses.logical.detach())
            sums["pauli"] += float(losses.pauli.detach())
        return {key: value / self.batches for key, value in sums.items()}

    @torch.no_grad()
    def evaluate(
        self, batches: int
    ) -> tuple[BBAggregateMetrics, BBAggregateMetrics, tuple[float, float, int, int]]:
        self.model.eval()
        neural_outcomes: list[BBShotOutcomes] = []
        vanilla_outcomes: list[BBShotOutcomes] = []
        for _ in range(batches):
            batch = self.eval_generator.sample(self.batch_size, device=self.device)
            prior = self._model_prior(batch)
            neural_logits = self.model(batch.syndrome, **prior, neural=True)
            vanilla_logits = self.model(batch.syndrome, **prior, neural=False)
            neural_outcomes.append(self._outcomes(neural_logits, batch))
            vanilla_outcomes.append(self._outcomes(vanilla_logits, batch))
        return (
            aggregate_bb_outcomes(neural_outcomes),
            aggregate_bb_outcomes(vanilla_outcomes),
            paired_success_gain(neural_outcomes, vanilla_outcomes),
        )

    def _record_training(self, epoch: int, losses: dict[str, float]) -> None:
        self.history["train_epoch"].append(epoch)
        self.history["train_loss"].append(losses["total"])
        self.history["train_syndrome_loss"].append(losses["syndrome"])
        self.history["train_logical_loss"].append(losses["logical"])
        self.history["train_pauli_loss"].append(losses["pauli"])

    def _record_evaluation(
        self,
        epoch: int,
        neural: BBAggregateMetrics,
        vanilla: BBAggregateMetrics,
        paired: tuple[float, float, int, int],
    ) -> None:
        gain, standard_error, _, _ = paired
        self.history["eval_epoch"].append(epoch)
        self.history["neural_logical_accuracy"].append(neural.logical_accuracy)
        self.history["vanilla_logical_accuracy"].append(vanilla.logical_accuracy)
        self.history["neural_syndrome_convergence"].append(neural.syndrome_convergence)
        self.history["vanilla_syndrome_convergence"].append(
            vanilla.syndrome_convergence
        )
        self.history["paired_gain"].append(gain)
        self.history["paired_standard_error"].append(standard_error)

    @staticmethod
    def _metric_line(
        prefix: str,
        epoch: int,
        neural: BBAggregateMetrics,
        vanilla: BBAggregateMetrics,
        paired: tuple[float, float, int, int],
    ) -> str:
        gain, gain_se, rescues, harms = paired
        return (
            f"[{prefix}] Epoch: {epoch} | Accuracy: {neural.logical_accuracy:.8f} "
            f"| Logical Error Rate: {neural.logical_error_rate:.8f} "
            f"| Syndrome Convergence: {neural.syndrome_convergence:.8f} "
            f"| Flagged: {neural.flagged_failure_rate:.8f} "
            f"| Unflagged Logical: {neural.unflagged_logical_failure_rate:.8f} "
            f"| Vanilla BP Accuracy: {vanilla.logical_accuracy:.8f} "
            f"| Paired Gain: {gain:+.8f} +/- {1.96 * gain_se:.8f} "
            f"| Rescued: {rescues} | Harmed: {harms} "
            f"| Eval Samples: {neural.samples}"
        )

    def _write_history(self) -> None:
        path = self.output_directory / "history.json"
        temporary = path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self.history, handle, indent=2)
        os.replace(temporary, path)

    def _plot_history(self) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logging.warning("matplotlib is unavailable; skipping BB plots.")
            return

        if self.history["train_epoch"]:
            figure, axis = plt.subplots(figsize=(8, 5))
            axis.plot(
                self.history["train_epoch"],
                self.history["train_loss"],
                label="total",
            )
            axis.plot(
                self.history["train_epoch"],
                self.history["train_syndrome_loss"],
                label="syndrome",
                alpha=0.8,
            )
            axis.plot(
                self.history["train_epoch"],
                self.history["train_logical_loss"],
                label="logical coset",
                alpha=0.8,
            )
            axis.set_xlabel("epoch")
            axis.set_ylabel("loss")
            axis.set_title(f"{self.code.name} equivariant neural BP training")
            axis.grid(alpha=0.25)
            axis.legend()
            figure.tight_layout()
            figure.savefig(self.output_directory / "loss_curve.png", dpi=180)
            plt.close(figure)

        if self.history["eval_epoch"]:
            figure, axis = plt.subplots(figsize=(8, 5))
            axis.plot(
                self.history["eval_epoch"],
                self.history["neural_logical_accuracy"],
                marker="o",
                label="equivariant neural BP4",
            )
            axis.plot(
                self.history["eval_epoch"],
                self.history["vanilla_logical_accuracy"],
                marker="s",
                label="vanilla BP4 (same shots)",
            )
            axis.set_xlabel("epoch")
            axis.set_ylabel("block logical accuracy")
            axis.set_ylim(0.0, 1.0)
            axis.set_title(f"{self.code.name} code-capacity decoding")
            axis.grid(alpha=0.25)
            axis.legend()
            figure.tight_layout()
            figure.savefig(
                self.output_directory / "logical_accuracy_curve.png", dpi=180
            )
            plt.close(figure)

    def train(self) -> None:
        """Run training, same-shot BP4 comparisons, and final best evaluation."""

        logging.info(
            "BB code: %s [[%d,%d,%d]] | checks: %d+%d | edges: %d",
            self.code.name,
            self.code.n,
            self.code.k,
            self.code.d,
            self.code.num_x_checks,
            self.code.num_z_checks,
            self.code.num_edges,
        )
        logging.info(
            "Samples - training per epoch: %d | validation: %d | final: %d",
            self.batch_size * self.batches,
            self.batch_size * self.eval_batches,
            self.batch_size * self.final_eval_batches,
        )
        final_epoch = self.start_epoch + self.epochs - 1
        for epoch in range(self.start_epoch, final_epoch + 1):
            started = time.time()
            losses = self._train_epoch()
            self._record_training(epoch, losses)
            logging.info(
                "[Train] Epoch: %d | Loss: %.8f | Syndrome: %.8f | "
                "Logical Surrogate: %.8f | Pauli Aux: %.8f | LR: %.6g | Time: %.1fs",
                epoch,
                losses["total"],
                losses["syndrome"],
                losses["logical"],
                losses["pauli"],
                self.scheduler.get_last_lr()[0],
                time.time() - started,
            )

            should_evaluate = (
                epoch - self.start_epoch + 1
            ) % self.eval_every == 0 or epoch == final_epoch
            if should_evaluate:
                neural, vanilla, paired = self.evaluate(self.eval_batches)
                self._record_evaluation(epoch, neural, vanilla, paired)
                logging.info(
                    self._metric_line("Epoch Eval", epoch, neural, vanilla, paired)
                )
                self._consider_best(epoch, neural)
                self._plot_history()
            self._write_history()
            self._save_latest(epoch)

        if self._best_state is None or self.best_epoch is None:
            raise RuntimeError("Training finished without an evaluated checkpoint.")

        final_state = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self._best_state)
        selected = self.evaluate(self.final_eval_batches)
        logging.info(
            self._metric_line(
                "Selected Best",
                self.best_epoch,
                selected[0],
                selected[1],
                selected[2],
            )
        )
        self.model.load_state_dict(final_state)
        if self.save_model:
            # best_model.pt was written at the actual selected epoch; keep its
            # epoch/optimizer/generator state internally consistent.  model.pt
            # remains the latest trainable checkpoint.
            self._save_latest(final_epoch)


__all__ = ["BBNeuralBPTrainer"]
