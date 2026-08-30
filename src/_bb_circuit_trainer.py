"""Trainer for the circuit-level BB neural belief-propagation decoder.

The training and evaluation streams are deliberately different views of the
same circuit, matching the toric circuit path.  Training samples the detector
error model with ``return_errors=True`` so the objective can see which fault
mechanisms fired.  Validation and the final evaluation sample the compiled
circuit detector sampler instead, so reported accuracy is measured on fresh
exact circuit shots rather than on latent DEM labels.

Every evaluation is paired.  The same shots are decoded by the neural model and
by vanilla normalised min-sum, and -- when OSD scoring is enabled -- by the same
ordered-statistics post-processor driven from each posterior in turn.  Because
the neural residual is zero-initialised, an untrained model reproduces the
vanilla decoder bitwise, so the paired gain starts at exactly zero.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from ._bb_circuit_loss import CircuitDegeneracyAwareLoss
from ._bb_circuit_metrics import (
    BBCircuitOutcomes,
    OsdPostprocessor,
    score_corrections,
)
from .bb_circuit_data import BBCircuitGenerator
from models._equivariant_neural_bp2 import EquivariantNeuralBP2

Z_95 = 1.959963984540054


@dataclass
class CircuitEvaluation:
    """One paired evaluation of the neural and vanilla decoders."""

    shots: int
    neural_accuracy: float
    vanilla_accuracy: float
    neural_converged: float
    vanilla_converged: float
    neural_flagged: float
    neural_unflagged: float
    paired_gain: float
    paired_gain_error: float
    rescued: int
    harmed: int
    osd_shots: int = 0
    neural_osd_accuracy: float | None = None
    vanilla_osd_accuracy: float | None = None
    osd_paired_gain: float | None = None
    osd_paired_gain_error: float | None = None


def _paired_gain(
    neural: np.ndarray, vanilla: np.ndarray
) -> tuple[float, float, int, int]:
    """Mean and 95% half-width of the per-shot accuracy difference."""

    difference = neural.astype(np.float64) - vanilla.astype(np.float64)
    shots = difference.size
    mean = float(difference.mean()) if shots else float("nan")
    if shots > 1:
        error = Z_95 * float(difference.std(ddof=1)) / math.sqrt(shots)
    else:
        error = float("nan")
    rescued = int(np.count_nonzero(neural & ~vanilla))
    harmed = int(np.count_nonzero(~neural & vanilla))
    return mean, error, rescued, harmed


class BBCircuitTrainer:
    """Train and evaluate an :class:`EquivariantNeuralBP2` decoder."""

    def __init__(
        self,
        *,
        model: EquivariantNeuralBP2,
        generator: BBCircuitGenerator,
        eval_generator: BBCircuitGenerator,
        criterion: CircuitDegeneracyAwareLoss,
        device: torch.device,
        epochs: int,
        batches: int,
        eval_batches: int,
        eval_every: int,
        final_eval_batches: int,
        learning_rate: float,
        weight_decay: float,
        gradient_clip: float,
        output_directory: str,
        experiment_config: dict[str, Any],
        save_model: bool = False,
        osd_eval_shots: int = 0,
        osd_method: str = "OSD_CS",
        osd_order: int = 7,
        load_model_path: str | os.PathLike[str] | None = None,
    ) -> None:
        for name, value in (
            ("epochs", epochs),
            ("batches", batches),
            ("eval_batches", eval_batches),
            ("eval_every", eval_every),
            ("final_eval_batches", final_eval_batches),
        ):
            if int(value) < 1:
                raise ValueError(f"{name} must be positive, got {value}.")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative.")
        if gradient_clip <= 0.0:
            raise ValueError("gradient_clip must be positive.")
        if osd_eval_shots < 0:
            raise ValueError("osd_eval_shots must be non-negative.")

        self.model = model
        self.generator = generator
        self.eval_generator = eval_generator
        self.criterion = criterion
        self.device = device
        self.epochs = int(epochs)
        self.batches = int(batches)
        self.eval_batches = int(eval_batches)
        self.eval_every = int(eval_every)
        self.final_eval_batches = int(final_eval_batches)
        self.gradient_clip = float(gradient_clip)
        self.output_directory = Path(output_directory)
        self.experiment_config = copy.deepcopy(experiment_config)
        self.save_model = bool(save_model)
        self.osd_eval_shots = int(osd_eval_shots)
        expected_selection = (
            "neural_osd_paired_gain"
            if self.osd_eval_shots > 0
            else "neural_paired_gain"
        )
        configured_selection = self.experiment_config.get(
            "checkpoint_selection_metric"
        )
        if configured_selection != expected_selection:
            raise ValueError(
                "experiment_config checkpoint_selection_metric does not match "
                f"osd_eval_shots: expected {expected_selection!r}, got "
                f"{configured_selection!r}."
            )

        self.output_directory.mkdir(parents=True, exist_ok=True)
        self._configure_logging()

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.start_epoch = 0
        self.history: dict[str, Any] = {"train": [], "eval": [], "phases": []}
        self.best_score = float("-inf")
        self.best_tiebreak = float("-inf")
        self.best_validation_accuracy = float("-inf")
        self.best_selection_metric = ""
        self.best_epoch = -1
        # Held in memory as well as on disk. The model is small, and validation
        # selection must work even when checkpoints are not being written.
        self.best_state: dict[str, Tensor] | None = None
        self.best_optimizer_state: dict[str, Any] | None = None
        self.best_scheduler_state: dict[str, Any] | None = None
        self.best_history: dict[str, Any] | None = None

        if load_model_path is not None:
            self._load_checkpoint(Path(load_model_path))
            # A resumed invocation is a new OneCycle phase at the explicitly
            # requested LR while retaining Adam's moment estimates.
            for group in self.optimizer.param_groups:
                group["lr"] = float(learning_rate)
                group["initial_lr"] = float(learning_rate)
                group["weight_decay"] = float(weight_decay)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            epochs=self.epochs,
            steps_per_epoch=self.batches,
        )
        self.history.setdefault("phases", []).append(
            {
                "start_epoch": self.start_epoch,
                "epochs": self.epochs,
                "learning_rate": float(learning_rate),
            }
        )

        self._osd: OsdPostprocessor | None = None
        if self.osd_eval_shots > 0:
            graph = self.eval_generator.graph
            self._osd = OsdPostprocessor(
                graph.check_matrix,
                priors=graph.priors,
                method=osd_method,
                order=osd_order,
            )

    def _configure_logging(self) -> None:
        handler = logging.FileHandler(self.output_directory / "training_log.txt")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        if not any(
            isinstance(existing, logging.FileHandler)
            and existing.baseFilename == handler.baseFilename
            for existing in logger.handlers
        ):
            logger.addHandler(handler)

    def _validate_checkpoint_config(self, saved: dict[str, Any]) -> None:
        """Reject checkpoints built for a different circuit or model."""

        required = (
            "architecture",
            "circuit_schema_version",
            "code",
            "graph_fingerprint",
            "rounds",
            "detector_frames",
            "gate_error_rate",
            "measurement_error_rate",
            "idle_error_rate",
            "num_detectors",
            "num_mechanisms",
            "num_edges",
            "num_orbits",
            "bp_iterations",
            "bp_residual_hidden_dim",
            "bp_orbit_embedding_dim",
            "bp_parameter_sharing",
            "bp_normalisation",
            "bp_residual_scale",
            "bp_max_relaxation_delta",
            "bp_deep_supervision_weight",
            "bb_syndrome_loss_weight",
            "bb_logical_loss_weight",
            "bb_pauli_loss_weight",
            "bb_weight_decay",
            "checkpoint_selection_metric",
            "bb_osd_method",
            "bb_osd_order",
            "seed",
        )
        mismatches = [
            f"{key}: checkpoint={saved.get(key)!r}, "
            f"current={self.experiment_config.get(key)!r}"
            for key in required
            if saved.get(key) != self.experiment_config.get(key)
        ]
        if mismatches:
            raise ValueError(
                "Checkpoint is incompatible with this BB circuit graph/model. "
                "Version-1 circuits used different round semantics and must be "
                "retrained.\n  "
                + "\n  ".join(mismatches)
            )

    def _load_checkpoint(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"BB circuit checkpoint not found: {path}")
        try:
            checkpoint = torch.load(
                path, map_location=self.device, weights_only=False
            )
        except TypeError:  # PyTorch versions predating weights_only.
            checkpoint = torch.load(path, map_location=self.device)
        if int(checkpoint.get("format_version", 0)) < 3:
            raise ValueError(
                "This BB circuit checkpoint predates corrected noisy-round and "
                "strict-recovery semantics; it cannot be resumed safely."
            )
        saved_config = checkpoint.get("experiment_config")
        if not isinstance(saved_config, dict):
            raise ValueError("Checkpoint has no BB circuit experiment_config.")
        self._validate_checkpoint_config(saved_config)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        self.start_epoch = int(checkpoint.get("epoch", -1)) + 1
        if self.start_epoch < 0:
            raise ValueError(f"Invalid checkpoint epoch {self.start_epoch - 1}.")

        loaded_history = checkpoint.get("history")
        if isinstance(loaded_history, dict):
            self.history = copy.deepcopy(loaded_history)
            self.history.setdefault("train", [])
            self.history.setdefault("eval", [])
            self.history.setdefault("phases", [])
            # A previous phase's final report is retained in its checkpoint;
            # the new phase will write a new selected-best final report.
            previous_final = self.history.pop("final", None)
            if previous_final is not None:
                self.history.setdefault("phase_finals", []).append(previous_final)
            self.history.pop("best_epoch", None)

        self.best_epoch = int(checkpoint.get("best_epoch", -1))
        self.best_score = float(checkpoint.get("best_score", float("-inf")))
        self.best_tiebreak = float(
            checkpoint.get("best_tiebreak", float("-inf"))
        )
        self.best_validation_accuracy = float(
            checkpoint.get("best_validation_accuracy", float("-inf"))
        )
        self.best_selection_metric = str(
            checkpoint.get("best_selection_metric", "")
        )
        saved_best = checkpoint.get("best_model_state_dict")
        if isinstance(saved_best, dict):
            self.best_state = {
                key: value.detach().to(self.device).clone()
                for key, value in saved_best.items()
            }
        saved_best_optimizer = checkpoint.get("best_optimizer_state_dict")
        if isinstance(saved_best_optimizer, dict):
            self.best_optimizer_state = copy.deepcopy(saved_best_optimizer)
        saved_best_scheduler = checkpoint.get("best_scheduler_state_dict")
        if isinstance(saved_best_scheduler, dict):
            self.best_scheduler_state = copy.deepcopy(saved_best_scheduler)
        saved_best_history = checkpoint.get("best_history")
        if isinstance(saved_best_history, dict):
            self.best_history = copy.deepcopy(saved_best_history)

        torch_state = checkpoint.get("torch_rng_state")
        if torch_state is not None:
            torch.set_rng_state(torch_state.cpu())
        cuda_states = checkpoint.get("cuda_rng_state_all")
        if cuda_states is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_states)
        numpy_state = checkpoint.get("numpy_rng_state")
        if numpy_state is not None:
            np.random.set_state(numpy_state)

        logging.info(
            "Resumed BB circuit checkpoint %s after epoch %d; running %d "
            "additional epochs with a fresh Stim sampler phase.",
            path,
            self.start_epoch - 1,
            self.epochs,
        )

    # ------------------------------------------------------------------
    def _train_epoch(self) -> dict[str, float]:
        self.model.train()
        totals = {"total": 0.0, "syndrome": 0.0, "logical": 0.0, "mechanism": 0.0}
        for _ in range(self.batches):
            batch = self.generator.sample_dem(device=self.device)
            posterior, history = self.model(
                batch.detectors, neural=True, return_all=True
            )
            output = self.criterion(
                posterior, batch.detectors, batch.mechanisms, history
            )
            if not torch.isfinite(output.total):
                raise FloatingPointError(
                    "BB circuit loss became non-finite; checkpoint was not updated."
                )
            self.optimizer.zero_grad(set_to_none=True)
            output.total.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip,
                error_if_nonfinite=True,
            )
            self.optimizer.step()
            self.scheduler.step()
            totals["total"] += float(output.total.detach())
            totals["syndrome"] += float(output.syndrome.detach())
            totals["logical"] += float(output.logical.detach())
            totals["mechanism"] += float(output.mechanism.detach())
        return {key: value / self.batches for key, value in totals.items()}

    def _decode_hard(self, detectors: Tensor, *, neural: bool) -> np.ndarray:
        posterior = self.model(detectors, neural=neural)
        return EquivariantNeuralBP2.hard_decision(posterior).cpu().numpy(), posterior

    @torch.no_grad()
    def evaluate(self, batches: int) -> CircuitEvaluation:
        self.model.eval()
        graph = self.eval_generator.graph
        scoring = {
            "check_matrix": graph.check_matrix,
            "observable_matrix": graph.observable_matrix,
        }
        neural_success: list[np.ndarray] = []
        vanilla_success: list[np.ndarray] = []
        neural_converged: list[np.ndarray] = []
        vanilla_converged: list[np.ndarray] = []
        osd_neural: list[np.ndarray] = []
        osd_vanilla: list[np.ndarray] = []
        osd_budget = self.osd_eval_shots

        for _ in range(batches):
            batch = self.eval_generator.sample_circuit(device=self.device)
            detectors = batch.detectors.cpu().numpy().astype(np.uint8)
            observables = batch.observables.cpu().numpy().astype(np.uint8)

            hard_neural, posterior_neural = self._decode_hard(
                batch.detectors, neural=True
            )
            hard_vanilla, posterior_vanilla = self._decode_hard(
                batch.detectors, neural=False
            )
            neural_outcome = score_corrections(
                hard_neural, detectors=detectors, observables=observables, **scoring
            )
            vanilla_outcome = score_corrections(
                hard_vanilla, detectors=detectors, observables=observables, **scoring
            )
            neural_success.append(neural_outcome.success)
            vanilla_success.append(vanilla_outcome.success)
            neural_converged.append(neural_outcome.syndrome_converged)
            vanilla_converged.append(vanilla_outcome.syndrome_converged)

            if self._osd is not None and osd_budget > 0:
                take = min(osd_budget, detectors.shape[0])
                osd_budget -= take
                sliced = detectors[:take]
                neural_osd_outcome = score_corrections(
                    self._osd.decode_batch(
                        sliced,
                        posterior=posterior_neural[:take].cpu().numpy(),
                    ),
                    detectors=sliced,
                    observables=observables[:take],
                    **scoring,
                )
                vanilla_osd_outcome = score_corrections(
                    self._osd.decode_batch(
                        sliced,
                        posterior=posterior_vanilla[:take].cpu().numpy(),
                    ),
                    detectors=sliced,
                    observables=observables[:take],
                    **scoring,
                )
                if not neural_osd_outcome.syndrome_converged.all():
                    raise RuntimeError(
                        "Neural-posterior OSD returned a correction that does not "
                        "satisfy the detector syndrome."
                    )
                if not vanilla_osd_outcome.syndrome_converged.all():
                    raise RuntimeError(
                        "Vanilla-posterior OSD returned a correction that does not "
                        "satisfy the detector syndrome."
                    )
                osd_neural.append(neural_osd_outcome.success)
                osd_vanilla.append(vanilla_osd_outcome.success)

        neural = np.concatenate(neural_success)
        vanilla = np.concatenate(vanilla_success)
        neural_conv = np.concatenate(neural_converged)
        vanilla_conv = np.concatenate(vanilla_converged)
        gain, gain_error, rescued, harmed = _paired_gain(neural, vanilla)

        osd_fields: dict[str, float | None] = {
            "neural_osd_accuracy": None,
            "vanilla_osd_accuracy": None,
            "osd_paired_gain": None,
            "osd_paired_gain_error": None,
        }
        if osd_neural:
            osd_n = np.concatenate(osd_neural)
            osd_v = np.concatenate(osd_vanilla)
            osd_gain, osd_error, _, _ = _paired_gain(osd_n, osd_v)
            osd_fields = {
                "neural_osd_accuracy": float(osd_n.mean()),
                "vanilla_osd_accuracy": float(osd_v.mean()),
                "osd_paired_gain": osd_gain,
                "osd_paired_gain_error": osd_error,
            }

        return CircuitEvaluation(
            shots=int(neural.size),
            neural_accuracy=float(neural.mean()),
            vanilla_accuracy=float(vanilla.mean()),
            neural_converged=float(neural_conv.mean()),
            vanilla_converged=float(vanilla_conv.mean()),
            neural_flagged=float((~neural_conv).mean()),
            neural_unflagged=float((neural_conv & ~neural).mean()),
            paired_gain=gain,
            paired_gain_error=gain_error,
            rescued=rescued,
            harmed=harmed,
            osd_shots=sum(array.size for array in osd_neural),
            **osd_fields,
        )

    # ------------------------------------------------------------------
    def _checkpoint(self, epoch: int) -> dict[str, Any]:
        return {
            "format_version": 3,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": int(epoch),
            "experiment_config": copy.deepcopy(self.experiment_config),
            "history": copy.deepcopy(self.history),
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "best_tiebreak": self.best_tiebreak,
            "best_validation_accuracy": self.best_validation_accuracy,
            "best_selection_metric": self.best_selection_metric,
            "best_model_state_dict": self.best_state,
            "best_optimizer_state_dict": self.best_optimizer_state,
            "best_scheduler_state_dict": self.best_scheduler_state,
            "best_history": self.best_history,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "numpy_rng_state": np.random.get_state(),
        }

    def _save(self, epoch: int, name: str) -> None:
        if not self.save_model:
            return
        self._save_payload(self._checkpoint(epoch), name)

    def _save_payload(self, payload: dict[str, Any], name: str) -> None:
        """Atomically persist one checkpoint payload."""

        path = self.output_directory / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)

    def _save_selected_best(self) -> None:
        if not self.save_model or self.best_state is None or self.best_epoch < 0:
            return
        if (
            self.best_optimizer_state is None
            or self.best_scheduler_state is None
            or self.best_history is None
        ):
            raise RuntimeError("Selected-best training state is incomplete.")
        payload = self._checkpoint(self.best_epoch)
        payload.update(
            {
                "checkpoint_role": "selected_best",
                "model_state_dict": self.best_state,
                "optimizer_state_dict": self.best_optimizer_state,
                "scheduler_state_dict": self.best_scheduler_state,
                "history": self.best_history,
                "epoch": self.best_epoch,
            }
        )
        self._save_payload(payload, "best_model.pt")

    def _selection_key(
        self, evaluation: CircuitEvaluation
    ) -> tuple[str, float, float, float]:
        """Metric, primary score, tie-break and displayed decoder accuracy."""

        if self._osd is not None:
            if (
                evaluation.osd_paired_gain is None
                or evaluation.neural_osd_accuracy is None
                or evaluation.osd_shots < 1
            ):
                raise RuntimeError("OSD checkpoint selection has no OSD samples.")
            return (
                "neural_osd_paired_gain",
                float(evaluation.osd_paired_gain),
                float(evaluation.neural_osd_accuracy),
                float(evaluation.neural_osd_accuracy),
            )
        return (
            "neural_paired_gain",
            float(evaluation.paired_gain),
            float(evaluation.neural_accuracy),
            float(evaluation.neural_accuracy),
        )

    def _consider_best(self, epoch: int, evaluation: CircuitEvaluation) -> None:
        metric, score, tiebreak, accuracy = self._selection_key(evaluation)
        if (score, tiebreak) <= (self.best_score, self.best_tiebreak):
            return
        self.best_epoch = int(epoch)
        self.best_score = score
        self.best_tiebreak = tiebreak
        self.best_validation_accuracy = accuracy
        self.best_selection_metric = metric
        self.best_state = {
            key: value.detach().clone()
            for key, value in self.model.state_dict().items()
        }
        self.best_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        self.best_scheduler_state = copy.deepcopy(self.scheduler.state_dict())
        self.best_history = copy.deepcopy(self.history)
        self._save_selected_best()

    def _write_history(self) -> None:
        with open(self.output_directory / "history.json", "w") as handle:
            json.dump(
                {
                    "config": self.experiment_config,
                    **self.history,
                    "best_epoch": self.best_epoch,
                    "best_selection_metric": self.best_selection_metric,
                    "best_score": (
                        self.best_score if math.isfinite(self.best_score) else None
                    ),
                    "best_validation_accuracy": (
                        self.best_validation_accuracy
                        if math.isfinite(self.best_validation_accuracy)
                        else None
                    ),
                },
                handle,
                indent=2,
            )

    @staticmethod
    def _format(evaluation: CircuitEvaluation, epoch: int, prefix: str) -> str:
        line = (
            f"[{prefix}] Epoch: {epoch} | "
            f"Accuracy: {evaluation.neural_accuracy:.8f} | "
            f"Logical Error Rate: {1.0 - evaluation.neural_accuracy:.8f} | "
            f"Syndrome Convergence: {evaluation.neural_converged:.8f} | "
            f"Flagged: {evaluation.neural_flagged:.8f} | "
            f"Unflagged Logical: {evaluation.neural_unflagged:.8f} | "
            f"Vanilla BP Accuracy: {evaluation.vanilla_accuracy:.8f} | "
            f"Paired Gain: {evaluation.paired_gain:+.8f} "
            f"+/- {evaluation.paired_gain_error:.8f} | "
            f"Rescued: {evaluation.rescued} | Harmed: {evaluation.harmed} | "
            f"Eval Samples: {evaluation.shots}"
        )
        if evaluation.neural_osd_accuracy is not None:
            line += (
                f" | Neural+OSD: {evaluation.neural_osd_accuracy:.8f}"
                f" | BP+OSD: {evaluation.vanilla_osd_accuracy:.8f}"
                f" | OSD Paired Gain: {evaluation.osd_paired_gain:+.8f}"
                f" +/- {evaluation.osd_paired_gain_error:.8f}"
                f" | OSD Eval Samples: {evaluation.osd_shots}"
            )
        return line

    def train(self) -> None:
        logging.info("Configuration: %s", json.dumps(self.experiment_config))
        final_epoch = self.start_epoch + self.epochs - 1
        for epoch in range(self.start_epoch, final_epoch + 1):
            start = time.perf_counter()
            losses = self._train_epoch()
            logging.info(
                "[Train] Epoch: %d | Loss: %.8f | Syndrome: %.8f | "
                "Logical Surrogate: %.8f | Mechanism Aux: %.8f | LR: %g | Time: %.1fs",
                epoch,
                losses["total"],
                losses["syndrome"],
                losses["logical"],
                losses["mechanism"],
                self.scheduler.get_last_lr()[0],
                time.perf_counter() - start,
            )
            self.history["train"].append({"epoch": epoch, **losses})

            should_evaluate = (
                (epoch - self.start_epoch + 1) % self.eval_every == 0
                or epoch == final_epoch
            )
            if should_evaluate:
                evaluation = self.evaluate(self.eval_batches)
                logging.info(self._format(evaluation, epoch, "Epoch Eval"))
                self.history["eval"].append({"epoch": epoch, **asdict(evaluation)})
                self._consider_best(epoch, evaluation)
            # Save every epoch so an interruption between evaluations loses at
            # most one epoch, and keep history continuous across resume phases.
            self._write_history()
            self._save(epoch, "model.pt")

        if self.best_state is None or self.best_epoch < 0:
            raise RuntimeError("Training finished without an evaluated checkpoint.")
        latest_state = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.best_state)
        final = self.evaluate(self.final_eval_batches)
        logging.info(
            self._format(
                final,
                self.best_epoch,
                f"Selected Best ({self.best_selection_metric})",
            )
        )
        self.history["final"] = asdict(final)
        self.history["best_epoch"] = self.best_epoch
        self._write_history()
        # Keep model.pt resumable from the latest optimisation state, while
        # best_model.pt and the final report refer to the selected checkpoint.
        self.model.load_state_dict(latest_state)
        # A resumed output directory must still contain the inherited selected
        # best even when no new epoch beats it.
        self._save_selected_best()
        self._save(final_epoch, "model.pt")


__all__ = ["BBCircuitTrainer", "CircuitEvaluation"]
