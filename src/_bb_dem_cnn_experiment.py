"""DEM CNN training; DEM labels for training, exact circuit shots for scoring."""

from __future__ import annotations

import copy
from dataclasses import asdict
import datetime
import json
import logging
from pathlib import Path
import time

import numpy as np
import torch

from models._bb_dem_cnn import BBDemCNN
from ._bb_circuit_experiment import _graph_fingerprint
from ._bb_circuit_loss import CircuitDegeneracyAwareLoss
from ._bb_circuit_metrics import score_corrections
from ._direct_osd import DirectOSD0
from .bb_circuit_data import BBCircuitGenerator
from .bb_code import BBCodeSpec
from .bb_stim_utils import CIRCUIT_SCHEMA_VERSION


def _read_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "bb_dem_cnn_v1":
        raise ValueError("Expected a bb_dem_cnn_v1 checkpoint.")
    return checkpoint


class BBDemCNNTrainer:
    """Select by raw correction success, retaining the earliest validation tie."""

    def __init__(self, *, model, train_generator, eval_generator, criterion, device,
                 output_directory, config, epochs, batches, eval_batches, eval_every,
                 final_eval_batches, learning_rate=3e-4, weight_decay=1e-4,
                 gradient_clip=1., osd_eval_shots=0, save_model=False, load_model_path=None):
        for name, value in (("epochs", epochs), ("batches", batches),
                            ("eval_batches", eval_batches), ("eval_every", eval_every),
                            ("final_eval_batches", final_eval_batches)):
            if value < 1:
                raise ValueError(f"{name} must be positive.")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if not np.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative.")
        if not np.isfinite(gradient_clip) or gradient_clip <= 0:
            raise ValueError("gradient_clip must be finite and positive.")
        if osd_eval_shots < 0:
            raise ValueError("osd_eval_shots must be nonnegative.")
        self.model, self.criterion, self.device = model, criterion, device
        self.train_generator, self.eval_generator = train_generator, eval_generator
        self.graph = train_generator.graph
        if _graph_fingerprint(self.graph) != _graph_fingerprint(eval_generator.graph):
            raise ValueError("Training and evaluation graphs must match.")
        self.directory, self.config = Path(output_directory), copy.deepcopy(config)
        self.epochs, self.batches = epochs, batches
        self.eval_batches, self.eval_every = eval_batches, eval_every
        self.final_eval_batches, self.gradient_clip = final_eval_batches, gradient_clip
        self.osd_eval_shots, self.save_model = osd_eval_shots, save_model
        self.osd = DirectOSD0(self.graph.check_matrix) if osd_eval_shots else None
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.start_epoch, self.best_epoch, self.best_accuracy = 0, -1, -1.
        self.best_state, self.best_snapshot = None, None
        self.history = {"train": [], "eval": [], "phases": []}
        if load_model_path:
            saved = _read_checkpoint(load_model_path)
            differences = [key for key in self.config if saved["config"].get(key) != self.config[key]]
            if differences:
                raise ValueError("Incompatible DEM CNN checkpoint: " + ", ".join(differences))
            self.model.load_state_dict(saved["model_state_dict"])
            self.optimizer.load_state_dict(saved["optimizer_state_dict"])
            self.start_epoch = saved["epoch"] + 1
            self.best_epoch, self.best_accuracy = saved["best_epoch"], saved["best_accuracy"]
            self.best_state = saved["best_model_state_dict"]
            self.best_snapshot = saved.get("best_snapshot")
            if self.best_snapshot is None and saved["epoch"] == self.best_epoch:
                self.best_snapshot = copy.deepcopy(saved)
            self.history = copy.deepcopy(saved["history"])
            self.history.pop("final", None)
            torch.set_rng_state(saved["torch_rng_state"])
            if torch.cuda.is_available() and saved.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
            for group in self.optimizer.param_groups:
                group.update(lr=learning_rate, initial_lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs * batches)
        self.history["phases"].append({
            "start_epoch": self.start_epoch, "epochs": epochs, "batches": batches,
            "batch_size": train_generator.batch_size, "learning_rate": learning_rate,
            "weight_decay": weight_decay, "gradient_clip": gradient_clip,
            "eval_batches": eval_batches, "eval_every": eval_every,
            "final_eval_batches": final_eval_batches, "osd_eval_shots": osd_eval_shots,
            "edge_chunk_size": model.first.edge_chunk_size,
            "gradient_checkpoint": model.gradient_checkpoint,
            "schedule": "cosine_restart" if load_model_path else "cosine",
            "sampler_policy": "independent_train_eval_seeds_from_base_seed_and_phase_start_epoch",
            "resume_source": str(load_model_path) if load_model_path else None,
        })
        self.directory.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format="%(message)s", force=True,
                            handlers=[logging.FileHandler(self.directory / "training_log.txt"),
                                      logging.StreamHandler()])
        logging.info("DEM CNN configuration: %s", json.dumps(self.config))
        logging.info("%s", self.graph.summary())

    def _checkpoint(self, epoch):
        return {
            "format": "bb_dem_cnn_v1", "epoch": epoch, "config": copy.deepcopy(self.config),
            "model_state_dict": copy.deepcopy(self.model.state_dict()),
            "optimizer_state_dict": copy.deepcopy(self.optimizer.state_dict()),
            "history": copy.deepcopy(self.history), "best_epoch": self.best_epoch,
            "best_accuracy": self.best_accuracy, "best_model_state_dict": copy.deepcopy(self.best_state),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    def _save(self, data, filename):
        if self.save_model:
            temporary = self.directory / (filename + ".tmp")
            torch.save(data, temporary)
            temporary.replace(self.directory / filename)

    def _write_history(self):
        temporary = self.directory / "history.json.tmp"
        temporary.write_text(json.dumps({"config": self.config, **self.history,
                                         "best_epoch": self.best_epoch,
                                         "best_raw_accuracy": self.best_accuracy},
                                        indent=2, allow_nan=False) + "\n")
        temporary.replace(self.directory / "history.json")

    def _train_epoch(self):
        self.model.train()
        totals = dict(total=0., syndrome=0., logical=0., mechanism=0.)
        for _ in range(self.batches):
            batch = self.train_generator.sample_dem(device=self.device)
            self.optimizer.zero_grad(set_to_none=True)
            llr = self.model(batch.detectors)
            losses = self.criterion(llr, batch.detectors, batch.mechanisms)
            if not torch.isfinite(losses.total):
                raise FloatingPointError("Non-finite DEM CNN loss.")
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip, error_if_nonfinite=True)
            self.optimizer.step()
            self.scheduler.step()
            for key in totals:
                totals[key] += float(getattr(losses, key).detach())
        return {key: value / self.batches for key, value in totals.items()}

    def _score(self, correction, detectors, observables):
        return score_corrections(correction, detectors=detectors, observables=observables,
                                 check_matrix=self.graph.check_matrix,
                                 observable_matrix=self.graph.observable_matrix)

    @staticmethod
    def _aggregate(success, converged, logical_match):
        success, converged, logical_match = map(np.concatenate, (success, converged, logical_match))
        return {
            "shots": int(success.size), "failures": int((~success).sum()),
            "logical_accuracy": float(success.mean()),
            "logical_error_rate": float((~success).mean()),
            "syndrome_convergence": float(converged.mean()),
            "flagged_failure": float((~converged).mean()),
            "unflagged_failure": float((converged & ~success).mean()),
            "observable_mismatch_rate": float((~logical_match).mean()),
        }

    @torch.no_grad()
    def evaluate(self, batches, *, save_shots=False):
        self.model.eval()
        raw, repaired = [[], [], []], [[], [], []]
        paired_raw, paired_fixed = [], []
        bank = {key: [] for key in ("detectors", "observables", "raw_correction", "raw_success",
                                    "raw_converged", "predicted_observables")}
        osd_bank = {key: [] for key in ("osd_indices", "osd_correction", "osd_success")}
        offset, osd_shots, osd_calls = 0, 0, 0
        nn_seconds, osd_seconds = 0., 0.
        for _ in range(batches):
            batch = self.eval_generator.sample_circuit(device=self.device)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            llr = self.model(batch.detectors)
            if not torch.isfinite(llr).all():
                raise FloatingPointError("Non-finite DEM CNN posterior.")
            hard = self.model.hard_decision(llr)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            nn_seconds += time.perf_counter() - started
            correction = hard.cpu().numpy()
            detectors = batch.detectors.cpu().numpy().astype(np.uint8)
            observables = batch.observables.cpu().numpy().astype(np.uint8)
            outcome = self._score(correction, detectors, observables)
            predicted = np.asarray(self.graph.observable_matrix @ correction.T).T % 2
            match = np.all(predicted == observables, axis=1)
            for dest, value in zip(raw, (outcome.success, outcome.syndrome_converged, match)):
                dest.append(value)
            count = min(len(correction), self.osd_eval_shots - osd_shots)
            if count > 0:
                started = time.perf_counter()
                fixed = correction[:count].copy()
                invalid = ~outcome.syndrome_converged[:count]
                if invalid.any():
                    probabilities = self.model.probabilities(llr[:count]).cpu().numpy()
                    fixed[invalid] = self.osd.decode_batch(detectors[:count][invalid], probabilities[invalid])
                after = self._score(fixed, detectors[:count], observables[:count])
                if not after.syndrome_converged.all():
                    raise RuntimeError("Direct OSD-0 returned an inconsistent correction.")
                osd_seconds += time.perf_counter() - started
                for dest, value in zip(repaired, (after.success, after.syndrome_converged, after.success)):
                    dest.append(value)
                paired_raw.append(outcome.success[:count])
                paired_fixed.append(after.success)
                osd_calls += int(invalid.sum())
                osd_shots += count
                if save_shots:
                    for key, value in (("osd_indices", np.arange(offset, offset + count)),
                                       ("osd_correction", fixed), ("osd_success", after.success)):
                        osd_bank[key].append(value)
            if save_shots:
                for key, value in (("detectors", detectors), ("observables", observables),
                                   ("raw_correction", correction), ("raw_success", outcome.success),
                                   ("raw_converged", outcome.syndrome_converged),
                                   ("predicted_observables", predicted)):
                    bank[key].append(value)
            offset += len(correction)
        result = {"shots": offset, "raw": self._aggregate(*raw), "osd_shots": osd_shots,
                  "nn_seconds": nn_seconds, "osd_with_transfer_seconds": osd_seconds}
        if osd_shots:
            before, after = np.concatenate(paired_raw), np.concatenate(paired_fixed)
            difference = after.astype(float) - before.astype(float)
            result.update(osd=self._aggregate(*repaired), osd_call_fraction=osd_calls / osd_shots,
                          osd_minus_raw_gain=float(difference.mean()),
                          paired_raw_accuracy=float(before.mean()),
                          paired_se=float(difference.std(ddof=1) / np.sqrt(osd_shots)) if osd_shots > 1 else None,
                          rescued=int((after & ~before).sum()), harmed=int((before & ~after).sum()))
        if save_shots:
            np.savez_compressed(self.directory / "final_shots.npz",
                                **{k: np.concatenate(v) for k, v in {**bank, **osd_bank}.items() if v},
                                metadata_json=json.dumps(self.config))
        return result

    def train(self):
        final_epoch = self.start_epoch + self.epochs - 1
        for epoch in range(self.start_epoch, final_epoch + 1):
            losses = self._train_epoch()
            self.history["train"].append({"epoch": epoch, **losses})
            logging.info("[Train] epoch %d | %s", epoch, json.dumps(losses))
            if (epoch - self.start_epoch + 1) % self.eval_every == 0 or epoch == final_epoch:
                row = self.evaluate(self.eval_batches)
                self.history["eval"].append({"epoch": epoch, **row})
                logging.info("[Validation] epoch %d | %s", epoch, json.dumps(row))
                if row["raw"]["logical_accuracy"] > self.best_accuracy:
                    self.best_epoch, self.best_accuracy = epoch, row["raw"]["logical_accuracy"]
                    self.best_state = copy.deepcopy(self.model.state_dict())
                    self.best_snapshot = self._checkpoint(epoch)
            latest = self._checkpoint(epoch)
            latest["best_snapshot"] = self.best_snapshot
            self._save(latest, "model.pt")
            if self.best_snapshot is not None:
                self._save(self.best_snapshot, "best_model.pt")
            self._write_history()
        latest_weights = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.best_state)
        try:
            self.history["final"] = {"epoch": self.best_epoch,
                                     **self.evaluate(self.final_eval_batches, save_shots=True)}
        finally:
            self.model.load_state_dict(latest_weights)
        logging.info("[Final selected best] %s", json.dumps(self.history["final"]))
        self._write_history()
        latest = self._checkpoint(final_epoch)
        latest["best_snapshot"] = self.best_snapshot
        self._save(latest, "model.pt")
        return str(self.directory)


def run_bb_dem_cnn_experiment(args):
    """Launch an independent DEM-CNN experiment, including resumable training."""
    phase_epoch = 0
    if args.load_model:
        saved = _read_checkpoint(args.load_model)
        seed, phase_epoch = saved["config"]["seed"], saved["epoch"] + 1
        if args.seed is not None and args.seed != seed:
            raise ValueError("Resume seed does not match the checkpoint.")
    else:
        seed = args.seed if args.seed is not None else int(np.random.SeedSequence().generate_state(1)[0])
    train_seed, eval_seed, host_seed = np.random.SeedSequence(
        [seed, phase_epoch, CIRCUIT_SCHEMA_VERSION]).spawn(3)
    torch.manual_seed(int(host_seed.generate_state(1)[0]))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    code = BBCodeSpec.from_name(args.code)
    kwargs = dict(code=code, rounds=args.rounds, gate_error_rate=args.p,
                  measurement_error_rate=args.measurement_error_rate,
                  idle_error_rate=args.bb_idle_error_rate,
                  circuit_noise_model=args.bb_circuit_noise_model, batch_size=args.batch_size)
    train = BBCircuitGenerator(**kwargs, seed=int(train_seed.generate_state(1)[0]))
    evaluation = BBCircuitGenerator(**kwargs, seed=int(eval_seed.generate_state(1)[0]), graph=train.graph)
    model = BBDemCNN(train.graph, width=args.bb_cnn_width, depth=args.bb_cnn_depth,
                     edge_chunk_size=args.edge_chunk_size,
                     gradient_checkpoint=not args.bb_dem_cnn_no_checkpoint).to(device)
    loss_weights = dict(syndrome_weight=args.bb_syndrome_loss_weight,
                        logical_weight=args.bb_logical_loss_weight,
                        mechanism_weight=args.bb_pauli_loss_weight, deep_supervision_weight=0.)
    criterion = CircuitDegeneracyAwareLoss(check_matrix=train.graph.check_matrix,
                                          observable_matrix=train.graph.observable_matrix,
                                          **loss_weights).to(device)
    config = dict(architecture="bb_dem_cnn", noise_model="circuit", code=code.name,
                  n=code.n, k=code.k, d=code.d, rounds=train.rounds,
                  circuit_schema_version=CIRCUIT_SCHEMA_VERSION,
                  detector_frames=train.graph.detector_frames,
                  num_detectors=train.graph.num_detectors, num_mechanisms=train.graph.num_mechanisms,
                  num_observables=train.graph.num_observables, num_orbits=train.graph.num_orbits,
                  noise_profile=asdict(train.noise_profile), graph_fingerprint=_graph_fingerprint(train.graph),
                  dem_fingerprint=train.graph.dem_fingerprint, seed=seed,
                  width=args.bb_cnn_width, depth=args.bb_cnn_depth,
                  kernel="orbit_factorized_channel_gate_v1", loss_weights=loss_weights,
                  checkpoint_selection="raw_syndrome_and_observable_accuracy",
                  osd_policy="optional_direct_hard_centered_osd0_v1",
                  parameter_count=sum(p.numel() for p in model.parameters()))
    now = datetime.datetime.now()
    directory = Path.cwd() / "outputs" / f"{now:%Y-%m-%d}" / (
        f"{now:%H-%M-%S-%f}_circuit_{code.name}_{train.noise_profile.name}_bb_dem_cnn_"
        f"p{args.p:g}_r{train.rounds}_w{args.bb_cnn_width}_d{args.bb_cnn_depth}_seed{seed}")
    trainer = BBDemCNNTrainer(
        model=model, train_generator=train, eval_generator=evaluation, criterion=criterion,
        device=device, output_directory=directory, config=config, epochs=args.epochs,
        batches=args.batches, eval_batches=args.eval_batches, eval_every=args.eval_every,
        final_eval_batches=args.final_eval_batches or args.eval_batches,
        learning_rate=args.lr if args.lr is not None else 3e-4, weight_decay=args.bb_weight_decay,
        gradient_clip=args.bb_cnn_gradient_clip, osd_eval_shots=args.bb_osd_eval_shots,
        save_model=args.save_model, load_model_path=args.load_model)
    return trainer.train()
