"""Training and paired raw/direct-OSD evaluation of the joint BB Tanner CNN."""

from __future__ import annotations

import copy
from dataclasses import asdict
import datetime
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import torch

from models._bb_tanner_cnn import BBTannerCNN
from ._bb_loss import DegeneracyAwareBPLoss
from ._bb_metrics import aggregate_bb_outcomes, bb_correction_outcomes, paired_success_gain
from ._direct_osd import DirectOSD0
from .bb_code import BBCodeSpec
from .bb_data_generator import BBCodeCapacityGenerator


class BBTannerCNNTrainer:
    def __init__(self, *, model, code, train_generator, eval_generator, criterion,
                 output_directory, config, device, epochs, batches, batch_size,
                 eval_batches, eval_every, final_eval_batches, learning_rate=3e-4,
                 weight_decay=1e-4, gradient_clip=1., save_model=False, load_model_path=None):
        for name, value in (("epochs", epochs), ("batches", batches), ("batch_size", batch_size),
                            ("eval_batches", eval_batches), ("eval_every", eval_every),
                            ("final_eval_batches", final_eval_batches)):
            if value < 1:
                raise ValueError(f"{name} must be positive.")
        self.model, self.code, self.criterion, self.device = model, code, criterion, device
        self.train_generator, self.eval_generator = train_generator, eval_generator
        self.directory = Path(output_directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config = copy.deepcopy(config)
        self.epochs, self.batches, self.batch_size = epochs, batches, batch_size
        self.eval_batches, self.eval_every, self.final_eval_batches = eval_batches, eval_every, final_eval_batches
        self.gradient_clip, self.save_model = gradient_clip, save_model
        self.osd_x, self.osd_z = DirectOSD0(code.hz), DirectOSD0(code.hx)
        self.scoring = {key: getattr(criterion, key) for key in ("hx", "hz", "logicals_x", "logicals_z")}
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.history = {"train": [], "eval": [], "phases": []}
        self.start_epoch = 0
        self.best_epoch, self.best_accuracy, self.best_state = -1, -1., None
        self.best_snapshot = None
        if load_model_path:
            self._load(load_model_path)
            for group in self.optimizer.param_groups:
                group.update(lr=learning_rate, initial_lr=learning_rate, weight_decay=weight_decay)
        # New schedule per resumed phase; optimizer moments and all RNG streams persist.
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs * batches,
        )
        self.history["phases"].append({"start_epoch": self.start_epoch, "epochs": epochs,
                                       "learning_rate": learning_rate, "batches": batches,
                                       "batch_size": batch_size, "eval_batches": eval_batches,
                                       "eval_every": eval_every, "final_eval_batches": final_eval_batches})
        logging.basicConfig(level=logging.INFO, format="%(message)s", force=True,
                            handlers=[logging.FileHandler(self.directory / "training_log.txt"),
                                      logging.StreamHandler()])
        logging.info("Executed command: python %s", " ".join(sys.argv))
        logging.info("Configuration: %s", json.dumps(config))

    def _load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("format") != "bb_tanner_cnn_v1":
            raise ValueError("Expected a BB Tanner CNN checkpoint.")
        saved = checkpoint["config"]
        mismatches = [key for key, value in self.config.items()
                      if key not in {"parameter_count"} and saved.get(key) != value]
        if mismatches:
            raise ValueError("Incompatible CNN checkpoint settings: " + ", ".join(mismatches))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.history = copy.deepcopy(checkpoint["history"])
        self.history.pop("final", None)
        self.best_epoch, self.best_accuracy = checkpoint["best_epoch"], checkpoint["best_accuracy"]
        self.best_state = copy.deepcopy(checkpoint["best_model_state_dict"])
        self.best_snapshot = checkpoint.get("best_snapshot")
        if self.best_snapshot is None and self.best_epoch == checkpoint["epoch"]:
            self.best_snapshot = copy.deepcopy(checkpoint)
        self.train_generator.load_state_dict(checkpoint["train_generator_state"])
        self.eval_generator.load_state_dict(checkpoint["eval_generator_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state"]])

    def _checkpoint(self, epoch):
        return {
            "format": "bb_tanner_cnn_v1", "epoch": epoch, "config": copy.deepcopy(self.config),
            "model_state_dict": copy.deepcopy(self.model.state_dict()),
            "optimizer_state_dict": copy.deepcopy(self.optimizer.state_dict()),
            "history": copy.deepcopy(self.history), "best_epoch": self.best_epoch,
            "best_accuracy": self.best_accuracy, "best_model_state_dict": self.best_state,
            "train_generator_state": self.train_generator.state_dict(),
            "eval_generator_state": self.eval_generator.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    def _save(self, value, name):
        if self.save_model:
            temporary = self.directory / (name + ".tmp")
            torch.save(value, temporary)
            temporary.replace(self.directory / name)

    def _write_history(self):
        temporary = self.directory / "history.json.tmp"
        temporary.write_text(json.dumps({"config": self.config, **self.history,
                                          "best_epoch": self.best_epoch,
                                          "best_raw_accuracy": self.best_accuracy}, indent=2,
                                         allow_nan=False) + "\n")
        temporary.replace(self.directory / "history.json")

    def _train_epoch(self):
        self.model.train()
        totals = dict(total=0., syndrome=0., logical=0., pauli=0.)
        for _ in range(self.batches):
            batch = self.train_generator.sample(self.batch_size, device=self.device)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(batch.syndrome, channel_probabilities=batch.channel_probabilities)
            losses = self.criterion(logits, batch.syndrome, batch.pauli)
            if not torch.isfinite(losses.total):
                raise FloatingPointError("Non-finite Tanner CNN loss.")
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip, error_if_nonfinite=True)
            self.optimizer.step()
            self.scheduler.step()
            for key in totals:
                totals[key] += float(getattr(losses, key).detach())
        return {key: value / self.batches for key, value in totals.items()}

    @torch.no_grad()
    def evaluate(self, batches, *, save_shots=False):
        self.model.eval()
        raw, repaired = [], []
        bank = {key: [] for key in ("syndrome", "pauli", "raw_correction", "osd_correction",
                                   "raw_success", "osd_success")}
        seconds_nn, seconds_osd, called_x, called_z, called_any = 0., 0., 0, 0, 0
        for _ in range(batches):
            batch = self.eval_generator.sample(self.batch_size, device=self.device)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            logits = self.model(batch.syndrome, channel_probabilities=batch.channel_probabilities)
            correction = self.model.hard_decision(logits)
            qx, qz = self.model.component_probabilities(logits)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            seconds_nn += time.perf_counter() - started
            raw.append(bb_correction_outcomes(correction, batch.syndrome, batch.pauli, **self.scoring))
            started = time.perf_counter()
            syndrome = batch.syndrome.cpu().numpy().astype(np.uint8)
            px, pz = qx.cpu().numpy(), qz.cpu().numpy()
            hard_x, hard_z = (px > .5).astype(np.uint8), (pz > .5).astype(np.uint8)
            need_x = np.any((hard_x @ self.code.hz.T) % 2 != syndrome[:, self.code.num_x_checks:], axis=1)
            need_z = np.any((hard_z @ self.code.hx.T) % 2 != syndrome[:, :self.code.num_x_checks], axis=1)
            called_x += int(need_x.sum())
            called_z += int(need_z.sum())
            called_any += int((need_x | need_z).sum())
            # Only inconsistent sectors are repaired; valid hard decisions are
            # kept even if their unknown logical class is wrong.
            if need_x.any():
                hard_x[need_x] = self.osd_x.decode_batch(syndrome[need_x, self.code.num_x_checks:], px[need_x])
            if need_z.any():
                hard_z[need_z] = self.osd_z.decode_batch(syndrome[need_z, :self.code.num_x_checks], pz[need_z])
            osd_correction = torch.as_tensor(hard_x + 3 * hard_z - 2 * hard_x * hard_z, device=self.device)
            seconds_osd += time.perf_counter() - started
            outcome = bb_correction_outcomes(osd_correction, batch.syndrome, batch.pauli, **self.scoring)
            if not outcome.syndrome_converged.all():
                raise RuntimeError("CNN+OSD returned an invalid syndrome.")
            repaired.append(outcome)
            if save_shots:
                for key, value in (("syndrome", batch.syndrome), ("pauli", batch.pauli),
                                   ("raw_correction", correction), ("osd_correction", osd_correction),
                                   ("raw_success", raw[-1].success), ("osd_success", outcome.success)):
                    bank[key].append(value.cpu().numpy().astype(np.uint8))
        gain, se, rescued, harmed = paired_success_gain(repaired, raw)
        raw_metrics, osd_metrics = aggregate_bb_outcomes(raw), aggregate_bb_outcomes(repaired)
        result = {"shots": raw_metrics.samples, "raw": asdict(raw_metrics), "osd": asdict(osd_metrics),
                  "osd_minus_raw_gain": gain, "paired_gain_se": se,
                  "rescued": rescued, "harmed": harmed,
                  "osd_called_shots": called_any, "osd_call_fraction": called_any / raw_metrics.samples,
                  "osd_x_calls": called_x, "osd_z_calls": called_z,
                  "nn_batch_seconds": seconds_nn, "osd_with_transfer_seconds": seconds_osd}
        if save_shots:
            np.savez_compressed(self.directory / "final_shots.npz",
                                **{key: np.concatenate(values) for key, values in bank.items()},
                                metadata_json=json.dumps(self.config))
        return result

    @staticmethod
    def _log_eval(label, epoch, row):
        logging.info("[%s] Epoch %d | CNN LER %.8f | CNN+OSD-0 LER %.8f | "
                     "CNN convergence %.8f | OSD called %.4f | paired gain %+.8f | shots %d",
                     label, epoch, row["raw"]["logical_error_rate"], row["osd"]["logical_error_rate"],
                     row["raw"]["syndrome_convergence"], row["osd_call_fraction"],
                     row["osd_minus_raw_gain"], row["shots"])

    def train(self):
        final_epoch = self.start_epoch + self.epochs - 1
        for epoch in range(self.start_epoch, final_epoch + 1):
            losses = self._train_epoch()
            self.history["train"].append({"epoch": epoch, **losses})
            logging.info("[Train] Epoch %d | %s", epoch, json.dumps(losses))
            if (epoch - self.start_epoch + 1) % self.eval_every == 0 or epoch == final_epoch:
                row = self.evaluate(self.eval_batches)
                self.history["eval"].append({"epoch": epoch, **row})
                self._log_eval("Validation", epoch, row)
                if row["raw"]["logical_accuracy"] > self.best_accuracy:
                    self.best_epoch, self.best_accuracy = epoch, row["raw"]["logical_accuracy"]
                    self.best_state = copy.deepcopy(self.model.state_dict())
                    # Selected weights, optimizer, history and sampler streams all
                    # belong to this epoch. No recursive snapshot nesting.
                    self.best_snapshot = self._checkpoint(epoch)
            latest = self._checkpoint(epoch)
            latest["best_snapshot"] = self.best_snapshot
            self._save(latest, "model.pt")
            if self.best_snapshot is not None:
                self._save(self.best_snapshot, "best_model.pt")
            self._write_history()
        if self.best_state is None:
            raise RuntimeError("No evaluated CNN checkpoint was selected.")
        latest_weights = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.best_state)
        row = self.evaluate(self.final_eval_batches, save_shots=True)
        self.history["final"] = {"epoch": self.best_epoch, **row}
        self._log_eval("Selected best by raw LER", self.best_epoch, row)
        self.model.load_state_dict(latest_weights)
        self._write_history()
        latest = self._checkpoint(final_epoch)
        latest["best_snapshot"] = self.best_snapshot
        self._save(latest, "model.pt")
        self._save(self.best_snapshot, "best_model.pt")


def run_bb_tanner_cnn_experiment(args):
    code = BBCodeSpec.from_name(args.code)
    if args.load_model:
        saved = torch.load(args.load_model, map_location="cpu", weights_only=False)
        if saved.get("format") != "bb_tanner_cnn_v1":
            raise ValueError("Expected a BB Tanner CNN checkpoint.")
        seed = saved["config"]["seed"]
        if args.seed is not None and args.seed != seed:
            raise ValueError("Resume seed does not match the saved CNN experiment.")
    else:
        seed = args.seed if args.seed is not None else int(np.random.SeedSequence().generate_state(1)[0])
    torch.manual_seed(seed)
    train_seed, eval_seed = np.random.SeedSequence(seed).spawn(2)
    common = dict(code=code, error_rate=args.p, batch_size=args.batch_size, noise_model=args.bb_channel,
                  x_error_rate=args.x_error_rate, z_error_rate=args.z_error_rate)
    train_generator = BBCodeCapacityGenerator(**common, seed=int(train_seed.generate_state(1)[0]))
    eval_generator = BBCodeCapacityGenerator(**common, seed=int(eval_seed.generate_state(1)[0]))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BBTannerCNN(code, width=args.bb_cnn_width, depth=args.bb_cnn_depth).to(device)
    buffers = code.torch_buffers(device=device)
    criterion = DegeneracyAwareBPLoss(
        **{key: buffers[key] for key in ("hx", "hz", "logicals_x", "logicals_z")},
        syndrome_weight=args.bb_syndrome_loss_weight, logical_weight=args.bb_logical_loss_weight,
        pauli_weight=args.bb_pauli_loss_weight, deep_supervision_weight=0.,
    ).to(device)
    fingerprint = hashlib.sha256(code.hx.tobytes() + code.hz.tobytes()).hexdigest()
    config = dict(architecture="bb_tanner_cnn", code=code.name, n=code.n, k=code.k, d=code.d,
                  graph_fingerprint=fingerprint, noise_model="capacity", channel=args.bb_channel,
                  error_rate=args.p, x_error_rate=args.x_error_rate, z_error_rate=args.z_error_rate,
                  channel_probabilities=train_generator.channel_probabilities.tolist(), seed=seed,
                  cnn_width=args.bb_cnn_width, cnn_depth=args.bb_cnn_depth,
                  syndrome_loss_weight=args.bb_syndrome_loss_weight,
                  logical_loss_weight=args.bb_logical_loss_weight, pauli_loss_weight=args.bb_pauli_loss_weight,
                  weight_decay=args.bb_weight_decay, gradient_clip=args.bb_cnn_gradient_clip,
                  hard_decision="component_marginal_gt_half", checkpoint_selection="raw_logical_accuracy",
                  osd="direct_hard_centered_osd0_v1", osd_order=0, osd_extra_bp_iterations=0,
                  osd_sectors="css_separated", osd_policy="only_invalid_sectors",
                  parameter_count=sum(p.numel() for p in model.parameters()), dtype="float32")
    now = datetime.datetime.now()
    output = Path.cwd() / "outputs" / now.strftime("%Y-%m-%d") / (
        f"{now:%H-%M-%S-%f}_capacity_{code.name}_bb_tanner_cnn_{args.bb_channel}_p{args.p:g}"
        f"_w{args.bb_cnn_width}_depth{args.bb_cnn_depth}_seed{seed}"
    )
    trainer = BBTannerCNNTrainer(
        model=model, code=code, train_generator=train_generator, eval_generator=eval_generator,
        criterion=criterion, output_directory=output, config=config, device=device,
        epochs=args.epochs, batches=args.batches, batch_size=args.batch_size,
        eval_batches=args.eval_batches, eval_every=args.eval_every,
        final_eval_batches=args.final_eval_batches or args.eval_batches,
        learning_rate=args.lr if args.lr is not None else 3e-4, weight_decay=args.bb_weight_decay,
        gradient_clip=args.bb_cnn_gradient_clip, save_model=args.save_model, load_model_path=args.load_model,
    )
    logging.info("Joint Tanner CNN: depth=%d width=%d | device=%s | parameters=%d",
                 model.depth, model.width, device, config["parameter_count"])
    logging.info("Paired CNN/CNN+direct OSD-0 evaluation; best selection uses raw CNN LER. Output: %s", output)
    if args.amp_dtype != "none":
        logging.warning("Tanner CNN and parity loss use float32; --amp_dtype=%s is not applied.", args.amp_dtype)
    trainer.train()
    return str(output)
