import math

import torch
from torch import nn

from src._trainer import Trainer
from src.metrics import paired_decoder_metrics


class _FixedGenerator:
    def __init__(self, baseline, truth):
        self.baseline = torch.as_tensor(baseline, dtype=torch.float32)
        self.truth = torch.as_tensor(truth, dtype=torch.long)

    def generate_batch(self, device):
        # The fake model reads a baseline class from the only input column.
        return self.baseline[:, None].to(device), self.truth.to(device)


class _FakeHybrid(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.residual_bias = nn.Parameter(torch.zeros(num_classes))
        self.register_buffer("gate_enabled", torch.tensor(False))
        self._last_baseline_classes = None
        self._last_residual_logits = None
        self.calibration_calls = 0

    @property
    def last_baseline_classes(self):
        return self._last_baseline_classes

    @property
    def last_residual_logits(self):
        return self._last_residual_logits

    def forward(self, inputs):
        baseline = inputs[:, 0].long()
        raw = self.residual_bias.unsqueeze(0).expand(inputs.shape[0], -1)
        self._last_baseline_classes = baseline
        self._last_residual_logits = raw
        indices = torch.arange(raw.shape[1], device=raw.device)
        permutation = torch.bitwise_xor(
            indices.unsqueeze(0), baseline.unsqueeze(1)
        )
        return raw.gather(1, permutation)

    def calibrate_gate(self, logits, true_classes, baseline_classes):
        self.calibration_calls += 1
        self.gate_enabled.fill_(True)
        return {
            "enabled": True,
            "margin": torch.tensor(0.25),
            "calibration_samples": logits.shape[0],
        }


class _CountingCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, output, target):
        self.calls += 1
        return nn.functional.cross_entropy(output, target)


def _trainer(tmp_path, *, criterion=None, calibration_batches=1):
    model = _FakeHybrid()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-4,
        epochs=2,
        steps_per_epoch=1,
    )
    trainer = Trainer(
        model=model,
        loss_function=criterion or nn.CrossEntropyLoss(),
        optimizers=[optimizer],
        schedulers=[scheduler],
        batch_size=4,
        epochs=2,
        batches=1,
        eval_batches=1,
        final_eval_batches=2,
        hybrid_calibration_batches=calibration_batches,
        save_model=True,
        save_directory=str(tmp_path),
    )
    return trainer, model


def _logits(classes, num_classes=4):
    classes = torch.as_tensor(classes, dtype=torch.long)
    values = torch.full((classes.shape[0], num_classes), -10.0)
    values.scatter_(1, classes[:, None], 10.0)
    return values


def test_paired_metrics_count_rescues_harms_and_standard_error():
    truth = torch.tensor([0, 1, 2, 3, 0])
    baseline = torch.tensor([0, 0, 2, 0, 1])
    # Rescue index 1, harm index 2, and change-but-still-wrong index 4.
    final_logits = _logits([0, 1, 0, 0, 2])

    metrics = paired_decoder_metrics(final_logits, truth, baseline)

    assert metrics.num_samples == 5
    assert metrics.rescued == 1
    assert metrics.harmed == 1
    assert metrics.corrections == 3
    assert metrics.hybrid_accuracy == metrics.baseline_accuracy
    assert metrics.net_gain == 0.0
    assert metrics.paired_standard_error == math.sqrt(2 / (5 * 4))


def test_eval_collects_exact_hybrid_tensors_without_calling_criterion(tmp_path):
    criterion = _CountingCriterion()
    trainer, model = _trainer(tmp_path, criterion=criterion)
    generator = _FixedGenerator([0, 1, 2, 3], [0, 1, 0, 3])

    _, (predictions, truth) = trainer._process_batches(
        generator,
        torch.device("cpu"),
        batches=1,
        train=False,
    )

    assert criterion.calls == 0
    torch.testing.assert_close(
        trainer._last_eval_baseline_classes,
        torch.tensor([0, 1, 2, 3]),
    )
    assert trainer._last_eval_residual_logits.shape == (4, 4)
    metrics = trainer._paired_metrics(predictions, truth)
    assert metrics is not None
    assert metrics.baseline_accuracy == 0.75

    metadata = trainer._calibrate_hybrid_gate(
        data_generator=generator,
        device=torch.device("cpu"),
        batches=1,
    )
    assert criterion.calls == 0
    assert model.calibration_calls == 1
    assert metadata["enabled"] is True
    assert metadata["margin"] == 0.25
    assert metadata["samples"] == 4


def test_zero_calibration_batches_disable_gate(tmp_path):
    trainer, model = _trainer(tmp_path, calibration_batches=0)
    model.gate_enabled.fill_(True)

    metadata = trainer._disable_hybrid_gate()

    assert not model.gate_enabled
    assert metadata["enabled"] is False
    assert metadata["samples"] == 0


def test_best_hybrid_checkpoint_uses_net_gain_before_accuracy(tmp_path):
    trainer, model = _trainer(tmp_path)
    truth = torch.tensor([0, 1, 2, 3])

    # Candidate A has lower raw accuracy but a larger gain over its baseline.
    candidate_a = paired_decoder_metrics(
        _logits([0, 1, 0, 0]),
        truth,
        torch.tensor([0, 0, 0, 0]),
    )
    with torch.no_grad():
        model.residual_bias.fill_(1.0)
    assert trainer._consider_best_checkpoint(
        epoch=3,
        accuracy=candidate_a.hybrid_accuracy,
        paired_metrics=candidate_a,
    )

    # Candidate B has perfect accuracy, but its already-perfect baseline means
    # zero paired gain and must not replace candidate A.
    candidate_b = paired_decoder_metrics(
        _logits([0, 1, 2, 3]),
        truth,
        truth,
    )
    with torch.no_grad():
        model.residual_bias.fill_(2.0)
    assert not trainer._consider_best_checkpoint(
        epoch=4,
        accuracy=candidate_b.hybrid_accuracy,
        paired_metrics=candidate_b,
    )

    checkpoint = torch.load(tmp_path / "best_model.pt", map_location="cpu")
    assert checkpoint["best_checkpoint"]["epoch"] == 3
    assert checkpoint["best_checkpoint"]["selection_metric"] == "net_gain"
    torch.testing.assert_close(
        checkpoint["model_state_dict"]["residual_bias"],
        torch.ones(4),
    )
    assert not (tmp_path / "best_model.pt.tmp").exists()


def test_selected_best_is_reevaluated_and_final_weights_are_restored(tmp_path):
    trainer, model = _trainer(tmp_path)
    truth = torch.tensor([0, 1, 2, 3])
    baseline = torch.tensor([0, 0, 0, 0])
    paired = paired_decoder_metrics(_logits([0, 1, 0, 0]), truth, baseline)

    with torch.no_grad():
        model.residual_bias.copy_(torch.tensor([4.0, 1.0, 0.0, 0.0]))
    trainer._consider_best_checkpoint(
        epoch=1,
        accuracy=paired.hybrid_accuracy,
        paired_metrics=paired,
    )
    with torch.no_grad():
        model.residual_bias.fill_(9.0)
    trainer.save_model(path=str(tmp_path), model_name="model", epoch=2)

    generator = _FixedGenerator([0, 1, 2, 3], [0, 1, 0, 3])
    trainer._evaluate_selected_best(generator, torch.device("cpu"))

    torch.testing.assert_close(model.residual_bias, torch.full((4,), 9.0))
    best = torch.load(tmp_path / "best_model.pt", map_location="cpu")
    final = torch.load(tmp_path / "model.pt", map_location="cpu")
    assert best["selected_best_evaluation"]["eval_samples"] == 8
    assert "paired_metrics" in best["selected_best_evaluation"]
    assert final["selected_best_evaluation"] == best["selected_best_evaluation"]
