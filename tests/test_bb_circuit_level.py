"""Circuit-level BB decoding: schedule, detector error model and BP2 decoder."""

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from models._equivariant_neural_bp2 import EquivariantNeuralBP2
from src._bb_circuit_loss import CircuitDegeneracyAwareLoss
from src._bb_circuit_metrics import OsdPostprocessor, score_corrections
from src._bb_circuit_trainer import BBCircuitTrainer, CircuitEvaluation
from src.bb_circuit_data import BBCircuitGenerator
from src.bb_code import BBCodeSpec
from src.bb_dem import build_bb_dem_graph
from src.bb_stim_utils import (
    CIRCUIT_SCHEMA_VERSION,
    DEFAULT_SCHEDULE,
    assert_detectors_deterministic,
    generate_bb_memory_circuit,
    search_schedules,
    validate_schedule,
)

NOISE = {"gate_error_rate": 3e-3, "measurement_error_rate": 3e-3}


@pytest.fixture(scope="module")
def small_graph():
    return build_bb_dem_graph("bb72", rounds=3, **NOISE)


def test_default_schedule_is_legal_and_has_depth_seven():
    assert validate_schedule(DEFAULT_SCHEDULE) == 7


def test_depth_six_is_impossible_and_depth_seven_is_not():
    # Six layers satisfy the rainbow constraints but never the determinism
    # condition, which is why a BB extraction cycle needs seven.
    assert search_schedules(6) == []
    assert search_schedules(7, limit=1)


@pytest.mark.parametrize(
    "violation",
    [
        {"x_left": (0, 0, 5)},  # one X ancilla driven twice in a layer
        {"z_right": (0, 1, 2)},  # collides with the left-block Z assignment
    ],
)
def test_illegal_schedules_are_rejected(violation):
    schedule = dict(DEFAULT_SCHEDULE)
    schedule.update(violation)
    with pytest.raises(ValueError):
        validate_schedule(schedule)


@pytest.mark.parametrize("name", ["bb72", "bb144"])
def test_circuit_reproduces_the_code_and_has_deterministic_detectors(name):
    code = BBCodeSpec.from_name(name)
    circuit = generate_bb_memory_circuit(code, rounds=3, **NOISE)
    # generate_bb_memory_circuit asserts the scheduled CNOT support equals
    # code.hx/code.hz internally, so reaching this point already proves it.
    assert circuit.num_detectors == (3 + 1) * code.num_checks
    assert circuit.num_observables == 2 * code.k
    assert_detectors_deterministic(circuit)


def test_determinism_condition_matches_stim():
    # This schedule satisfies every rainbow constraint, so only the derived
    # determinism condition rules it out. Stim must agree that it is broken,
    # which is what makes the condition trustworthy rather than merely
    # plausible.
    broken = {
        "x_left": (0, 1, 2),
        "x_right": (3, 4, 5),
        "z_left": (3, 4, 5),
        "z_right": (0, 1, 2),
    }
    with pytest.raises(ValueError, match="determinism"):
        validate_schedule(broken)

    circuit = generate_bb_memory_circuit(
        BBCodeSpec.from_name("bb72"),
        rounds=2,
        gate_error_rate=0.0,
        measurement_error_rate=0.0,
        schedule=broken,
        check_schedule=False,
    )
    with pytest.raises(ValueError, match="noiseless circuit"):
        assert_detectors_deterministic(circuit)


def test_dem_graph_is_consistent(small_graph):
    graph = small_graph
    assert graph.check_matrix.shape == (graph.num_detectors, graph.num_mechanisms)
    assert graph.observable_matrix.shape == (
        graph.num_observables,
        graph.num_mechanisms,
    )
    assert graph.edge_detector.size == graph.num_edges
    # Edges are the check matrix's nonzeros, in row-major order.
    assert graph.check_matrix.nnz == graph.num_edges
    assert np.all(np.diff(graph.edge_detector) >= 0)
    assert graph.priors.min() > 0.0 and graph.priors.max() < 1.0
    # Circuit-level priors vary by orders of magnitude, unlike code capacity.
    assert np.ptp(graph.prior_log_odds) > 1.0


def test_orbit_count_is_independent_of_code_size_and_rounds():
    # Orbits describe the local space-time structure of one extraction cycle,
    # so refining the lattice or adding bulk rounds must not create new ones.
    # This is the property that keeps the decoder's parameter count fixed as
    # the code and the experiment grow.
    baseline = build_bb_dem_graph("bb72", rounds=4, **NOISE).num_orbits
    assert build_bb_dem_graph("bb72", rounds=6, **NOISE).num_orbits == baseline
    assert build_bb_dem_graph("bb144", rounds=4, **NOISE).num_orbits == baseline


def test_orbit_structure_is_degenerate_without_bulk_rounds():
    # rounds counts noisy cycles and there is one extra closing frame. With
    # boundary_width=1, two noisy rounds still have no genuine bulk frame.
    assert (
        build_bb_dem_graph("bb72", rounds=2, **NOISE).num_orbits
        < build_bb_dem_graph("bb72", rounds=3, **NOISE).num_orbits
    )


def test_one_noisy_round_has_a_separate_closing_frame_and_real_faults():
    code = BBCodeSpec.from_name("bb72")
    circuit = generate_bb_memory_circuit(code, rounds=1, **NOISE)
    assert circuit.num_detectors == 2 * code.num_checks
    assert circuit.detector_error_model(decompose_errors=False).num_errors > 0

    # Only the requested noisy cycle contains two-qubit depolarizing channels;
    # the reference and closing cycles are exact boundary measurements.
    depolarize2 = sum(
        instruction.name == "DEPOLARIZE2" for instruction in circuit
    )
    assert depolarize2 == 7

    graph = build_bb_dem_graph("bb72", rounds=1, **NOISE, circuit=circuit)
    assert graph.circuit_schema_version == CIRCUIT_SCHEMA_VERSION
    assert graph.rounds == 1
    assert graph.detector_frames == 2
    assert graph.num_mechanisms > 0
    times, counts = np.unique(graph.detector_coordinates[:, 3], return_counts=True)
    assert np.array_equal(times, [0, 1])
    assert np.array_equal(counts, [code.num_checks, code.num_checks])


@pytest.mark.parametrize(
    "keyword",
    [
        {"gate_error_rate": 0.750001, "idle_error_rate": 0.0},
        {"gate_error_rate": 0.0, "idle_error_rate": 0.750001},
    ],
)
def test_one_qubit_depolarizing_rates_reject_overmixing(keyword):
    with pytest.raises(ValueError, match="at most 0.75"):
        generate_bb_memory_circuit(
            "bb72",
            rounds=1,
            measurement_error_rate=0.0,
            **keyword,
        )


def test_reused_graph_must_match_the_exact_circuit_noise():
    graph = build_bb_dem_graph("bb72", rounds=1, **NOISE)
    with pytest.raises(ValueError, match="incompatible"):
        BBCircuitGenerator(
            "bb72",
            rounds=1,
            batch_size=1,
            seed=10,
            graph=graph,
            gate_error_rate=NOISE["gate_error_rate"] * 2,
            measurement_error_rate=NOISE["measurement_error_rate"],
        )


def test_orbits_are_invariant_under_cyclic_translation(small_graph):
    graph = small_graph
    code = BBCodeSpec.from_name("bb72")
    coordinates = graph.detector_coordinates
    lookup = {tuple(row): index for index, row in enumerate(coordinates.tolist())}

    shift_i, shift_j = 2, 3
    permutation = np.empty(graph.num_detectors, dtype=np.int64)
    for detector, (check_type, i, j, time) in enumerate(coordinates.tolist()):
        permutation[detector] = lookup[
            (check_type, (i + shift_i) % code.ell, (j + shift_j) % code.m, time)
        ]

    csc = graph.check_matrix.tocsc()
    orbit_of = {}
    for mechanism in range(graph.num_mechanisms):
        detectors = tuple(
            sorted(int(d) for d in csc.indices[csc.indptr[mechanism] : csc.indptr[mechanism + 1]])
        )
        edges = np.flatnonzero(graph.edge_mechanism == mechanism)
        orbit_of[detectors] = tuple(sorted(graph.edge_orbit[edges].tolist()))

    translated_hits = 0
    for detectors, orbits in orbit_of.items():
        moved = tuple(sorted(int(permutation[d]) for d in detectors))
        if moved in orbit_of:
            translated_hits += 1
            assert orbit_of[moved] == orbits
    # The translation must map the mechanism set onto itself.
    assert translated_hits == len(orbit_of)


def test_untrained_model_is_bitwise_vanilla_min_sum(small_graph):
    torch.manual_seed(0)
    model = EquivariantNeuralBP2(
        small_graph, iterations=4, hidden_dim=16, orbit_embedding_dim=4
    ).eval()
    syndrome = (
        torch.rand(6, small_graph.num_detectors) < 0.05
    ).to(torch.float32)
    with torch.no_grad():
        neural = model(syndrome, neural=True)
        vanilla = model(syndrome, neural=False)
    assert torch.equal(neural, vanilla)


@pytest.mark.parametrize("sharing", ["orbit", "global", "edge"])
def test_every_sharing_mode_runs_and_scales_by_embedding_only(sharing, small_graph):
    # Unlike the code-capacity decoder, per-edge sharing costs one embedding
    # row rather than one MLP, so it is actually runnable here.
    model = EquivariantNeuralBP2(
        small_graph, iterations=2, hidden_dim=8, orbit_embedding_dim=2, sharing=sharing
    ).eval()
    syndrome = torch.zeros(2, small_graph.num_detectors)
    with torch.no_grad():
        posterior = model(syndrome, neural=True)
    assert posterior.shape == (2, small_graph.num_mechanisms)
    assert torch.isfinite(posterior).all()


def test_loss_is_degenerate_aware_and_has_gradients(small_graph):
    torch.manual_seed(1)
    model = EquivariantNeuralBP2(
        small_graph, iterations=3, hidden_dim=16, orbit_embedding_dim=4
    )
    generator = BBCircuitGenerator(
        "bb72", rounds=3, batch_size=4, seed=2, graph=small_graph, **NOISE
    )
    criterion = CircuitDegeneracyAwareLoss(
        check_matrix=small_graph.check_matrix,
        observable_matrix=small_graph.observable_matrix,
    )
    batch = generator.sample_dem()
    posterior, history = model(batch.detectors, neural=True, return_all=True)
    output = criterion(posterior, batch.detectors, batch.mechanisms, history)
    assert torch.isfinite(output.total)
    output.total.backward()
    assert any(
        parameter.grad is not None and torch.any(parameter.grad != 0)
        for parameter in model.parameters()
    )


def test_parity_loss_has_finite_gradients_at_zero_llr():
    # tanh(0)=0 used to reach log(0): the forward value looked finite but every
    # gradient was NaN.  Exercise that exact uncertain-posterior boundary.
    criterion = CircuitDegeneracyAwareLoss(
        check_matrix=sp.csr_matrix([[1, 1]], dtype=np.uint8),
        observable_matrix=sp.csr_matrix([[1, 0]], dtype=np.uint8),
    )
    posterior = torch.zeros((1, 2), requires_grad=True)
    output = criterion(
        posterior,
        torch.zeros((1, 1)),
        torch.zeros((1, 2)),
    )
    assert torch.isfinite(output.total)
    output.total.backward()
    assert posterior.grad is not None
    assert torch.isfinite(posterior.grad).all()


def test_generator_exposes_fired_mechanisms_that_explain_the_syndrome(small_graph):
    generator = BBCircuitGenerator(
        "bb72", rounds=3, batch_size=8, seed=3, graph=small_graph, **NOISE
    )
    batch = generator.sample_dem()
    mechanisms = batch.mechanisms.numpy().astype(np.uint8)
    detectors = batch.detectors.numpy().astype(np.uint8)
    check = small_graph.check_matrix.toarray().astype(np.uint8)
    assert np.array_equal((mechanisms @ check.T) % 2, detectors)

    observables = batch.observables.numpy().astype(np.uint8)
    observable = small_graph.observable_matrix.toarray().astype(np.uint8)
    assert np.array_equal((mechanisms @ observable.T) % 2, observables)

    # Circuit shots must not leak DEM labels.
    assert generator.sample_circuit(4).mechanisms is None


def test_osd_postprocessing_beats_plain_bp_and_scores_exactly(small_graph):
    generator = BBCircuitGenerator(
        "bb72", rounds=3, batch_size=32, seed=4, graph=small_graph, **NOISE
    )
    batch = generator.sample_circuit()
    detectors = batch.detectors.numpy().astype(np.uint8)
    observables = batch.observables.numpy().astype(np.uint8)
    model = EquivariantNeuralBP2(
        small_graph, iterations=6, hidden_dim=8, orbit_embedding_dim=2
    ).eval()
    with torch.no_grad():
        posterior = model(batch.detectors, neural=False)

    scoring = {
        "check_matrix": small_graph.check_matrix,
        "observable_matrix": small_graph.observable_matrix,
    }
    plain = score_corrections(
        EquivariantNeuralBP2.hard_decision(posterior).numpy(),
        detectors=detectors,
        observables=observables,
        **scoring,
    )
    osd = OsdPostprocessor(
        small_graph.check_matrix, priors=small_graph.priors, method="OSD_0", order=0
    )
    corrected = score_corrections(
        osd.decode_batch(detectors, posterior=posterior.numpy()),
        detectors=detectors,
        observables=observables,
        **scoring,
    )
    # OSD always returns a syndrome-satisfying correction; plain BP does not.
    assert corrected.syndrome_converged.all()
    assert corrected.accuracy >= plain.accuracy
    assert np.isclose(
        plain.flagged_failure + plain.unflagged_failure, plain.logical_error_rate
    )


def test_correction_success_requires_syndrome_and_logical_match():
    check = sp.csr_matrix([[1]], dtype=np.uint8)
    observable = sp.csr_matrix([[1]], dtype=np.uint8)
    correction = np.zeros((4, 1), dtype=np.uint8)
    outcomes = score_corrections(
        correction,
        detectors=np.asarray([[0], [0], [1], [1]], dtype=np.uint8),
        observables=np.asarray([[0], [1], [0], [1]], dtype=np.uint8),
        check_matrix=check,
        observable_matrix=observable,
    )
    assert np.array_equal(outcomes.syndrome_converged, [True, True, False, False])
    assert np.array_equal(outcomes.success, [True, False, False, False])
    assert outcomes.flagged_failure == 0.5
    assert outcomes.unflagged_failure == 0.25
    assert np.isclose(
        outcomes.flagged_failure
        + outcomes.unflagged_failure,
        outcomes.logical_error_rate,
    )


def _dummy_circuit_evaluation(
    *,
    neural_accuracy: float = 0.5,
    paired_gain: float = 0.0,
    neural_osd_accuracy: float | None = None,
    osd_paired_gain: float | None = None,
) -> CircuitEvaluation:
    return CircuitEvaluation(
        shots=4,
        neural_accuracy=neural_accuracy,
        vanilla_accuracy=neural_accuracy - paired_gain,
        neural_converged=1.0,
        vanilla_converged=1.0,
        neural_flagged=0.0,
        neural_unflagged=1.0 - neural_accuracy,
        paired_gain=paired_gain,
        paired_gain_error=0.0,
        rescued=0,
        harmed=0,
        osd_shots=4 if neural_osd_accuracy is not None else 0,
        neural_osd_accuracy=neural_osd_accuracy,
        vanilla_osd_accuracy=(
            None
            if neural_osd_accuracy is None or osd_paired_gain is None
            else neural_osd_accuracy - osd_paired_gain
        ),
        osd_paired_gain=osd_paired_gain,
        osd_paired_gain_error=(0.0 if osd_paired_gain is not None else None),
    )


def _dummy_experiment_config() -> dict[str, object]:
    return {
        "architecture": "bb_neural_bp_circuit",
        "circuit_schema_version": CIRCUIT_SCHEMA_VERSION,
        "code": "bb72",
        "graph_fingerprint": "test-graph",
        "rounds": 1,
        "detector_frames": 2,
        "gate_error_rate": 0.003,
        "measurement_error_rate": 0.003,
        "idle_error_rate": 0.0,
        "num_detectors": 1,
        "num_mechanisms": 1,
        "num_edges": 1,
        "num_orbits": 1,
        "bp_iterations": 1,
        "bp_residual_hidden_dim": 1,
        "bp_orbit_embedding_dim": 0,
        "bp_parameter_sharing": "global",
        "bp_normalisation": 0.625,
        "bp_residual_scale": 1.0,
        "bp_max_relaxation_delta": 0.5,
        "bp_deep_supervision_weight": 0.0,
        "bb_syndrome_loss_weight": 1.0,
        "bb_logical_loss_weight": 1.0,
        "bb_pauli_loss_weight": 0.1,
        "bb_weight_decay": 0.0,
        "checkpoint_selection_metric": "neural_paired_gain",
        "bb_osd_method": "OSD_CS",
        "bb_osd_order": 7,
        "seed": 7,
    }


def _dummy_trainer(tmp_path, *, model, load_model_path=None):
    return BBCircuitTrainer(
        model=model,
        generator=object(),
        eval_generator=object(),
        criterion=torch.nn.Identity(),
        device=torch.device("cpu"),
        epochs=1,
        batches=2,
        eval_batches=1,
        eval_every=1,
        final_eval_batches=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        gradient_clip=1.0,
        output_directory=str(tmp_path),
        experiment_config=_dummy_experiment_config(),
        save_model=True,
        load_model_path=load_model_path,
    )


def _install_dummy_training(trainer: BBCircuitTrainer) -> None:
    parameter = next(trainer.model.parameters())

    def train_epoch():
        value = 0.0
        for _ in range(trainer.batches):
            trainer.optimizer.zero_grad(set_to_none=True)
            loss = (parameter - 1.0).square().sum()
            loss.backward()
            trainer.optimizer.step()
            trainer.scheduler.step()
            value += float(loss.detach())
        average = value / trainer.batches
        return {
            "total": average,
            "syndrome": average,
            "logical": 0.0,
            "mechanism": 0.0,
        }

    trainer._train_epoch = train_epoch  # type: ignore[method-assign]
    trainer.evaluate = lambda batches: _dummy_circuit_evaluation(  # type: ignore[method-assign]
        neural_accuracy=0.75, paired_gain=0.25
    )


def test_circuit_checkpoint_resume_restores_epoch_history_and_optimizer(tmp_path):
    first = _dummy_trainer(
        tmp_path / "first", model=torch.nn.Linear(1, 1, bias=False)
    )
    _install_dummy_training(first)
    first.train()
    checkpoint_path = tmp_path / "first" / "model.pt"
    assert checkpoint_path.is_file()

    resumed = _dummy_trainer(
        tmp_path / "resumed",
        model=torch.nn.Linear(1, 1, bias=False),
        load_model_path=checkpoint_path,
    )
    assert resumed.start_epoch == 1
    assert [entry["epoch"] for entry in resumed.history["train"]] == [0]
    assert resumed.optimizer.state
    _install_dummy_training(resumed)
    resumed.train()

    best_path = tmp_path / "resumed" / "best_model.pt"
    assert best_path.is_file()
    selected = torch.load(best_path, map_location="cpu", weights_only=False)
    assert selected["checkpoint_role"] == "selected_best"
    assert selected["epoch"] == 0

    saved = torch.load(
        tmp_path / "resumed" / "model.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert saved["epoch"] == 1
    assert [entry["epoch"] for entry in saved["history"]["train"]] == [0, 1]


def test_osd_enabled_checkpoint_selection_uses_paired_osd_gain(tmp_path):
    trainer = _dummy_trainer(
        tmp_path / "selection", model=torch.nn.Linear(1, 1, bias=False)
    )
    trainer._osd = object()  # selection behavior only; no decoding in this test.
    trainer._consider_best(
        0,
        _dummy_circuit_evaluation(
            neural_accuracy=0.9,
            paired_gain=0.2,
            neural_osd_accuracy=0.70,
            osd_paired_gain=0.01,
        ),
    )
    trainer._consider_best(
        1,
        _dummy_circuit_evaluation(
            neural_accuracy=0.8,
            paired_gain=0.1,
            neural_osd_accuracy=0.71,
            osd_paired_gain=0.02,
        ),
    )
    assert trainer.best_epoch == 1
    assert trainer.best_selection_metric == "neural_osd_paired_gain"
