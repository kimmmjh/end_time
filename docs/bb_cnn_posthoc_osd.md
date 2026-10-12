# Frozen CNN evaluation with syndrome OSD-0

`scripts/evaluate_bb_cnn_osd.py` evaluates existing Tanner CNN checkpoints with
**zero additional training steps**. It does not instantiate a trainer or an
optimizer and does not modify the source checkpoints, history, or shot banks.

## What is compared

All three arms use the same frozen best weights and the same saved final shots:

| Arm | Basis ordering | Nonbasis bits | Equation |
| --- | --- | --- | --- |
| Raw CNN | No OSD | CNN marginal hard decision | None |
| Existing residual OSD-0 | Increasing absolute marginal LLR | Preserve CNN hard decision | Solve residual syndrome and XOR the repair |
| Syndrome OSD-0 | Decreasing marginal error probability | Set to zero | Solve the original syndrome |

The syndrome arm implements the OSD-0 procedure in section V of
[Roffe et al., *Decoding Across the Quantum LDPC Code Landscape*](https://arxiv.org/pdf/2005.07016).
It selects independent columns in decreasing error probability, sets free bits
to zero, and solves over GF(2). Probability ties retain original column order.
This is a NumPy implementation of the published algorithm, not a bit-for-bit
claim about every `ldpc` version's pivoting or ties. There is no extra BP and no
higher-order candidate search. This is not an ASTRA reproduction.

Both OSD arms receive the same `P(X)+P(Y)` and `P(Z)+P(Y)` and repair **only
syndrome-invalid X/Z sectors**. Valid sectors remain unchanged, even if their
unknown logical class is incorrect. Holding the probabilities, hard decisions,
and fallback policy fixed isolates the OSD convention. In particular, the
ASTRA-specific probability ratios, Pauli argmax, and both-sector fallback are
not introduced into this comparison.

## Server launchers

Use root `run_bb_1.slurm` for the latest 44 models (four GPU shards), or
`run_bb_0.slurm` to extend BB144 depth 5 p=.04 to 800 epochs and then evaluate
standard OSD. [Submission commands and required existing files](bb_cnn_resume800_standard_osd.md).
`--auto-roots --latest-only --require-complete` selects the latest completed
result per exact configuration from both server-root and archived jobs.
`--shard-count`/`--shard-index` partition the sorted selected runs; each worker
writes its own summary index.

## Inputs and commands

Each evaluated run requires its original directory structure and three files:

- `model.pt`: latest checkpoint containing `best_model_state_dict`.
- `history.json`: final selected epoch and configuration.
- `final_shots.npz`: saved input syndrome, truth, corrections, and success flags.

The script uses **best weights**, not the latest training weights. Saved final
banks contain no soft probabilities, so one inference pass is required. No
training shots or `chunk_*.npz` files are needed. Configurations, graph hashes,
selected epochs, truth/syndrome consistency, and saved success flags are checked.
Incomplete runs without final banks are explicitly listed as skipped; no new
evaluation data is generated as a substitute.

From the repository root, evaluate the three jobs archived on October 12:

```bash
python scripts/evaluate_bb_cnn_osd.py \
  results/bb/code_capacity/depolarizing/tanner_cnn/continued_training \
  --output-root results/analysis/bb_cnn_standard_osd0_new_jobs
```

Or evaluate the archived 400-epoch campaign:

```bash
python scripts/evaluate_bb_cnn_osd.py \
  results/bb/code_capacity/depolarizing/tanner_cnn/400_epochs \
  --output-root results/analysis/bb_cnn_standard_osd0_400_epochs
```

Any result tree or individual run directory can be supplied. Multiple roots
are accepted and identical resolved run paths are deduplicated. Use
`--dry-run` to list available runs and missing files without inference. It does
not validate checkpoint contents. Use `--max-shots 256` with a separate output
root for a partial-bank smoke test; full-bank evaluation is the default.
CUDA is used automatically if available; `--device cpu` forces CPU. On the
local Mac, the project interpreter is
`/Users/justinkim/miniconda3/envs/nd/bin/python`. The Slurm environment's activated
`python` can run the same command on an allocated GPU node.
If the server still keeps jobs directly under `~/end_time`, pass
`resdir_59622483 resdir_59622488 resdir_59622490` as the input roots instead.

## Outputs and interpretation

Each run writes `report.json` and `paired_shots.npz` under the chosen output
root, retaining its job/date/run names. Existing run outputs are refused rather
than overwritten. `summary.json` indexes the runs completed by that invocation.

Reports contain input and implementation hashes, best epoch, actual shot count,
partial/full status, device, raw and both OSD metrics, paired rescues/harms and
standard error. Positive `standard_vs_residual.ler_reduction` means the new
syndrome convention has fewer failures on the same shots. The raw and residual
outputs are regenerated; mismatch counts against historical corrections and
success flags are recorded separately. Cross-device floating-point changes can
affect hard thresholds or ordering, so inspect these counts before treating a
historical number as exactly reproduced.

Timing is measured for this NumPy evaluation, not a benchmark against optimized
`ldpc` or ASTRA. No claim that syndrome OSD must outperform residual OSD is made.
These comparisons are post-hoc evaluations, not a new independent test bank for
selecting a model or tuning a decoder.
