# Result archive

Experiment directories keep their original `resdir_<id>` names and are grouped
by code family, noise model, and decoder:

```text
results/
├── toric/
│   ├── phenomenological/{cnn3d,convgru,pymatching}/
│   └── circuit/{convgru_mwpm,convgru_weighted_mwpm}/
├── bb/
│   ├── code_capacity/depolarizing/orbit/{bb72,bb144}/
│   └── circuit/{q_equals_p_idle0,no_osd,noise_balance,neural_relay}/
├── analysis/
├── plots/neural_relay/
└── local_smoke/
```

## Curated figures

Headline results:

- [`plots/threshold_ConvGRU_PyMatching_L9_L11_L13_L15.png`](plots/threshold_ConvGRU_PyMatching_L9_L11_L13_L15.png): toric phenomenological threshold study
- [`plots/bb_campaign_2026_08_decoders.png`](plots/bb_campaign_2026_08_decoders.png): BB code-capacity decoder comparison
- [`plots/bb_circuit_campaign_2026_08.png`](plots/bb_circuit_campaign_2026_08.png): BB circuit-level Neural+OSD reference sweep

Supporting and diagnostic figures:

- [`neural_relay/overview.png`](plots/neural_relay/overview.png): BB72/BB144 final LER, paired gain, and historical BP/OSD references
- [`neural_relay/training_bb72.png`](plots/neural_relay/training_bb72.png): all eight BB72 training losses and paired validation gains
- [`neural_relay/training_bb144.png`](plots/neural_relay/training_bb144.png): all eight BB144 training losses and paired validation gains
- [`plots/bb_campaign_2026_08_ablations.png`](plots/bb_campaign_2026_08_ablations.png): code-capacity ablations
- [`plots/bb_circuit_campaign_2026_08_ablations.png`](plots/bb_circuit_campaign_2026_08_ablations.png): circuit-level ablations
- [`plots/bb_circuit_raw_vs_osd_2026_09.png`](plots/bb_circuit_raw_vs_osd_2026_09.png): raw-versus-OSD diagnostic; the two pipelines use separately selected checkpoints

Neural Relay figures are consolidated into these **three PNGs** in
[`plots/neural_relay/`](plots/neural_relay/). Older overlapping figures and their
PDF copies were removed. Regeneration writes PNG only. Failure-type and
iteration-count details remain in the analysis reports and CSVs.

The current curated ConvGRU/PyMatching threshold plot can be reproduced from
its selected-point CSV:

```bash
python scripts/plot_threshold.py \
  results/analysis/threshold_ConvGRU_PyMatching_L9_L11_L13_L15.csv \
  --out results/plots/threshold_ConvGRU_PyMatching_L9_L11_L13_L15.png \
  --title "Phenomenological Threshold: ConvGRU vs PyMatching"
```

Do not pass the entire archive when it contains learning-rate branches for the
same `(L,p)` point: the generic plotter treats those branches as replicate runs
and averages them. The curated CSV selects the intended branch explicitly.

Known partial runs are retained in `cnn3d/resdir_55562044`,
`cnn3d/resdir_55860108`, and `convgru/resdir_1252167/exp_6`.

## September 23, 2026: ordinary BP evaluation scripts

`run_bb_0.slurm` through `run_bb_4.slurm` now launch a
[standalone BP sweep](../docs/bb_plain_bp_campaign.md), without training, Relay,
or OSD. Each experiment compares max-12 and max-1000 BP on 4096 saved circuit
shots. Outputs are `resdir_<jobid>/exp_<index>/{results.json,summary.csv,shots.npz,outcomes.npz}`.
New server measurements have not yet been imported here.

## September 22, 2026: BP baseline stopping and iteration caps

New BB circuit evaluations report ordinary BP with caps 12 and 1000 by default,
on the same circuit samples. Each shot stops at its first syndrome-valid
correction. The short cap follows `--bp_iterations`; the additional cap is
`--bb_bp_reference_iterations`. Both references, including LER and mean actual
iterations, are saved in each evaluation's `bp_baselines` fields in
`history.json`. Relay runs retain their paired non-neural Relay comparison.
Non-Relay neural evaluation stops within its trained depth, while training
continues full unrolling. OSD, when enabled, processes only unconverged shots.

Archived results and plots have not been recalculated. New histories record
`bp_evaluation_policy=first_syndrome_valid_v1`; old checkpoint-selection histories
cannot be resumed under the changed policy. These baseline changes preserve
the configured min-sum scaling and noise model, so they do not establish
literature-matched LERs. See the [decoder documentation](../docs/bb_neural_relay_bp.md).

## September 17, 2026: non-relay BP clipping fix

`EquivariantNeuralBP2` now subtracts the recipient check message before clipping
the outgoing variable message. This corrects saturated messages and their
gradients in both the plain and neural non-relay paths. Regression tests cover
positive/negative saturation, degree-one variables, and extrinsic gradients.
The September 16 measurements remain results of the old implementation; the
audit script explicitly replays that old update when generating archived rows.
Re-evaluating an old non-relay checkpoint with the current code uses the fixed
update and can produce different outputs.

## September 16, 2026: circuit baseline implementation audit

The [implementation audit](analysis/bb_circuit_baseline_audit_2026_09_16.md)
distinguishes the archived circuit baselines from literature reproductions.
The archived OSD wrapper reseeds an additional BP pass with the supplied
posterior; it does not hand that posterior directly to OSD. The older non-relay
variable update also clips before removing the recipient check message. The
Relay implementation already removes the recipient before clipping.
Noise, logical observables, OSD order, and iteration budgets differ from the
papers. Existing runs and plots retain their original values and describe
comparisons within those settings, not a matched literature benchmark.

The [audit script](../scripts/audit_bb_circuit_baseline.py) saves a shared circuit
shot bank, decoder outcomes, numerical counterexamples, and direct `ldpc`
BP-OSD reference measurements. It leaves the training decoder unchanged.

## September 15, 2026: BB144 Neural Relay results

The newly imported `resdir_58164812` and `resdir_58164813` (jobs 2 and 3)
are archived with their original names under
[`bb/circuit/neural_relay/q_equals_p_idle0/orbit/bb144/`](bb/circuit/neural_relay/q_equals_p_idle0/orbit/bb144/).
All eight runs completed 100 epochs and a fresh 4,096-shot selected-best
evaluation, without OSD. The 102 original files, including checkpoints, were
verified by SHA-256 before and after moving them into the archive.

The [BB144 report](analysis/bb_neural_relay_bb144_2026_09_15.md) includes the
[final comparison CSV](analysis/bb_neural_relay_bb144_2026_09_15.csv),
[training records](analysis/bb_neural_relay_bb144_2026_09_15_train.csv),
[validation records](analysis/bb_neural_relay_bb144_2026_09_15_validation.csv), and
[original-file manifest](analysis/bb_neural_relay_bb144_2026_09_15_manifest.csv).
Only p=0.005 shows a significant improvement, from 98.54% to 93.14% LER.
p=0.001/0.004 degrade after Holm correction across the eight BB144 tests.
Across all 16 BB72+BB144 tests, the p=0.004 degradation remains significant,
while p=0.001 does not. All observed BB144 Neural failures are syndrome invalid;
seven of eight runs end with higher average training loss than they started.

The [combined CSV](analysis/bb_neural_relay_bb72_bb144_2026_09_15.csv) and
[comparison figure](plots/neural_relay/overview.png) retain
both code-specific settings and multiple-testing scopes. Incomplete historical
BB144 high-p OSD runs remain marked partial and have no final LER substituted.
Neural Relay figures are saved as PNG only.

Reproduce numerical extraction, audits, file manifests, and figures for both codes:

```bash
python scripts/summarize_bb_neural_relay.py --code all
```

Use `--code bb144` for only the new results. Markdown interpretation is maintained
in the report rather than automatically overwritten by the script.

## September 13, 2026: first Neural Relay results

`resdir_58164810` and `resdir_58164811` contain the completed BB72 low/high
sweeps from jobs 0 and 1. Both are archived under
[`bb/circuit/neural_relay/q_equals_p_idle0/orbit/bb72/`](bb/circuit/neural_relay/q_equals_p_idle0/orbit/bb72/).
All eight points have 100 epochs and 4,096 fresh selected-best shots.
On September 15, the 102 original files were moved from the repository root
and verified against the [SHA-256 manifest](analysis/bb_neural_relay_bb72_2026_09_13_manifest.csv).
Report and CSV source paths point to the archive. BB144 Relay results were
added separately on September 15, as described above.

The [analysis](analysis/bb_neural_relay_bb72_2026_09_13.md),
[final comparison CSV](analysis/bb_neural_relay_bb72_2026_09_13.csv), and
[overview figure](plots/neural_relay/overview.png) distinguish
non-neural Relay from the historical single-pass BP baseline. Learning improves
LER at p=0.004/0.005/0.008 after exact paired tests with Holm correction.
The historical Neural+OSD pipeline still has lower LER; later training degrades
several Relay models. This is one training seed per point, not a threshold study.

Regenerate and audit the numerical extraction and figures with:

```bash
python scripts/summarize_bb_neural_relay.py --code bb72
```

## Recent ConvGRU resume sweep

`toric/phenomenological/convgru/resdir_1084621` and `resdir_1085072`
contain the completed learning-rate sweep for resumed pure-ConvGRU models.
The extracted comparison, including source checkpoints and PyMatching
baselines, is in `analysis/convgru_resume_lr_sweep.csv`; conclusions are in
`analysis/convgru_resume_lr_sweep.md`.

## Recent pure ConvGRU threshold-gap sweep

`toric/phenomenological/convgru/resdir_1116847` and `resdir_1117203`
contain eight completed L13/L15 runs filling the narrow p-grid from 0.011 to
0.019. Extracted settings and metrics are in
[`analysis/convgru_threshold_gap_2026_08.csv`](analysis/convgru_threshold_gap_2026_08.csv),
with interpretation in
[`analysis/convgru_threshold_gap_2026_08.md`](analysis/convgru_threshold_gap_2026_08.md).

Three runs stopped near a cross-entropy loss of `ln(4)` and are classified as
optimization plateaus. They remain in the curated threshold CSV and plot, but
are shown as `X` markers rather than connected threshold-curve points.

## BB Neural BP depolarizing sweep

The completed orbit-shared Neural BP sweeps are archived as
`bb/code_capacity/depolarizing/orbit/bb72/resdir_57181711` and
`bb/code_capacity/depolarizing/orbit/bb144/resdir_57181713`. Together they
cover `p=0.04,0.06,0.08,0.10`; every point has 300 epochs and a fresh
131,072-shot selected-best evaluation against vanilla BP4 on the same samples.

The extracted data and interpretation are in
[`analysis/bb_neural_bp_depolarizing_orbit.csv`](analysis/bb_neural_bp_depolarizing_orbit.csv)
and
[`analysis/bb_neural_bp_depolarizing_orbit.md`](analysis/bb_neural_bp_depolarizing_orbit.md).
Its early BP-only figure has been retired because the newer same-bank decoder
comparison below contains the Neural/vanilla curves together with the stronger
OSD and LSD baselines.

## August 2026 BB campaign

The ten `run_bb_0`--`run_bb_9` jobs are archived by purpose under:

```text
bb/code_capacity/
├── depolarizing/
│   ├── orbit/replicates/
│   ├── ablations/{sharing,iterations,mechanism,loss_auxiliary,loss_core}/
│   └── classical/css_separated/{bb72,bb144}/
└── independent_xz/marginal_matched/orbit/
```

Completion must be determined from each output's `training_log.txt` and
`history.json`, not only the top-level srun exit code. Of 32 Neural BP runs, 28
have a fresh 131,072-shot `[Selected Best]` evaluation and four are genuinely
partial. All eight classical `(code,p)` points completed.

The complete extraction and interpretation are in
[`analysis/bb_campaign_2026_08.md`](analysis/bb_campaign_2026_08.md). Its
machine-readable inputs are
[`analysis/bb_campaign_2026_08_neural.csv`](analysis/bb_campaign_2026_08_neural.csv)
and
[`analysis/bb_campaign_2026_08_classical.csv`](analysis/bb_campaign_2026_08_classical.csv).
The selected-best Neural/BP4 checkpoints are also evaluated on the exact saved
classical error banks in
[`analysis/bb_neural_vs_classical_paired.csv`](analysis/bb_neural_vs_classical_paired.csv).
Rebuild the summaries and plots with:

```bash
python scripts/summarize_bb_campaign.py
```

The paired same-bank CSV itself is reproduced with:

```bash
python scripts/bb_neural_vs_classical_paired.py
```

The main figures are
[`plots/bb_campaign_2026_08_ablations.png`](plots/bb_campaign_2026_08_ablations.png)
and
[`plots/bb_campaign_2026_08_decoders.png`](plots/bb_campaign_2026_08_decoders.png).

## August–September 2026 BB circuit-level campaign

The corrected circuit-schema-v2 campaign is archived by purpose:

```text
bb/circuit/
├── q_equals_p_idle0/
│   ├── orbit/{bb72,bb144,replicates}/
│   └── ablations/{sharing,iterations,mechanism,loss_auxiliary}/
├── no_osd/q_equals_p_idle0/orbit/{bb72,bb144}/
└── noise_balance/orbit/
```

The archive contains 40 histories: 35 completed all 100 epochs plus a fresh
4,096-shot selected-best OSD-0 evaluation, while five BB144 runs reached the
24-hour Slurm limit and are retained as resumable partial runs. The primary
reference sweep contains eight complete BB72 points (`p=0.001` through
`0.010`) and four complete BB144 points (`p=0.001` through `0.004`). The four
higher-p BB144 reference points are partial.

The new campaign also contains three training seeds per code at `p=0.004`,
global-versus-orbit sharing, T=6/12/24, learned-residual and relaxation
component ablations, auxiliary-loss ablations, and `idle=p`/`q=2p` noise
controls. The headline comparison is Neural-BP posterior + OSD-0 versus
vanilla-BP posterior + the same OSD-0 post-processor; plain BP remains a
convergence diagnostic.

The complete extraction and interpretation are in
[`analysis/bb_circuit_campaign_2026_08.csv`](analysis/bb_circuit_campaign_2026_08.csv)
and
[`analysis/bb_circuit_campaign_2026_08.md`](analysis/bb_circuit_campaign_2026_08.md).
Interrupted-run provenance and resume information are in
[`analysis/bb_circuit_campaign_2026_08_partial.csv`](analysis/bb_circuit_campaign_2026_08_partial.csv).
Rebuild the summaries and plots with:

```bash
python scripts/summarize_bb_circuit_campaign.py
```

The figures are
[`plots/bb_circuit_campaign_2026_08.png`](plots/bb_circuit_campaign_2026_08.png)
and
[`plots/bb_circuit_campaign_2026_08_ablations.png`](plots/bb_circuit_campaign_2026_08_ablations.png).
This is still not a threshold estimate: BB144 high-p completion, more training
seeds, a larger fixed test bank, and stronger OSD-CS/LSD comparisons remain
necessary.

## September 2026 BB circuit-level no-OSD diagnostic

The two raw-decoder jobs are archived as
`bb/circuit/no_osd/q_equals_p_idle0/orbit/bb72/resdir_57890809` and
`bb/circuit/no_osd/q_equals_p_idle0/orbit/bb144/resdir_57890812`. All eight
points (`p=0.001` through `0.004` for each code) completed 100 epochs and a
fresh 4,096-shot selected-best evaluation. They use raw Neural BP2 versus raw
vanilla BP2 on paired Stim samples; no OSD evaluation or repair is present.

The extracted data and interpretation are in
[`analysis/bb_circuit_no_osd_2026_09.csv`](analysis/bb_circuit_no_osd_2026_09.csv)
and
[`analysis/bb_circuit_no_osd_2026_09.md`](analysis/bb_circuit_no_osd_2026_09.md).
The empty partial-run inventory is
[`analysis/bb_circuit_no_osd_2026_09_partial.csv`](analysis/bb_circuit_no_osd_2026_09_partial.csv).
Rebuild the raw summary tables with:

```bash
python scripts/summarize_bb_circuit_no_osd.py
```

The separate raw-only figure has been retired because the combined figure below
contains the same raw failure and paired-gain curves. These points are a
convergence diagnostic, not a threshold curve: raw success is almost identical
to syndrome convergence, and BB144 collapses for `p>=0.002` without a global
repair stage.

The common `p=0.001`--`0.004` raw and OSD-assisted results are overlaid in
[`plots/bb_circuit_raw_vs_osd_2026_09.png`](plots/bb_circuit_raw_vs_osd_2026_09.png),
with the merged values in
[`analysis/bb_circuit_raw_vs_osd_2026_09.csv`](analysis/bb_circuit_raw_vs_osd_2026_09.csv).
Recreate both with:

```bash
python scripts/plot_bb_circuit_raw_vs_osd.py
```

This overlay is descriptive rather than a clean OSD ablation. The raw campaign
selects checkpoints by raw paired gain, while the older OSD campaign selects
them by OSD paired gain. A causal before/after comparison requires applying OSD
to the same frozen checkpoint on the same saved test bank.
