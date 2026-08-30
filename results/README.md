# Result archive

Experiment directories keep their original `resdir_<id>` names and are grouped
by code family, noise model, and decoder:

```text
results/
├── toric/
│   ├── phenomenological/{cnn3d,convgru,pymatching}/
│   └── circuit/{convgru_mwpm,convgru_weighted_mwpm}/
├── bb/
│   └── code_capacity/depolarizing/orbit/{bb72,bb144}/
├── analysis/
├── plots/
└── local_smoke/
```

The current curated ConvGRU/PyMatching threshold plot can be reproduced from
its selected-point CSV:

```bash
python scripts/plot_threshold.py \
  results/plots/threshold_ConvGRU_PyMatching_L9_L11_L13_L15.csv \
  --out results/plots/threshold_ConvGRU_PyMatching_L9_L11_L13_L15.png \
  --title "Phenomenological Threshold: ConvGRU vs PyMatching"
```

Do not pass the entire archive when it contains learning-rate branches for the
same `(L,p)` point: the generic plotter treats those branches as replicate runs
and averages them. The curated CSV selects the intended branch explicitly.

Known partial runs are retained in `cnn3d/resdir_55562044`,
`cnn3d/resdir_55860108`, and `convgru/resdir_1252167/exp_6`.

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
Recreate the comparison figure with:

```bash
python scripts/plot_bb_results.py
```

The generated figure is
[`plots/bb_neural_bp_vs_vanilla_bp.png`](plots/bb_neural_bp_vs_vanilla_bp.png).

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
