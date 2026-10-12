# Tanner CNN: depth-5 recovery and selected 400→800 continuation

**Campaign completed:** jobs `59622483`, `59622488`, and `59622490` completed all
12 models: 11 at 800 epochs and the recovery at 400. Original files are archived
under `results/bb/code_capacity/depolarizing/tanner_cnn/continued_training/`.
See the [October 12 analysis](../results/analysis/bb_update_2026_10_12.md) for
paired gains and checkpoint-selection results. The commands below document the
completed campaign; resubmitting them repeats these continuations.

The historical launchers are archived in
[`scripts/slurm_archive/bb_depth45_resume800/`](../scripts/slurm_archive/bb_depth45_resume800/). They launched
12 existing models on three four-GPU allocations; none starts from scratch.
Prior numbered launchers 0–8 are retained verbatim in
[`scripts/slurm_archive/bb_capacity400/`](../scripts/slurm_archive/bb_capacity400/).

Current root scripts now implement the
[BB144 p=.04 continuation and standard OSD evaluation](bb_cnn_resume800_standard_osd.md).

## Why these runs

The [400-epoch analysis](../results/analysis/bb_update_2026_10_10.md) shows the
strongest completed results at depth 5. We prioritize those models, recover the
missing BB144 p=.04 point, and extend depth 4 as a comparison. The late curves
are relatively flat, so further gains are an experiment, not a promise.
Depth 2/3 and p=.02 are omitted from this continuation: higher-p depth 4/5 is
more useful for distinguishing improvements with the existing final shot budget.
This is additional training of the same seeds, not a seed-reproducibility study.

| Script | Exp | Code | Depth | p | Seed | Source job | Completed → target |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 0 | 0 | BB144 | 5 | .04 | 14401040 | 59122450 | **114 → 400 recovery** |
| 0 | 1 | BB144 | 5 | .06 | 14401060 | 59122450 | 400 → 800 |
| 0 | 2 | BB144 | 5 | .08 | 14401080 | 59122450 | 400 → 800 |
| 0 | 3 | BB72 | 5 | .08 | 7201080 | 59122449 | 400 → 800 |
| 1 | 0 | BB72 | 5 | .04 | 7201040 | 59122449 | 400 → 800 |
| 1 | 1 | BB72 | 5 | .06 | 7201060 | 59122449 | 400 → 800 |
| 1 | 2 | BB144 | 4 | .04 | 14401040 | 59122448 | 400 → 800 |
| 1 | 3 | BB144 | 4 | .06 | 14401060 | 59122448 | 400 → 800 |
| 2 | 0 | BB72 | 4 | .04 | 7201040 | 59122447 | 400 → 800 |
| 2 | 1 | BB72 | 4 | .06 | 7201060 | 59122447 | 400 → 800 |
| 2 | 2 | BB72 | 4 | .08 | 7201080 | 59122447 | 400 → 800 |
| 2 | 3 | BB144 | 4 | .08 | 14401080 | 59122448 | 400 → 800 |

The failed model first completes 400 total epochs, filling the missing point.
It is deliberately not labelled as an 800-epoch model. A later continuation to
800 can use its newly completed job as a source.

## Failure and numeric fix

`59122450/exp_1` failed after training zero-based epoch 114, during validation:
`DirectOSD0` rejected component probabilities as not finite/in-range/matching
shape. The latest saved checkpoint is epoch 113 (114 completed epochs).
Its latest and selected-best model weights are finite. Checking 4,096 CPU
validation shots at that saved checkpoint did not reproduce the original error;
the failed epoch's posterior and GPU arithmetic are not available locally.

The same float32 softmax-marginal calculation does reproduce a rounding hazard:

```text
logits = [-9.301547050476074, -0.05920518934726715,
           7.499649524688721, -12.816020011901855]
P(X)+P(Y) = 1.0000001192092896
```

This is a plausible explanation, not proof of that specific server failure.
The model now bounds the computed component probabilities to [0,1]. Softmax
roundoff no longer trips OSD's strict range check. NaNs remain NaNs; evaluation
rejects invalid component probabilities and saves the offending syndrome,
logits and marginals to `invalid_component_probabilities.npz`. Direct OSD now
reports shape, nonfinite count or min/max separately. Genuine invalid inputs
are not silently repaired. Each new training phase records
`component_probability_policy=softmax_marginal_clamp_v1`; original archived
outputs and config identities are preserved.

## Continuation settings

- Latest `model.pt`, with optimizer moments, RNG, samplers and the old selected
  best retained. The launcher refuses a selected `best_model.pt` or a mismatched
  code/p/seed/depth/width/completed-epoch source.
- Completed runs add **400 epochs** at initial LR **1e-4**, with a new cosine
  decay, reaching 800 epochs / 6,553,600 training shots. Recovery adds 286 epochs
  at LR 1e-4 to reach 400 total. Recovery is explicitly marked `--allow-incomplete`.
- Width 64; 128 batches × 64 shots per epoch; loss, weight decay and gradient
  clip unchanged. Validation: 4,096 shots every five additional epochs.
- Raw validation LER still selects the checkpoint. Final: 65,536 fresh shots,
  raw CNN and direct OSD-0. The frozen pre-resume best is reevaluated on exactly
  the same final bank with paired rescued/harmed counts. For recovery that
  reference is the best checkpoint selected before the crash, not a completed
  400-epoch model.
- A latest checkpoint is saved each epoch. The new model's selected best may
  remain the previous best if further training fails to improve validation.
- Perlmutter account m5328_g, 24 hours, one node, four tasks, one GPU and
  16 CPUs per task. Four source checkpoints are validated before launching.

## Locate checkpoints and run

`scripts/continue_bb_tanner_cnn.py` searches the server's original
`$HOME/end_time/resdir_JOB/outputs/.../model.pt` and the organized local archive
`results/bb/code_capacity/depolarizing/tanner_cnn/400_epochs/{code,replicates}/resdir_JOB/`.
New mixed-code jobs are also found under
`results/bb/code_capacity/depolarizing/tanner_cnn/continued_training/resdir_JOB/`.
Old 100-epoch archive paths remain supported. Missing or ambiguous sources fail;
there is no fresh-training fallback. `BB_CNN_SOURCE_ROOT` overrides the directory
containing the source job directories. The old root source folders can remain
on the cluster even though downloaded results were organized locally.

Both Python changes and Slurm scripts must be present on the cluster before
submission; updating a launcher alone would not fix the probability calculation.

```bash
# Inspect shell-generated arguments (no training or Python imports):
BB_DRY_RUN=1 bash scripts/slurm_archive/bb_depth45_resume800/run_bb_0.slurm

# Validate the actual failed checkpoint and print its recovery command:
python scripts/continue_bb_tanner_cnn.py \
  --source-job=59122450 --code=bb144 --depth=5 --p=.04 --seed=14401040 \
  --expected-epochs=114 --target-epochs=400 --lr=.0001 \
  --allow-incomplete --dry-run

# Reproduce the three historical jobs:
for i in 0 1 2; do sbatch "scripts/slurm_archive/bb_depth45_resume800/run_bb_${i}.slurm"; done
```

The launchers create new `resdir_<SLURM_JOB_ID>` folders and never overwrite the
source results. Previously submitted batch scripts are not altered by these edits.

## Local validation

The 23 focused model, OSD, trainer and continuation tests pass, including the
float32 roundoff regression and nonfinite diagnostic capture. All 12 actual
source checkpoints and shell dry-run commands were validated. A temporary CPU
smoke run restored the failed BB144 depth-5 model, ran one short epoch
(114→115 with two training batches), and completed both raw/OSD evaluation and
the paired frozen-best comparison. This checks recovery plumbing; it is not a
new performance result or a GPU reproduction of the original failure. No Slurm
job was submitted. All nine previous scripts were preserved byte-for-byte.
