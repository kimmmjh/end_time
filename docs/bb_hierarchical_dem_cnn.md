# Hierarchical DEM graph CNN

The [November 2 follow-up scripts](bb_dem_followup_2026_11_02.md) separate
additional DEM CNN training from immediate best/latest hierarchical evaluation.
The imported hierarchical results below used existing best weights with no retraining.

## Full imported campaign (November 2, 2026)

All 24 evaluations in the manifest are imported: BB72 depths 4/6/8/12 and
BB144 depths 4/6, each at four p values and 8,192 paired shots. All hierarchical
outputs are reported syndrome-consistent. BB72 D12 improves the point estimates
at p=0.2%/0.4%/0.6%, but depth does not improve every point monotonically.
BB144 D4 at p=0.1% has 0/8,192 failures, with 95% Wilson upper bound 0.0469%.
BB144 decoding takes roughly 624–1,769 amortized ms/shot with this dense backend.

See the [combined analysis](../results/analysis/bb_dem_2026_11_02.md),
[depth comparison](../results/plots/bb/circuit/standard/dem_cnn/hierarchical/depth_comparison.png),
and [timing](../results/plots/bb/circuit/standard/dem_cnn/hierarchical/depth_diagnostics.png).
Shot banks/checkpoints remain absent locally; direct CNN+OSD was not compared.
BB144 D8/D12 hierarchical evaluations are not included, even though their raw
training runs have now completed.

## Imported evaluation (November 1, 2026)

BB72 D4 job 60636229 completed all four p values with 8,192 paired shots each,
using existing best weights and confidence 0.9. Raw versus hierarchical block
failure at p=0.1% is 16.772% versus 0.110%; at p=0.2% it is 54.028% versus
3.174%. No retraining occurred. High-p failure remains large. Direct CNN+OSD
was not evaluated, so these results do not isolate an advantage of hierarchy
over ordinary CNN+OSD. Only metadata/logs were imported, not shot banks or weights.

See the [analysis](../results/analysis/bb_hierarchical_dem_2026_11_01.md) and
[performance plot](../results/plots/bb/circuit/standard/dem_cnn/hierarchical/overview.png).

## Implementation

`models/_bb_hierarchical_dem_cnn.py` defines `BBHierarchicalDemCNN`, a
feed-forward local graph CNN followed by a global residual decoder. It keeps
the undecomposed DEM hyperedges, joint X/Z information, variable-degree
aggregation, orbit sharing and temporal boundaries of `BBDemCNN`. There is
no recurrence or geometric pooling.

```python
from models import BBHierarchicalDemCNN

model = BBHierarchicalDemCNN(
    graph, width=64, depth=2, confidence_threshold=0.9,
)
# Train with the existing CircuitDegeneracyAwareLoss and DEM samples.
llrs = model(batch.detectors)
loss = criterion(llrs, batch.detectors, batch.mechanisms).total
loss.backward()

model.eval()
result = model.decode(detectors)  # default: CPU residual DirectOSD0
corrections = result.correction  # uint8 [batch, mechanisms], model device
predicted_observables = (graph.observable_matrix @ corrections.cpu().numpy().T).T % 2
```

The forward pass is identical to `BBDemCNN` and accepts its state dict with
`strict=True`. This implementation changes the inference hierarchy, not the
training objective. Choose a shallow depth for local preprocessing; imported
deep checkpoints keep their larger receptive field. The threshold and
`active_candidates_only` setting are experiment configuration and are not
stored in the state dict. Rebuild with the same fingerprint-checked graph.
This API is not registered as a new main.py training architecture. The
evaluation-only launcher below reuses existing checkpoints without training.

## Inference contract

1. Predict LLRs `log(P(no fault) / P(fault))` using the graph CNN.
2. Apply local flips `a` only where fault probability is at least the confidence
   threshold. By default, a candidate must touch at least one active detector.
   All local decisions use the original syndrome and are applied together.
3. Form residual detectors `r = s XOR H a` and residual LLRs
   `lambda_res = (1 - 2*a) * lambda` for the variable `b = e XOR a`.
4. Skip global completion if `r` is zero. Otherwise pass residual syndromes and
   probabilities to `backend.decode_batch(syndrome, probabilities)` using NumPy
   arrays. Keep every original mechanism column available, including local
   flips, so the backend can undo mistakes.
5. Return `a XOR b`, checking `H(a XOR b) = s`. Backend errors propagate;
   invalid shapes/bits or syndrome-inconsistent outputs raise exceptions.

`preprocess` returns the local decisions, original LLRs, residual detectors and
residual LLRs without invoking the backend. `decode` additionally returns the
global residual correction, combined correction, per-shot backend-use mask,
and syndrome convergence. Both methods are inference-only; train via `forward`.
True observables and sampled fault labels are never inputs to decoding.

## Interpretation and benchmarks

Abstention means leaving a mechanism unchanged locally, not declaring it
fault-free. Confidence is not automatically calibrated. A zero residual does
not establish logical success; measure the final observable signature as well.
Residual neural probabilities are not independent channel priors, particularly
when used in a BP backend.

The default dense CPU OSD-0 backend is a correctness reference, not a scalable
speed claim. On shots sent to it, sign-transforming the LLRs and XOR-composing
the result is algebraically equivalent to hard-centered OSD-0 on the original
CNN predictions, apart from numerical ties/clipping. The local zero-residual
early exit can change which prediction is returned. Thus this implementation
provides a controlled hierarchical baseline; it does not by itself promise
an accuracy improvement over CNN + OSD-0. Alternative residual backends can be
supplied through the same interface.

Benchmark against the original CNN, the classical backend alone and CNN plus
OSD-0 on identical circuit shots. Report block failure, syndrome failure,
backend-use fraction and total decoding time including transfer. The local CNN
still processes the full graph; candidate gating does not make its forward
pass sparse in the observed events.

## Evaluate the previously trained models on Perlmutter

The launcher `run_bb_dem_hierarchical_eval.slurm` uses the existing Perlmutter
options: account `m5328_g`, regular GPU queue, one node, four tasks with one
GPU and 16 CPUs each, 32 GB per worker step, and a 24-hour limit. It loads
`python/3.10` and activates `$PSCRATCH/envs/nde`. The repository defaults to
`$HOME/end_time`; override with `BB_REPO_ROOT` if needed.

Upload these new files into the same repository on Perlmutter (along with
any changed dependencies if the server repository is older):

- `run_bb_dem_hierarchical_eval.slurm`
- `scripts/evaluate_bb_hierarchical_dem_cnn.py`
- `scripts/manifests/bb_dem_hierarchical_800_epochs.json`
- `models/_bb_hierarchical_dem_cnn.py`

The manifest records the 24 completed 800-epoch runs found in
`results/bb/circuit/standard/dem_cnn/800_epochs`. Original task numbering is
preserved:

| Array task | Code | Depth | Original source job |
|---|---|---:|---|
| 0 | BB72 | 4 | 60310802 |
| 1 | BB144 | 4 | 60310803 |
| 2 | BB72 | 6 | 60316098 |
| 3 | BB144 | 6 | 60316182 |
| 4 | BB72 | 8 | 60318763 |
| 6 | BB72 | 12 | 60330386 |

Each array task runs four workers for p=.001/.002/.004/.006, respectively.
BB144 depths 8/12 are omitted because completed results are not present in
this imported campaign. Missing results are not treated as failed decodes.

Local imports contain histories and logs **but no DEM CNN `.pt` files**.
The evaluator first searches the original server `resdir_JOB/outputs/...`
path, then the imported `results/...` path. Keep the original trained weights
on Perlmutter. A JSON history is not enough to evaluate a network.
`BB_DEM_SOURCE_ROOT` can point to another parent containing the original
`resdir_*` directories. No fuzzy/latest-by-modification-time selection occurs.

```bash
cd ~/end_time

# Print all 24 source selections. Reports absent checkpoints without running.
BB_DRY_RUN=1 bash run_bb_dem_hierarchical_eval.slurm

# Start with 128 shots per rate for BB72 D4 to measure dense OSD cost.
sbatch --array=0 --export=ALL,BB_HIER_SHOTS=128 run_bb_dem_hierarchical_eval.slurm

# Full campaign: 8,192 paired fresh circuit shots per model, confidence 0.9.
sbatch run_bb_dem_hierarchical_eval.slurm

# Optional direct CNN+OSD comparator on the same shots, in addition to hierarchy.
sbatch --export=ALL,BB_HIER_COMPARE_DIRECT_OSD=1 run_bb_dem_hierarchical_eval.slurm
```

The default chooses `best_model.pt`. If it is absent, it reads the embedded
`best_model_state_dict` in `model.pt`, not the latest weights. The checkpoint
configuration is checked against the manifest and reconstructed graph before
inference. No optimizer is constructed, no training occurs, and source files
are never overwritten. High-p BB144 best weights can still be epoch 5 because
the original validation results tied; selecting `latest` is a separate ablation.

| Environment variable | Default | Meaning |
|---|---|---|
| `BB_HIER_SHOTS` | 8192 | Total shots per model; all are decoded by both branches |
| `BB_HIER_BATCH_SIZE` | 4 | GPU batch size; partial last batches are supported |
| `BB_HIER_EDGE_CHUNK_SIZE` | 4096 | Sparse convolution chunk size |
| `BB_HIER_CONFIDENCE` | 0.9 | Fixed local flip threshold, strictly between 0.5 and 1 |
| `BB_HIER_WEIGHTS` | best | `best` or `latest` checkpoint weights |
| `BB_HIER_SEED` | 20261101 | Seed for a new circuit evaluation stream |
| `BB_HIER_COMPARE_DIRECT_OSD` | 0 | Also run direct CNN+OSD-0; increases CPU work |
| `BB_HIER_SAVE_CORRECTIONS` | 0 | Save full mechanism correction arrays per batch |
| `BB_HIER_MANIFEST` | bundled manifest | Explicit alternative source manifest |
| `BB_PYTHON` | python on server | Python executable; useful for local previews |

New results go to `resdir_JOB_hierarchical/experiment_INDEX/`:

- `metadata.json`: source checkpoint SHA-256, actual selected epoch, graph/noise
  config, seed, software versions, threshold and timing conventions.
- `results.json`: running/final counts, Wilson 95% block-failure intervals,
  syndrome and observable failures, paired gain/rescued/harmed counts, global
  call fraction, residual weight and measured decoding time.
- `shots/batch_*.npz`: paired detector/observable samples, predictions and
  success flags; optional full corrections. Completed batches survive an
  interruption. Partial `status=running` results are not full-bank results.

The job directory also preserves logs, commands, exit codes, manifest and
source snapshots. Output directories must be new; interrupted evaluation is
not automatically resumed. The full-bank dense OSD runtime is unmeasured and
may exceed the wall-time limit. The pilot estimates this cost; sparse residuals
do not necessarily make this dense solver fast.

Both branches use identical fresh exact-circuit shots, not the original final
shot bank. Seeds are shared across depths for equal code/rounds/noise; exact
cross-job pairing also requires identical Stim versions and sampling call
sizes, so retain the saved shot arrays for verification. Threshold selection
must use a separate validation seed/bank, not the final test results.

For an individual checkpoint or a completed BB144 D8/D12 continuation, use
the helper directly in an appropriate allocation:

```bash
python scripts/evaluate_bb_hierarchical_dem_cnn.py \
  --checkpoint /path/to/resumed/run/model.pt --weights best \
  --output /path/to/new/evaluation --shots 8192 --batch-size 4 --device cuda
```

For a local CPU smoke run, use `--device cpu --shots 2`. For a strict source
preflight, add `--dry-run --require-checkpoint` to a helper invocation. The
Slurm launcher performs that existence check for all four workers before
launching them. To run a different campaign through the array launcher, copy
the manifest, specify its run paths/configurations explicitly, and select its
task IDs with `sbatch --array=...` and `BB_HIER_MANIFEST`.
