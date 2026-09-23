# Neural Relay BP2 for BB circuit noise

`models/_neural_relay_bp2.py` combines the existing neural check update with
the variable-memory and relay construction of
[Relay-BP, Sections II–III and Algorithm 1](https://arxiv.org/html/2506.01779v2).
It is a new hybrid, not a reproduction of the paper's full decoder settings.
No OSD or matrix inversion is needed. The legacy decoder remains the default
when `--bp_relay_legs=0`.

## Update and outputs

The original normalized min-sum check update, residual MLP, and learned
relaxation remain unchanged. For variable `v`, let `prior[v]` be its original
physical-prior LLR and `posterior[v]` its preceding posterior LLR. Each iteration
uses:

```text
bias[v] = prior[v] + gamma[v] * (posterior[v] - prior[v])
total[v] = bias[v] + sum(neural_check_messages[:, v])
posterior[v] = clip(total[v])
variable_to_check[v, c] = clip(total[v] - check_to_variable[c, v])
```

The bias is included once per variable, not once per edge. Original priors stay
fixed across the whole relay. The extrinsic sum excludes the recipient before
clipping. Since the September 17 fix, the non-Relay decoder also follows this
order; older archived runs clipped the posterior first. Within the Relay model,
zero-initialized neural and non-neural paths
are identical when supplied the same memory draws.

At each leg boundary, edge messages restart from the original priors (and zero
check messages); the final posterior becomes the next leg's initial memory.
Memory is constant within a leg. The first leg uses one configured strength;
later legs draw independent strengths for every shot and mechanism. These
strengths are not trainable parameters. The residual and relaxation are trained
through the resulting dynamics.

`forward(detectors, return_all=True)` returns the terminal posterior `[B,N]`
and history `[B, legs*iterations, N]`. Training unrolls every step, including
gradients through leg boundaries. Random memory is sampled outside activation
checkpointing, so backward recomputation uses exactly the same values.

`decode(detectors)` performs inference with per-shot stopping. Each leg stops
at its first syndrome-valid correction. The relay stops after the requested
number of successful legs or the leg budget; successful legs can return the
same correction. Among valid candidates it retains the one minimizing
`sum(correction[v] * original_prior_llr[v])`. This is a mechanism likelihood
cost, not an exact logical-coset likelihood. No observable or mechanism labels
enter candidate selection. If no valid candidate is found, it returns the last
hard decision with `converged=False`.

The result contains `correction`, the corresponding `posterior`, `converged`,
and per-shot counts `iterations`, `legs`, and `solutions`. Positive LLR means
the mechanism is believed absent; `correction = (posterior < 0)`.

Fixed-budget training and early-stopped inference have different execution
paths. Checkpoint selection uses the actual inference decoder, not the last
training iteration's hard decision.

## Run without OSD

```bash
python main.py --code=bb72 --architecture=bb_neural_bp \
  --noise_model=circuit --rounds=6 --p=0.003 --measurement_error_rate=0.003 \
  --bp_iterations=12 --bp_residual_hidden_dim=32 --bp_orbit_embedding_dim=8 \
  --bb_bp_reference_iterations=1000 \
  --bp_relay_legs=4 --bp_relay_solutions=2 \
  --bp_relay_memory_strength=0.125 \
  --bp_relay_memory_min=-0.24 --bp_relay_memory_max=0.66 \
  --epochs=100 --batch_size=8 --batches=128 \
  --eval_batches=64 --eval_every=10 --final_eval_batches=512 \
  --bb_osd_eval_shots=0 --lr=0.0003 --amp_dtype=none --seed=7203001 --save_model
```

The Relay decoder uses at most 48 BP iterations per evaluation shot, and exactly
48 during training. Evaluation also runs ordinary BP with the separate
1,000-iteration cap described below. It is more expensive than a single T=12 run;
begin with a small smoke
run before scheduling a campaign. Memory bounds must lie strictly inside
`(-1,1)`, can include negative values, and are tunable starting settings rather
than validated optima for this repository. All Relay options apply only to BB
circuit noise.

## Evaluation, reproducibility and comparisons

The paired baseline for a Relay run is **Relay min-sum with neural residual and
relaxation disabled**, not legacy single-pass BP. Both paths receive identical
per-shot memory tensors and iteration budgets, and are scored against the same
fresh exact Stim circuit samples. The historical `vanilla_*` JSON fields hold
this non-neural Relay baseline; metadata records `baseline_decoder=relay_min_sum`
and `architecture=bb_neural_relay_bp_circuit`. Do not mix these runs into the
archived single-pass summaries. Text logs label the baseline `Relay BP`.

Evaluation uses a local seeded Torch generator, so its memory draws do not
consume training randomness. Training uses the Torch RNG already stored in
checkpoints. `sample_memory(..., generator=...)` and the explicit `memory=`
argument on both `forward` and `decode` support reproducible custom comparisons.
The JSON history and text log include mean iterations and mean legs for each
decoder. These are operation counts, not wall-clock latency measurements.

### Ordinary BP references (September 22 evaluation policy)

Every validation and final evaluation also reports ordinary normalized min-sum
BP at two caps: `--bp_iterations` (normally 12) and
`--bb_bp_reference_iterations` (default 1000). Both use the same exact circuit
shots as the neural and Relay decoders. A single ordinary-BP trajectory records
both caps. Each shot stops at its first correction satisfying `H @ correction = s`;
later updates cannot replace that correction. Shots that exhaust the cap retain
their last hard decision and count as flagged failures. Logical labels are used
only for scoring, so a syndrome-valid but logically wrong correction still stops.

These references disable neural terms and Relay memory, including in Relay
runs. They retain the configured normalization (default 0.625) and message clip
(30); increasing the cap alone does not reproduce a paper's complete decoder.
For Relay, the short reference uses the per-leg depth, not the total leg budget.
If both caps are equal, a single reference is recorded. The extra reference can
increase validation time substantially when many shots fail to converge.

`history.json` stores `bp_baselines.bp_12` and `bp_baselines.bp_1000` under each
evaluation with the default caps. Each row includes LER, syndrome convergence,
flagged/unflagged failures, mean BP iterations, and paired neural gain with
rescued/harmed counts. Text logs report both references. Names use the actual
caps, such as `bp_48` for a 48-step model.

Non-Relay Neural BP also stops at its first valid correction within its trained
depth. Training still unrolls the full depth with gradients. Relay inference
and its candidate selection are unchanged. Checkpoint selection retains its
primary paired baseline: short-cap BP for non-Relay runs and matched Relay for
Relay runs. The 1000-cap reference is additional reporting, not a new selection
target.

If OSD evaluation is enabled, only shots that failed to find a valid correction
are sent to the existing posterior-reseed BP/OSD wrapper. Valid corrections are
preserved. The wrapper itself is unchanged and is not a direct posterior-to-OSD
handoff. This comparison uses the primary baseline; ordinary BP-1000 is reported
without OSD.

The neural update retains orbit parameter sharing. Independent random memory
preserves translation symmetry in distribution, not for a fixed unpermuted
random draw. To test pointwise equivariance, transform the memory assignments
along with the syndrome and graph. Candidate ties can further require a
specified tie-breaking convention.

`--load_model` resumes only a compatible experiment: Relay legs, solution
budget, memory settings, BP evaluation policy, and reference cap must match.
The new policy is recorded as `bp_evaluation_policy=first_syndrome_valid_v1`.
Checkpoints lacking it cannot resume their old selection history under this
evaluation. The neural state-dictionary layout is shared,
so a custom inference experiment can explicitly load old neural weights into
`NeuralRelayBP2`; changing the decoder this way is a new evaluation, not a
continuation of the old checkpoint-selection history.
