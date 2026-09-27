# Circuit BP baseline campaign

The active 0–4 scripts now evaluate library plain BP, BP+OSD-0 and BP+OSD-CS3
on a paper-style circuit memory experiment. See the complete
[baseline campaign specification](bb_baseline_campaign.md).

The previous 0.625 standalone sweep and its derived plot/CSVs have been removed
at the user's request. The earlier Torch-only evaluator
`scripts/evaluate_bb_circuit_bp.py` remains available for implementation checks;
it is not the entrypoint used by the current Slurm campaign.
