# Result archive

Experiment directories keep their original `resdir_<id>` names and are grouped
by code family, noise model, and decoder:

```text
results/
├── toric/
│   ├── phenomenological/{cnn3d,convgru,pymatching}/
│   └── circuit/{convgru_mwpm,convgru_weighted_mwpm}/
├── plots/
└── local_smoke/
```

The threshold plotter searches recursively, so the archive can be passed as a
single input path:

```bash
python scripts/plot_threshold.py results/toric \
  --out results/plots/threshold.png
```

Known partial runs are retained in `cnn3d/resdir_55562044`,
`cnn3d/resdir_55860108`, and `convgru/resdir_1252167/exp_6`.
