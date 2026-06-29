# results

A couple of example metric files, not the full set. Each training run writes one JSON like
these, and `analysis/aggregate_ablation.py` reads them from `results/operator_ablation/`.

`operator_ablation/` holds two runs at the same seed: the main model
(`graphconvlstm_gat_seed42`) and the convolution-gate baseline (`convlstm_gate_seed42`). That
is enough to see the output format and the per-run structure. With the full set of seeds, this
same folder is what the aggregation and the paired test run on.

Each JSON records the operator, the seed, the training loss, and `val`/`test` blocks with MSE,
RMSE, SSIM, and CSI (mean, per threshold, and pooled at 4 and 16). The complete per-seed
numbers and the tables are in the paper.
