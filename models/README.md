# models

Data prep and training scripts. Set the data and output paths near the top of each file
before running.

`02_veri_indir2.py` downloads the SEVIR `CATALOG.csv`, keeps the VIL rows, and saves them as
`benim_tez_katalogum.csv` in the current folder, so run it from inside `models/`. It also writes
`verileri_indir.bat`, a Windows batch file with one `aws s3 sync` line per VIL HDF5 file. You
only need it to rebuild the catalog; the CSV is already here.

`benim_tez_katalogum.csv` is the catalog that `05_sevir_dataset.py` reads: 20,393 VIL rows
covering 14 HDF5 files from 2017 to 2019. The `pct_missing` filter and the date split are
applied in `05`, not in this file.

`05_sevir_dataset.py` builds the training tensors from the raw SEVIR HDF5 files: 3x3
max-pool down to 128x128, the 7->6 framing at 10-minute spacing, the `pct_missing == 0`
event filter, and the event-based test split (time_utc >= 2019-06-01).

`06_sevir_subset_olustur.py` makes a small subset so you can test the pipeline on a laptop.

`15a_graphconvlstm_gat_truba.py` is the main model, the one with the graph-attention gate.
The seed is read from the `SEED` env var (we ran ten seeds: 17, 42, 123, 271, 456, 789,
2024, 8675, 13579, 90210).

`15a_graphconvlstm_gat_sigma_probe.py` is the same script with the Gaussian edge-weight
bandwidth exposed as the `SIGMA` env var (default 1.0). We used it for a single-seed check
at sigma = 0.5 and 2.0; outputs are tagged `_sigmaX` so they do not overwrite the main run.

`15b_convlstm_gate_truba.py` is the same model and pipeline but the gate is a fixed 3x3
convolution instead of graph attention. This is the operator ablation, and it also doubles
as a ConvLSTM baseline trained under our own setup.

`14_sevir_unet_convgru_gat_b1_truba.py` is the larger variant (U-Net encoder/decoder,
ConvGRU bottleneck, multi-hop graph attention). Used for the capacity ablation.

`13_sevir_graphconvlstm_v5b_truba.py` is an earlier version of the model.
