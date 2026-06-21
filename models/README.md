# models

Data prep and training scripts. Set the data and output paths near the top of each file
before running.

`05_sevir_dataset.py` builds the training tensors from the raw SEVIR HDF5 files: 3x3
max-pool down to 128x128, the 7->6 framing at 10-minute spacing, the `pct_missing == 0`
event filter, and the event-based test split (time_utc >= 2019-06-01).

`06_sevir_subset_olustur.py` makes a small subset so you can test the pipeline on a laptop.

`15a_graphconvlstm_gat_truba.py` is the main model, the one with the graph-attention gate.
The seed is read from the `SEED` env var (we ran 42, 123, 456, 789, 2024).

`15b_convlstm_gate_truba.py` is the same model and pipeline but the gate is a fixed 3x3
convolution instead of graph attention. This is the operator ablation, and it also doubles
as a ConvLSTM baseline trained under our own setup.

`14_sevir_unet_convgru_gat_b1_truba.py` is the larger variant (U-Net encoder/decoder,
ConvGRU bottleneck, multi-hop graph attention). Used for the capacity ablation.

`13_sevir_graphconvlstm_v5b_truba.py` and the matching `..._kaggle.ipynb` are an earlier
version of the model (TRUBA script and the Kaggle notebook).
