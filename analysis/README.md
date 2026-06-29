# analysis

Things you run after training, once you have the per-seed metric json files and a saved
checkpoint. Paths to those are set at the top of each script.

`aggregate_ablation.py` collects the per-seed metrics, prints the mean and standard
deviation for each metric, and runs a paired t-test (plus a Wilcoxon signed-rank check)
comparing the graph-attention model against the convolution baseline.

`f3_spatial_spectral.py` does the post-hoc look at the test forecasts: radially averaged
power spectra, frequency bias, and the Fractions Skill Score against persistence. It is the
slow one (it loops over the whole test set).

`make_f3_figs.py` just redraws the FSS and RAPSD figures from the json that the previous
script writes, so you can tweak the plots without rerunning the model.
