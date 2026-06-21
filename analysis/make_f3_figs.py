"""
Regenerate the F.3 figures from the SAVED metrics (analysis/f3_out/f3_metrics.json).
No model re-run, no recomputation -- just re-plots the stored arrays.

Outputs:
  - FSS  (paper figure):  article_tex/figures/fig_3_fss.pdf  + fig_3_fss.png (600 dpi)
  - RAPSD (cleaned, side): analysis/f3_out/f3_rapsd_clean.pdf + .png (600 dpi)
        formal title + x-axis cropped to k<=50 (drops the Nyquist-edge artefact).
        NOT included in the manuscript -- kept for reference only.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(REPO, "analysis", "f3_out", "f3_metrics.json")))

# FSS  --  model vs persistence, at +60 min (leadtime index 5), 4 thresholds
f = d["fss"]
thr = f["thresholds"]                  # ['16','74','133','181']
scales = np.array(f["scales"])         # [1,3,5,9,17,33]
lt_idx = f["leadtimes_min"].index(60)  # +60 min
model = np.array(f["model"])           # (thr, scale, leadtime)
pers = np.array(f["pers"])

fig, axes = plt.subplots(1, len(thr), figsize=(15, 3.6), sharey=True)
for j, t in enumerate(thr):
    ax = axes[j]
    ax.plot(scales, model[j, :, lt_idx], "o-", color="#d62728", lw=2,
            ms=6, label="GraphConvLSTM")
    ax.plot(scales, pers[j, :, lt_idx], "s--", color="#2ca02c", lw=2,
            ms=6, label="Persistence")
    ax.set_title(f"$\\tau={t}/255$  ($+60$ min)", fontsize=12)
    ax.set_xlabel("Neighbourhood (px)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)
    if j == 0:
        ax.set_ylabel("FSS", fontsize=11)
        ax.legend(fontsize=10, loc="lower right")
fig.suptitle("Fractions Skill Score vs. spatial scale (test set, $+60$ min)",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])

fss_pdf = os.path.join(REPO, "article_tex", "figures", "fig_3_fss.pdf")
fss_png = os.path.join(REPO, "article_tex", "figures", "fig_3_fss.png")
fig.savefig(fss_pdf)                       # vector
fig.savefig(fss_png, dpi=600)              # >=600 dpi raster
plt.close(fig)
print("FSS  ->", fss_pdf, "+ fig_3_fss.png (600 dpi)")

# RAPSD (cleaned) -- formal title, x cropped to k<=50, side reference only
r = d["rapsd"]
k = np.array(r["wavenumber"], dtype=float)
mask = (k >= 1) & (k <= 50)             # drop k=0 and the Nyquist-edge artefact
fig2, ax = plt.subplots(figsize=(6.2, 4.6))
ax.plot(k[mask], np.array(r["gt"])[mask], "-", color="black", lw=2, label="Ground truth")
ax.plot(k[mask], np.array(r["pers"])[mask], "--", color="#2ca02c", lw=1.8, label="Persistence")
ax.plot(k[mask], np.array(r["blur_gt_sigma1.0"])[mask], ":", color="#1f77b4", lw=1.8,
        label="Blurred GT ($\\sigma=1$)")
ax.plot(k[mask], np.array(r["pred"])[mask], "-", color="#d62728", lw=2, label="GraphConvLSTM")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Radial wavenumber $k$", fontsize=11)
ax.set_ylabel("Radially averaged PSD", fontsize=11)
ax.set_title("Radially averaged power spectrum (test set, $+60$ min)", fontsize=12)
ax.grid(True, which="both", alpha=0.25)
ax.legend(fontsize=10)
fig2.tight_layout()
rap_pdf = os.path.join(REPO, "analysis", "f3_out", "f3_rapsd_clean.pdf")
rap_png = os.path.join(REPO, "analysis", "f3_out", "f3_rapsd_clean.png")
fig2.savefig(rap_pdf)
fig2.savefig(rap_png, dpi=600)
plt.close(fig2)
print("RAPSD (clean) ->", rap_pdf, "+ .png (600 dpi)  [side reference, NOT in paper]")
