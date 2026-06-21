"""
Faz 3 — operatör ablation sonuç toplama (laptop).
10 seed-etiketli *_metrics.json (5 GAT + 5 Conv) okur, operatör başına test
metriklerinin mean ± std'sini hesaplar, GAT vs Conv karşılaştırma tablosu basar
+ kaba anlamlılık (|Δ| > birleşik std mı).

Varsayılan klasör: results/operator_ablation/  (rsync ile indirilen JSON'lar)
Her JSON kendini tanımlar: top-level "operator" ("GAT"/"Conv") + "seed" + "test" bloğu.
"""
import os, sys, glob, json
import numpy as np

REPO = r"/path/to/project"
DEFAULT_DIR = os.path.join(REPO, "results", "operator_ablation")

# Raporlanacak test metrikleri: (JSON anahtarı, görünen ad)
METRICS = [
    ("csi_m",        "CSI-M"),
    ("csi_pool4_m",  "CSI-pool4"),
    ("csi_pool16_m", "CSI-pool16"),
    ("ssim",         "SSIM"),
]
# Ağır threshold'lar (csi_per_threshold listesinden): index 4 = tau181, 5 = tau219
HEAVY = [(4, "CSI@181"), (5, "CSI@219")]


def load_runs(folder):
    runs = {"GAT": [], "Conv": []}
    # düz + alt-klasörler (per-run klasör yapısı) — ikisini de tara
    files = sorted(set(glob.glob(os.path.join(folder, "*_metrics.json")) +
                       glob.glob(os.path.join(folder, "**", "*_metrics.json"),
                                 recursive=True)))
    if not files:
        sys.exit(f"JSON bulunamadı: {folder}\\**\\*_metrics.json")
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            d = json.load(f)
        op = d.get("operator")
        if op not in runs:
            print(f"  ! atlandı (operator '{op}' tanınmadı): {os.path.basename(fp)}")
            continue
        runs[op].append(d)
        print(f"  yüklendi: {os.path.basename(fp)}  (operator={op}, seed={d.get('seed')})")
    return runs


def collect(run_list, key, heavy_idx=None):
    vals = []
    for d in run_list:
        t = d["test"]
        if heavy_idx is None:
            vals.append(float(t[key]))
        else:
            vals.append(float(t["csi_per_threshold"][heavy_idx]))
    return np.array(vals, dtype=np.float64)


def fmt(arr):
    return f"{arr.mean():.4f} ± {arr.std(ddof=1):.4f}" if len(arr) > 1 else f"{arr.mean():.4f} (n=1)"


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    print(f"Klasör: {folder}\n")
    runs = load_runs(folder)
    n_gat, n_conv = len(runs["GAT"]), len(runs["Conv"])
    print(f"\nGAT run: {n_gat}  |  Conv run: {n_conv}\n")
    if n_gat == 0 or n_conv == 0:
        sys.exit("Her iki operatör için de en az 1 run gerekli.")

    print(f"{'Metrik':<12}{'GAT (μ±σ)':<22}{'Conv (μ±σ)':<22}{'Δ(GAT−Conv)':<14}{'|Δ|>σ?':<8}")
    print("-" * 78)

    rows = list(METRICS) + [(("HEAVY", i), name) for i, name in HEAVY]
    for key, name in rows:
        if isinstance(key, tuple) and key[0] == "HEAVY":
            g = collect(runs["GAT"], None, heavy_idx=key[1])
            c = collect(runs["Conv"], None, heavy_idx=key[1])
        else:
            g = collect(runs["GAT"], key)
            c = collect(runs["Conv"], key)
        delta = g.mean() - c.mean()
        # kaba anlamlılık: |Δ| birleşik std'yi (karekök(σ_g²+σ_c²)) aşıyor mu
        comb = np.sqrt(
            (g.std(ddof=1) ** 2 if len(g) > 1 else 0.0) +
            (c.std(ddof=1) ** 2 if len(c) > 1 else 0.0))
        sig = "evet" if comb > 0 and abs(delta) > comb else "hayır"
        print(f"{name:<12}{fmt(g):<22}{fmt(c):<22}{delta:+.4f}        {sig:<8}")

    print("-" * 78)
    print("Not: 'Δ' pozitifse GAT üstün. '|Δ|>σ' kaba bir işaret — birleşik std'yi")
    print("aşan fark gürültü-üstü kabul edilir (kesin test için Welch t-test eklenebilir).")
    print("Param: GAT 161,505  |  Conv 683,745 (~4×) — efficiency yorumunda kullan.")


if __name__ == "__main__":
    main()
