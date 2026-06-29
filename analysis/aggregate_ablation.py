"""
Faz 3 — operatör ablation sonuç toplama (laptop).
Seed-etiketli *_metrics.json (GAT + Conv) okur, operatör başına test
metriklerinin mean ± std'sini hesaplar ve EŞLİ (paired) istatistik uygular:
paired t-test (ttest_rel) + Wilcoxon signed-rank (n>=6).

Neden eşli: aynı seed iki kolda da (GAT + Conv) kullanılıyor → fark dizisi
d_i = GAT_i - Conv_i seed etkisini sadeleştirir. Bu yüzden bağımsız-grup
(unpaired) bir kıyas (örn. |Δ| > sqrt(σ_g²+σ_c²)) YANLIŞTIR: korelasyonu ve
ortalama-farkın standart hatasını (std(d)/sqrt(n)) yok sayar.

Kullanım:
    python aggregate_ablation.py [JSON_KLASORU]
Varsayılan: results/operator_ablation/  (rsync ile indirilen JSON'lar)
Her JSON kendini tanımlar: top-level "operator" ("GAT"/"Conv") + "seed" + "test".
"""
import os, sys, glob, json
import numpy as np
from scipy.stats import ttest_rel, wilcoxon, t as tdist

REPO = r"/path/to/project"
DEFAULT_DIR = os.path.join(REPO, "results", "operator_ablation")
# Eksik-seed koruması: final operatör run'da `EXPECT_N=10 python aggregate_ablation.py ...`
# ver; paired n beklenenden azsa (bir seed FAILED/OOM) gözden kaçmaz uyarı basar.
EXPECT_N = int(os.environ.get("EXPECT_N", "0"))  # 0 = kontrol kapalı

# (JSON anahtarı, görünen ad, PRIMARY mi)
METRICS = [
    ("csi_m",        "CSI-M",      True),   # ön-tanımlı BİRİNCİL endpoint
    ("csi_pool4_m",  "CSI-pool4",  False),
    ("csi_pool16_m", "CSI-pool16", False),
    ("ssim",         "SSIM",       False),
]
# Ağır threshold'lar (csi_per_threshold listesinden): index 4 = tau181, 5 = tau219
HEAVY = [(4, "CSI@181"), (5, "CSI@219")]
# Geçerli bir run'ın test bloğunda bulunması gereken metrikler (+ csi_per_threshold[6])
REQUIRED_TEST = ("csi_m", "csi_pool4_m", "csi_pool16_m", "ssim")


def safe_load(fp):
    """JSON'u güvenle oku + 'test' bloğu / gerekli metrikleri doğrula. Bozuk veya
    eksik dosyada LOUD uyarı basıp None döner → run ATLANIR (yarım yazılmış HPC
    çıktısı tüm aggregation'ı çökertmesin). Düşen seed SANITY + EXPECT_N ile yakalanır."""
    try:
        with open(fp, "r", encoding="utf-8") as f:
            d = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  ! BOZUK JSON atlandı ({type(e).__name__}): {os.path.basename(fp)}")
        return None
    t = d.get("test")
    if not isinstance(t, dict):
        print(f"  ! GEÇERSİZ (test bloğu yok) atlandı: {os.path.basename(fp)}")
        return None
    cpt = t.get("csi_per_threshold")
    if not (isinstance(cpt, list) and len(cpt) == 6):
        print(f"  ! GEÇERSİZ (csi_per_threshold != 6) atlandı: {os.path.basename(fp)}")
        return None
    miss = [k for k in REQUIRED_TEST if k not in t]
    if miss:
        print(f"  ! GEÇERSİZ (eksik metrik {miss}) atlandı: {os.path.basename(fp)}")
        return None
    return d


def load_runs(folder):
    """{op: {seed: data}} döndürür — seed bazında, (op,seed) dedup'lu.
    Bozuk/eksik dosyalar safe_load ile elenir."""
    runs = {"GAT": {}, "Conv": {}}
    files = sorted(set(glob.glob(os.path.join(folder, "*_metrics.json")) +
                       glob.glob(os.path.join(folder, "**", "*_metrics.json"),
                                 recursive=True)))
    if not files:
        sys.exit(f"JSON bulunamadı: {folder}\\**\\*_metrics.json")
    for fp in files:
        d = safe_load(fp)
        if d is None:
            continue
        op, seed = d.get("operator"), d.get("seed")
        if op not in runs:
            print(f"  ! atlandı (operator '{op}' tanınmadı): {os.path.basename(fp)}")
            continue
        if seed is None:
            print(f"  ! atlandı (seed eksik/None): {os.path.basename(fp)}")
            continue
        if seed in runs[op]:
            print(f"  ! UYARI dup atlandı: {op} seed={seed} ({os.path.basename(fp)})")
            continue
        runs[op][seed] = d
        print(f"  yüklendi: {os.path.basename(fp)}  (operator={op}, seed={seed})")
    return runs


def val(d, key, heavy_idx=None):
    t = d["test"]
    return float(t["csi_per_threshold"][heavy_idx]) if heavy_idx is not None else float(t[key])


def main():
    try:  # Windows cp1254 konsolda μ/±/σ çökmesin (Spyder zaten UTF-8)
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    folder = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    print(f"Klasör: {folder}\n")
    runs = load_runs(folder)   # yükleme mesajları yalnız stdout (tanı amaçlı)

    # İstatistik raporu: hem stdout'a bas hem PAIRED_TTEST.txt'e yaz.
    report = []
    def P(line=""):
        print(line)
        report.append(line)

    gat_seeds, con_seeds = set(runs["GAT"]), set(runs["Conv"])
    P("=== SANITY ===")
    P(f"GAT  ({len(gat_seeds)}): {sorted(gat_seeds)}")
    P(f"Conv ({len(con_seeds)}): {sorted(con_seeds)}")
    common = sorted(gat_seeds & con_seeds)
    if gat_seeds != con_seeds:
        P(f"  ! UYARI: seed kümeleri FARKLI — paired testler KESİŞİM üzerinde "
          f"(n={len(common)}). Conv'da-yok={sorted(gat_seeds - con_seeds)} "
          f"GAT'ta-yok={sorted(con_seeds - gat_seeds)}")
    else:
        P(f"  OK: seed kümeleri EŞİT (n={len(common)}).")
    if not common:
        sys.exit("Ortak seed yok — paired test yapılamaz.")
    n = len(common)
    P(f"\nPaired n = {n}  (seed: {common})")
    if EXPECT_N and n != EXPECT_N:
        P(f"\n  !! DİKKAT: beklenen n={EXPECT_N} ama paired n={n} — EKSİK SEED.")
        P(f"      Sonuç '{EXPECT_N}-seed' DEĞİL ({n}-seed). Makaleye yazmadan KONTROL ET.")
    P()

    P(f"{('Metrik(n='+str(n)+')'):<13}{'GAT μ±σ':<15}{'Conv μ±σ':<15}"
      f"{'Δ(G−C)':<9}{'%95 GA(Δ)':<22}{'d_z':<6}{'t p':<9}{'W p':<10}")
    P("-" * 99)

    rows = [(k, nm, pr, None) for k, nm, pr in METRICS] + \
           [(None, nm, False, idx) for idx, nm in HEAVY]
    for key, name, primary, hidx in rows:
        g = np.array([val(runs["GAT"][s],  key, hidx) for s in common], dtype=np.float64)
        c = np.array([val(runs["Conv"][s], key, hidx) for s in common], dtype=np.float64)
        d = g - c                                                        # eşli fark dizisi
        md = d.mean()
        tp = ttest_rel(g, c).pvalue if n >= 2 else float("nan")          # eşli t
        if n >= 6 and np.any(d != 0):                                    # sıra-tabanlı
            # zero_method='zsplit': fark=0 çiftini atmak yerine böler → n sabit, öngörülebilir
            ws = f"{wilcoxon(g, c, zero_method='zsplit').pvalue:.4f}"
        else:
            ws = "n<6"
        # Δ üzerine paired %95 GA + Cohen's d_z (paired). Dağılım = std(d), std(g)/std(c) DEĞİL.
        if n > 1:
            sd_d = d.std(ddof=1)
            se = sd_d / np.sqrt(n)
            tcrit = tdist.ppf(0.975, n - 1)
            ci = f"[{md - tcrit*se:+.4f},{md + tcrit*se:+.4f}]"
            dz = md / sd_d if sd_d > 0 else float("nan")
            gm = f"{g.mean():.4f}±{g.std(ddof=1):.4f}"
            cm = f"{c.mean():.4f}±{c.std(ddof=1):.4f}"
        else:
            ci, dz, gm, cm = "(n=1)", float("nan"), f"{g.mean():.4f}", f"{c.mean():.4f}"
        tag = "  [PRIMARY]" if primary else ""
        P(f"{name:<13}{gm:<15}{cm:<15}{md:+.4f}  {ci:<22}{dz:<6.2f}{tp:<9.4f}{ws:<10}{tag}")

    P("-" * 99)
    P(f"n = {n} eşli seed.  Δ>0 → GAT üstün.  BİRİNCİL endpoint = CSI-M; "
      f"diğer 5 ikincil/destekleyici (M1c çerçevesi).")
    P("Test = paired t (ttest_rel): deney eşli (aynı seed iki kolda). "
      "Wilcoxon yalnız n>=6 (sıra-tabanlı, normallik varsaymaz).")
    P("Δ %95 GA = md ± t_{n-1,.975}·std(d)/√n  (paired; dağılım std(d), std(g)+std(c) DEĞİL).")
    P("d_z = mean(d)/std(d) (paired Cohen's d). DESIGN-SPECIFIC: fark-SD'si kullanır, "
      "Cohen'in between-group eşikleriyle (0.2/0.5/0.8) kıyaslanamaz (Lakens 2013).")
    P("μ±σ, Δ ve tüm testler n ORTAK seed üzerinden (kollar eşitse = tam set).")
    P("std = örnek std (ddof=1).")
    P("Param: GAT 161,505 | Conv 683,745 (~4×) — parameter-efficiency yorumu.")

    out_path = os.path.join(folder, "PAIRED_TTEST.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print(f"\n[yazıldı] {out_path}")


if __name__ == "__main__":
    main()
