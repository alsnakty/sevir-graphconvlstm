# 06_sevir_subset_olustur.py
#
# 05_sevir_dataset.py çıktılarından (train/val/test) subset oluşturur.
# seq[::2] ile 25 kare → 13 kare (10dk aralık, PreDiff protokolü).
# Çıktı shape: (N, 13, 128, 128) — Kaggle notebook N_INPUT=7, N_PREDICT=6 kullanır.
#
# Amaç: an HPC cluster full eğitim öncesi Kaggle'da yaklaşık sonuçları görmek.

import numpy as np
import os
import gc

DATA_DIR   = r"/path/to/data_root\data\sevir_processed"
TRAIN_FILE = os.path.join(DATA_DIR, "sevir_vil_train.npy")
VAL_FILE   = os.path.join(DATA_DIR, "sevir_vil_val.npy")
TEST_FILE  = os.path.join(DATA_DIR, "sevir_vil_test.npy")

N_TRAIN = 7000    # ~4.5 GB (13 kare × 7000)
N_VAL   = 1750    # ~1.1 GB (13 kare × 1750)
N_TEST  = 1500    # ~1.0 GB (13 kare × 1500)

rng = np.random.default_rng(42)

for src_path, n, tag in [(TRAIN_FILE, N_TRAIN, "train"),
                         (VAL_FILE,   N_VAL,   "val"),
                         (TEST_FILE,  N_TEST,  "test")]:
    print(f"\n{tag} yükleniyor (memmap)...")
    data = np.load(src_path, mmap_mode='r')
    total = data.shape[0]
    print(f"  Toplam: {total}  →  subset: {n}  (shape: {data.shape})")

    idx = rng.choice(total, size=n, replace=False)
    idx.sort()   # sıralı okuma disk için daha hızlı

    # seq[::2]: 25 kare → 13 kare (5dk → 10dk aralık, PreDiff protokolü)
    subset = np.array(data[idx, ::2, :, :])
    print(f"  seq[::2] sonrası shape: {subset.shape}")  # (N, 13, 128, 128)

    out_path = os.path.join(DATA_DIR, f"sevir_vil_{tag}_subset.npy")
    print(f"  Kaydediliyor: {out_path}")
    np.save(out_path, subset)
    size_gb = os.path.getsize(out_path) / 1e9
    print(f"  Kaydedildi: {size_gb:.2f} GB")
    del subset, data
    gc.collect()

print("\nSubset oluşturma tamamlandı.")
print(f"Dosyalar: {DATA_DIR}")
