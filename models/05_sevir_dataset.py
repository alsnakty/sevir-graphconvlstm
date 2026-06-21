# 05_sevir_dataset.py
#
# SEVIR VIL verisi için event-bazlı train/val/test pipeline.
# - PreDiff/SEVIR standardı: event-bazlı tarih cutoff (2019-06-01)
# - Catalog filter: pct_missing == 0 (PreDiff sevir_dataloader.py default)
# - Train+val: time_utc < 2019-06-01 → sequence-level random %80/%20 (seed=42)
# - Test: time_utc >= 2019-06-01
# - Native 384x384 → 128x128 (3x3 max-pool, PreDiff protokolü)
# - Her event 49 kareden 25-karelik diziler (stride=12)
# - Eğitimde seq[::2] ile 13 kare, T=7 input + L=6 target (10dk aralık)

import gc
import os
import datetime

import numpy as np
import pandas as pd
import h5py
from skimage.measure import block_reduce

# AYARLAR

DATA_ROOT  = r"/path/to/data_root"
RAW_DATA_DIR = os.path.join(DATA_ROOT, "data")           # data/vil/{2017,2018,2019}/*.h5
OUT_DIR    = os.path.join(DATA_ROOT, "data", "sevir_processed")

# Catalog: 02_veri_indir2.py tarafından üretilir, proje dizininde
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CATALOG_PATH = os.path.join(SCRIPT_DIR, "benim_tez_katalogum.csv")

# PreDiff/SEVIR resmi split (Veillette 2020 + PreDiff 2023)
TRAIN_TEST_SPLIT_DATE = datetime.datetime(2019, 6, 1)
VAL_RATIO = 0.2
SEED      = 42

# Sequence parametreleri
IMG_SIZE  = 128                # SEVIR-LR (PreDiff protokolü)
RAW_SIZE  = 384                # SEVIR native
SEQ_LEN   = 25                 # depolama window'u (eğitimde seq[::2] ile 13 olur)
STRIDE    = 12                 # overlapping pencere kaydırma
RAW_T     = 49                 # raw event frame sayısı

# YARDIMCI FONKSİYONLAR

def downsample(frames, size=IMG_SIZE):
    """
    frames: (T, RAW_SIZE, RAW_SIZE) uint8
    Dönüş: (T, size, size) float32 [0,1]

    PreDiff protokolü: 3x3 max-pool, /255 normalize.
    Kaynak: PreDiff sevir_dataloader.py PREPROCESS_SCALE_01 (vil: 1/255).
    Missing handling: catalog seviyesinde pct_missing == 0 ile temizleniyor
    (PreDiff catalog_filter='default'), pixel seviyesinde işlem yapmıyoruz.
    """
    factor = frames.shape[1] // size           # 384//128 = 3
    out = np.zeros((frames.shape[0], size, size), dtype=np.float32)
    for t in range(frames.shape[0]):
        pooled = block_reduce(frames[t], block_size=(factor, factor), func=np.max)
        out[t] = pooled.astype(np.float32) / 255.0
    return out


def extract_sequences(frames_norm, seq_len=SEQ_LEN, stride=STRIDE):
    """
    frames_norm: (T, H, W) float32 (tek event)
    Dönüş: list of (seq_len, H, W)
    """
    T = frames_norm.shape[0]
    return [frames_norm[s : s + seq_len]
            for s in range(0, T - seq_len + 1, stride)]


def seqs_per_event(T=RAW_T, seq_len=SEQ_LEN, stride=STRIDE):
    return len(range(0, T - seq_len + 1, stride))


# CATALOG OKUMA + EVENT-BAZLI SPLIT

def read_catalog(path=CATALOG_PATH):
    """
    Filtreli catalog oku (sadece vil event'leri, 02_veri_indir2.py üretti).
    time_utc'yi datetime'a parse et.
    PreDiff hizalaması: pct_missing == 0 olan event'ler (sevir_dataloader.py
    catalog_filter='default').
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Catalog bulunamadı: {path}\n"
            f"Önce 02_veri_indir2.py çalıştırın."
        )
    df = pd.read_csv(path, low_memory=False)
    n0 = len(df)
    # Sadece vil olduğunu doğrula
    df = df[df['img_type'] == 'vil'].copy()
    # PreDiff catalog_filter='default': pct_missing == 0
    if 'pct_missing' not in df.columns:
        raise KeyError("Catalog'da 'pct_missing' sütunu yok — 02 yeniden çalıştır.")
    n_before_pct = len(df)
    df = df[df['pct_missing'] == 0].copy()
    n_dropped_pct = n_before_pct - len(df)
    df['time_utc'] = pd.to_datetime(df['time_utc'], errors='coerce')
    n_before_time = len(df)
    df = df.dropna(subset=['time_utc'])
    n_dropped_time = n_before_time - len(df)
    print(f"Catalog: {n0} satır → {n_before_pct} (vil) → "
          f"{n_before_time} (pct_missing==0, -{n_dropped_pct}) → "
          f"{len(df)} (time_utc parse, -{n_dropped_time})")
    return df.reset_index(drop=True)


def split_events_by_date(catalog, cutoff=TRAIN_TEST_SPLIT_DATE):
    """
    Event'leri tarihsel cutoff'a göre train+val ve test'e ayır.
    """
    cutoff_pd = pd.Timestamp(cutoff)
    train_val = catalog[catalog['time_utc'] < cutoff_pd].copy()
    test      = catalog[catalog['time_utc'] >= cutoff_pd].copy()
    print(f"Train+val event sayısı (time_utc < {cutoff.date()}): {len(train_val)}")
    print(f"Test event sayısı       (time_utc >= {cutoff.date()}): {len(test)}")
    return train_val, test


# POOL'DAN MEMMAP YAZIM (file_name bazlı sıralı okuma — h5 bir kez açılır)

def build_pool_memmap(pool_catalog, pool_name, out_path):
    """
    Bir event havuzunu (train_val veya test) memmap'e yaz.
    Catalog file_name ile sıralanır; aynı h5'teki event'ler ardarda işlenir
    (dosya bir kez açılır, performans için kritik).
    Dönüş: actual_n_sequences
    """
    expected_total = len(pool_catalog) * seqs_per_event()
    print(f"\n[{pool_name}] {len(pool_catalog)} event, ~{expected_total} dizi")
    print(f"  Tahmini boyut: {expected_total * SEQ_LEN * IMG_SIZE * IMG_SIZE * 4 / 1e9:.2f} GB")

    # Catalog'u dosya bazında sırala — aynı h5'te ardışık event'ler aynı handle ile okunur
    pool_sorted = pool_catalog.sort_values(['file_name', 'file_index']).reset_index(drop=True)

    mmap = np.lib.format.open_memmap(
        out_path, mode='w+', dtype=np.float32,
        shape=(expected_total, SEQ_LEN, IMG_SIZE, IMG_SIZE)
    )

    idx = 0
    current_h5_path = None
    current_h5 = None
    try:
        for _, row in pool_sorted.iterrows():
            rel_path = row['file_name']
            file_index = int(row['file_index'])
            h5_path = os.path.join(RAW_DATA_DIR, rel_path)

            if h5_path != current_h5_path:
                # Dosya değişti: eskiyi kapat, yeniyi aç
                if current_h5 is not None:
                    current_h5.close()
                    current_h5 = None
                if not os.path.exists(h5_path):
                    print(f"  UYARI: dosya yok, atlandı: {os.path.basename(rel_path)}")
                    current_h5_path = h5_path
                    continue
                current_h5 = h5py.File(h5_path, 'r')
                current_h5_path = h5_path
                print(f"  açıldı: {os.path.basename(rel_path)}", flush=True)

            if current_h5 is None:
                continue   # bu h5 dosyası eksik, atla

            # Event oku → downsample → sequences çıkar
            frames = current_h5['vil'][file_index].transpose(2, 0, 1)
            frames_norm = downsample(frames, IMG_SIZE)
            for seq in extract_sequences(frames_norm):
                mmap[idx] = seq
                idx += 1
    finally:
        if current_h5 is not None:
            current_h5.close()

    mmap.flush()
    print(f"[{pool_name}] tamamlandı: {idx} / {expected_total} dizi yazıldı.")
    return idx


def trim_memmap_chunked(path, n_actual, chunk=500):
    """
    Memmap'i n_actual boyutuna kes (RAM güvenli, chunked write).
    """
    src_mmap = np.load(path, mmap_mode='r')
    expected = src_mmap.shape[0]
    if n_actual >= expected:
        del src_mmap
        return

    tmp_path = path + ".trim"
    dst_mmap = np.lib.format.open_memmap(
        tmp_path, mode='w+', dtype=np.float32,
        shape=(n_actual,) + src_mmap.shape[1:]
    )
    for start in range(0, n_actual, chunk):
        end = min(start + chunk, n_actual)
        dst_mmap[start:end] = src_mmap[start:end]
    dst_mmap.flush()
    del src_mmap, dst_mmap
    gc.collect()
    os.remove(path)
    os.rename(tmp_path, path)
    print(f"  trim: {expected} → {n_actual} ({path})")


# ANA AKIŞ

def build_dataset():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Catalog oku ve event'leri böl
    print("=" * 70)
    print("Catalog okunuyor + event-bazlı tarih split (PreDiff/SEVIR standardı)")
    print("=" * 70)
    catalog = read_catalog()
    print(f"Toplam vil event satırı: {len(catalog)}")

    train_val_cat, test_cat = split_events_by_date(catalog, TRAIN_TEST_SPLIT_DATE)

    # 2. Train+val pool memmap'e yaz
    tv_tmp_path = os.path.join(OUT_DIR, "_tmp_trainval.npy")
    n_tv = build_pool_memmap(train_val_cat, "TRAIN+VAL", tv_tmp_path)

    # 2a. Memmap'i gerçek boyuta indir (eksik h5 vs nedeniyle expected > actual ise)
    trim_memmap_chunked(tv_tmp_path, n_tv)

    # 3. Train+val'i sequence-level random %80/%20 split (seed=42)
    print(f"\nTrain+val pool: {n_tv} dizi → random split %{int((1-VAL_RATIO)*100)}/%{int(VAL_RATIO*100)} (seed={SEED})")
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_tv)
    n_train = int((1 - VAL_RATIO) * n_tv)
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:]
    print(f"  Train: {len(train_idx)} dizi | Val: {len(val_idx)} dizi")

    # 3a. Train.npy ve val.npy yaz
    train_path = os.path.join(OUT_DIR, "sevir_vil_train.npy")
    val_path   = os.path.join(OUT_DIR, "sevir_vil_val.npy")

    tv_mmap = np.load(tv_tmp_path, mmap_mode='r')
    CHUNK = 500

    print("Train kaydediliyor...")
    train_mmap = np.lib.format.open_memmap(
        train_path, mode='w+', dtype=np.float32,
        shape=(len(train_idx), SEQ_LEN, IMG_SIZE, IMG_SIZE)
    )
    for start in range(0, len(train_idx), CHUNK):
        end = min(start + CHUNK, len(train_idx))
        train_mmap[start:end] = tv_mmap[train_idx[start:end]]
        if start % (CHUNK * 20) == 0:
            print(f"  Train: {end}/{len(train_idx)}", flush=True)
    train_mmap.flush()
    del train_mmap
    gc.collect()

    print("Val kaydediliyor...")
    val_mmap = np.lib.format.open_memmap(
        val_path, mode='w+', dtype=np.float32,
        shape=(len(val_idx), SEQ_LEN, IMG_SIZE, IMG_SIZE)
    )
    for start in range(0, len(val_idx), CHUNK):
        end = min(start + CHUNK, len(val_idx))
        val_mmap[start:end] = tv_mmap[val_idx[start:end]]
    val_mmap.flush()
    del val_mmap, tv_mmap
    gc.collect()

    # 3b. Geçici dosyayı sil
    os.remove(tv_tmp_path)

    # 4. Test pool memmap'e yaz (doğrudan, split yok)
    test_path = os.path.join(OUT_DIR, "sevir_vil_test.npy")
    n_test = build_pool_memmap(test_cat, "TEST", test_path)

    # 4a. Test memmap'i gerçek boyuta trim et (RAM güvenli)
    trim_memmap_chunked(test_path, n_test)

    # 5. Özet
    print("\n" + "=" * 70)
    print("TAMAMLANDI")
    print("=" * 70)
    print(f"  Train: ({len(train_idx)}, {SEQ_LEN}, {IMG_SIZE}, {IMG_SIZE})  → {train_path}")
    print(f"  Val  : ({len(val_idx)},  {SEQ_LEN}, {IMG_SIZE}, {IMG_SIZE})  → {val_path}")
    print(f"  Test : ({n_test}, {SEQ_LEN}, {IMG_SIZE}, {IMG_SIZE})  → {test_path}")
    print(f"\n  Cutoff: {TRAIN_TEST_SPLIT_DATE.date()} (PreDiff/SEVIR standart)")
    print(f"  Val_ratio: {VAL_RATIO}, seed: {SEED}")


if __name__ == "__main__":
    build_dataset()
