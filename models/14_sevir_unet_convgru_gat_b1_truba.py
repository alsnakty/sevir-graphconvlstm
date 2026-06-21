# 14_sevir_unet_convgru_gat_b1_truba.py
#
# S14 B1 — v1'in iki noktada değiştirilmiş hali (her ikisi de compound error
# / blur-to-mean probleminin AR feedback yönünü hedefler):
#   1) SS_EPOCHS: 40 → 60   (teacher→AR geçişi daha geniş tabana yayılır)
#   2) AR_NO_DETACH_STEPS = 2   (yarı-truncated BPTT — ilk 2 AR feedback
#      step'inde gradient AR yolundan da akar; sonrası eski mantık)
#
# DEĞİŞMEYEN: mimari, B-MSE ağırlıkları, linear SS schedule, LR, BATCH, PATIENCE,
#             EPOCHS=80, seed=42. Yani fark sadece eğitim protokolünde.
#
# Çıktı dosya isimleri _b1 suffix ile — v1 sonuçlarının üstüne yazmaz.
#
# an HPC cluster-spesifik (Kaggle notebook'tan farklar):
#   - Yollar /path/to/ altında
#   - matplotlib Agg backend (plt.show() yok)
#   - num_workers=8 (V100 16GB)
#   - AMP, torch.cuda.amp.* API (an HPC cluster modül uyumu)
#   - SMOKE_TEST=1 env ile 1 epoch + 2 batch → ortam doğrulama
#   - seq[::2] __getitem__ içinde (an HPC cluster .npy 25-kare → 13-kare)
#   - subprocess pip install YOK; an HPC cluster modülünde torch_geometric hazır

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib
matplotlib.use('Agg')
import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.nn import GATConv
from skimage.metrics import structural_similarity as sk_ssim

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"

# AYARLAR

DATA_DIR = "/path/to/sevir_data"
OUT_DIR  = "/path/to/outputs"
FIG_DIR  = "/path/to/figures/sevir_s14_unet_convgru_gat_b1"

TRAIN_FILE = os.path.join(DATA_DIR, "sevir_vil_train.npy")
VAL_FILE   = os.path.join(DATA_DIR, "sevir_vil_val.npy")
TEST_FILE  = os.path.join(DATA_DIR, "sevir_vil_test.npy")

MODEL_SAVE_PATH = os.path.join(OUT_DIR, "sevir_s14_unet_convgru_gat_b1.pt")
METRICS_PATH    = os.path.join(OUT_DIR, "sevir_metrics_s14_b1.json")

# ---- Sekans ----
N_INPUT   = 7
N_PREDICT = 6
IMG_SIZE  = 128
GRID_BN   = 16        # bottleneck 16×16 → 256 düğüm

# ---- Mimari (v1 ile birebir) ----
CH_L0     = 16
CH_L1     = 32
CH_L2     = 64
CH_L3     = 128
GAT_HEADS = 4
GAT_ALPHA = 0.3
HOPS      = (1, 2, 4)
SIGMA     = 1.0

# ---- Eğitim ----
EPOCHS     = 1 if SMOKE_TEST else 80
PATIENCE   = 15
LR         = 1e-3
BATCH_SIZE = 4
GRAD_CLIP  = 1.0

# B1 değişiklik #1: scheduled sampling süresi 40 → 60
SS_EPOCHS  = 60

# B1 değişiklik #2: yarı-truncated BPTT — ilk N AR feedback step'i gradient akıtır
# (t < AR_NO_DETACH_STEPS iken pred_in = clamp(...) detach YOK; sonrası detach)
AR_NO_DETACH_STEPS = 2

# ---- Metrik ----
CSI_THRESHOLDS = [16/255, 74/255, 133/255, 160/255, 181/255, 219/255]

# B-MSE çarpanları — v1/v5b ile aynı (B1'de değişmedi)
BMSE_WEIGHTS = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
if SMOKE_TEST:
    print(">>> SMOKE TEST modu aktif (1 epoch, 2 batch) <<<")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# RENK HARİTASI

def get_radar_cmap():
    colors = ['white', '#8ec7ff', 'dodgerblue', 'lime',
              'yellow', 'darkorange', 'red', 'darkmagenta']
    return col.LinearSegmentedColormap.from_list('radar', colors)

CMAP = get_radar_cmap()

# VERİ

class SEVIRDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, max_samples=None):
        print(f"Yükleniyor (memmap): {filepath}")
        self.data = np.load(filepath, mmap_mode='r')
        if max_samples is not None:
            self.data = self.data[:max_samples]
        self.n = self.data.shape[0]
        print(f"  {self.n} örnek  {self.data.shape}")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        seq = self.data[idx].astype(np.float32)
        seq = seq[::2]   # 25 kare → 13 kare (5dk → 10dk, PreDiff protokolü)
        x = torch.from_numpy(seq[:N_INPUT]).unsqueeze(1)
        y = torch.from_numpy(seq[N_INPUT:]).unsqueeze(1)
        return x, y

# MULTI-HOP GRAPH YAPISI (16×16, hops=(1,2,4))

def build_multihop_edges(H=GRID_BN, W=GRID_BN, hops=HOPS):
    """Her düğüm için hop ∈ {1,2,4} mesafedeki 8 yönde komşular.
    Hop'lar çakışmaz (h=1, h=2, h=4 farklı offsetler). Self-loop atlanır.
    """
    edges = []
    for i in range(H):
        for j in range(W):
            node = i * W + j
            for hop in hops:
                for di in (-hop, 0, hop):
                    for dj in (-hop, 0, hop):
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < H and 0 <= nj < W:
                            edges.append([node, ni * W + nj])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def batch_edge_index(edge_index_single, batch_size, num_nodes):
    return torch.cat([edge_index_single + i * num_nodes
                      for i in range(batch_size)], dim=1)


def compute_edge_attr(x_nodes, edge_index, sigma=SIGMA):
    """w_ij = exp(-||h_i - h_j||² / (2σ²)) — her çağrıda yeniden hesaplanır."""
    src, dst = edge_index[0], edge_index[1]
    diff    = x_nodes[src] - x_nodes[dst]
    dist_sq = (diff ** 2).sum(dim=-1, keepdim=True)
    return torch.exp(-dist_sq / (2 * sigma ** 2))

# YAPI BLOKLARI — ConvBlock, DownBlock, UpBlock, ConvGRUCell

class ConvBlock(nn.Module):
    """Conv-GN-LReLU ×2 (U-Net standart)"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        gn = min(8, out_ch)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1),
            nn.GroupNorm(gn, out_ch), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(gn, out_ch), nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x):
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ConvGRUCell(nn.Module):
    """Conv-GRU (Shi+2015 ConvLSTM ailesinin GRU varyantı):
      z = sigmoid(Conv([x, h]))
      r = sigmoid(Conv([x, h]))
      n = tanh(Conv([x, r*h]))
      h' = (1 - z) * n + z * h
    """
    def __init__(self, in_ch, hid_ch, k=3):
        super().__init__()
        pad = k // 2
        self.hid_ch  = hid_ch
        self.conv_zr = nn.Conv2d(in_ch + hid_ch, 2 * hid_ch, k, padding=pad)
        self.conv_n  = nn.Conv2d(in_ch + hid_ch,     hid_ch, k, padding=pad)

    def init_hidden(self, batch_size, H, W, device):
        return torch.zeros(batch_size, self.hid_ch, H, W, device=device)

    def forward(self, x, h):
        zr = torch.sigmoid(self.conv_zr(torch.cat([x, h], dim=1)))
        z, r = zr.chunk(2, dim=1)
        n = torch.tanh(self.conv_n(torch.cat([x, r * h], dim=1)))
        return (1 - z) * n + z * h

# U-Net Encoder/Decoder

class UNetEncoder(nn.Module):
    """128×128 → 16×16 bottleneck. 4 feat döndürür: (L0, L1, L2, L3)."""
    def __init__(self, in_ch=1):
        super().__init__()
        self.enc0 = ConvBlock(in_ch, CH_L0)      # 128, CH_L0
        self.enc1 = DownBlock(CH_L0,  CH_L1)     # 64,  CH_L1
        self.enc2 = DownBlock(CH_L1,  CH_L2)     # 32,  CH_L2
        self.enc3 = DownBlock(CH_L2,  CH_L3)     # 16,  CH_L3

    def forward(self, x):
        f0 = self.enc0(x)
        f1 = self.enc1(f0)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        return f0, f1, f2, f3


class UNetDecoder(nn.Module):
    """16×16 h + 3 skip → 128×128 raw output. sigmoid YOK."""
    def __init__(self, out_ch=1):
        super().__init__()
        self.up2  = UpBlock(CH_L3, CH_L2, CH_L2)
        self.up1  = UpBlock(CH_L2, CH_L1, CH_L1)
        self.up0  = UpBlock(CH_L1, CH_L0, CH_L0)
        self.head = nn.Conv2d(CH_L0, out_ch, 1)

    def forward(self, h, skip0, skip1, skip2):
        x = self.up2(h, skip2)
        x = self.up1(x, skip1)
        x = self.up0(x, skip0)
        return self.head(x)

# MODEL — S14 B1: forward() içinde yarı-truncated BPTT (AR_NO_DETACH_STEPS)

class SEVIRUNetConvGRUGAT(nn.Module):
    """
    S14 mimarisi (v1 ile birebir); B1 farkı sadece forward() AR branch'inde.
    Mimari: U-Net (3-seviye) encoder/decoder + ConvGRU 16×16 bottleneck +
    Multi-hop GAT (1,2,4 hop) residual yan-akım (α=GAT_ALPHA).

    Eğitim protokolü farkı (B1):
      - Scheduled sampling: training'de p_teacher ile y_true feedback.
      - AR feedback (TF seçilmediği step'lerde):
          if t < self.ar_no_detach_steps:
              pred_in = pred_raw.clamp(0, 1)            # gradient akar (no detach)
          else:
              pred_in = pred_raw.clamp(0, 1).detach()   # truncated BPTT (v1 default)
      - Eval'de gradient zaten kapalı; AR_NO_DETACH_STEPS sadece training'de etkili.
    """
    def __init__(self):
        super().__init__()
        self.encoder = UNetEncoder(in_ch=1)
        self.decoder = UNetDecoder(out_ch=1)
        self.convgru = ConvGRUCell(in_ch=CH_L3, hid_ch=CH_L3, k=3)

        self.gat1 = GATConv(CH_L3, CH_L3 // GAT_HEADS,
                            heads=GAT_HEADS, concat=True, edge_dim=1)
        self.gat_norm = nn.LayerNorm(CH_L3)
        self.gat2 = GATConv(CH_L3, CH_L3 // GAT_HEADS,
                            heads=GAT_HEADS, concat=True, edge_dim=1)

        self.register_buffer('edge_index_single', build_multihop_edges())
        self.ss_prob = 1.0                              # epoch başında set edilir
        self.ar_no_detach_steps = AR_NO_DETACH_STEPS    # B1: yarı-truncated BPTT

    def _gat_branch(self, h, B, edge_index):
        if GAT_ALPHA == 0:
            return h
        N = GRID_BN * GRID_BN
        nodes = h.flatten(2).permute(0, 2, 1).reshape(B * N, CH_L3)

        ea1   = compute_edge_attr(nodes, edge_index)
        nodes = F.leaky_relu(self.gat1(nodes, edge_index, edge_attr=ea1), 0.2)
        nodes = self.gat_norm(nodes)

        ea2   = compute_edge_attr(nodes, edge_index)
        nodes = F.leaky_relu(self.gat2(nodes, edge_index, edge_attr=ea2), 0.2)

        h_gat = nodes.reshape(B, N, CH_L3).permute(0, 2, 1) \
                     .reshape(B, CH_L3, GRID_BN, GRID_BN)
        return h + GAT_ALPHA * h_gat

    def forward(self, x_in, y_true=None):
        """x_in: (B, T_in, 1, 128, 128); y_true: (B, T_pred, 1, 128, 128) ya da None.
        return: (B, T_pred, 1, 128, 128)
        """
        B, T_in, _, H, W = x_in.shape
        edge_index = batch_edge_index(self.edge_index_single, B,
                                      GRID_BN * GRID_BN).to(x_in.device)

        skip0_sum = torch.zeros(B, CH_L0, IMG_SIZE,    IMG_SIZE,    device=x_in.device)
        skip1_sum = torch.zeros(B, CH_L1, IMG_SIZE//2, IMG_SIZE//2, device=x_in.device)
        skip2_sum = torch.zeros(B, CH_L2, IMG_SIZE//4, IMG_SIZE//4, device=x_in.device)

        h = self.convgru.init_hidden(B, GRID_BN, GRID_BN, x_in.device)

        for t in range(T_in):
            f0, f1, f2, f3 = self.encoder(x_in[:, t])
            skip0_sum = skip0_sum + f0
            skip1_sum = skip1_sum + f1
            skip2_sum = skip2_sum + f2
            h = self.convgru(f3, h)
            h = self._gat_branch(h, B, edge_index)

        skip0_avg = skip0_sum / T_in
        skip1_avg = skip1_sum / T_in
        skip2_avg = skip2_sum / T_in

        pred_in = x_in[:, -1]
        outputs = []
        for t in range(N_PREDICT):
            f0, f1, f2, f3 = self.encoder(pred_in)
            h = self.convgru(f3, h)
            h = self._gat_branch(h, B, edge_index)

            pred_raw = self.decoder(h, skip0_avg, skip1_avg, skip2_avg)
            outputs.append(pred_raw.unsqueeze(1))

            # Scheduled sampling — sadece training + y_true varsa
            if self.training and y_true is not None and t < y_true.shape[1] - 1:
                if torch.rand(1).item() < self.ss_prob:
                    pred_in = y_true[:, t]                    # teacher forcing
                else:
                    # B1: yarı-truncated BPTT
                    if t < self.ar_no_detach_steps:
                        pred_in = pred_raw.clamp(0, 1)        # gradient akar
                    else:
                        pred_in = pred_raw.clamp(0, 1).detach()  # truncated
            else:
                pred_in = pred_raw.clamp(0, 1).detach() if not self.training \
                          else pred_raw.clamp(0, 1)

        return torch.cat(outputs, dim=1)

# LOSS — Balanced MSE (B1'de değişmedi)

def bmse_loss(pred, target):
    w = torch.ones_like(target)
    w = torch.where(target >= 16/255,  torch.full_like(w, BMSE_WEIGHTS[1]), w)
    w = torch.where(target >= 74/255,  torch.full_like(w, BMSE_WEIGHTS[2]), w)
    w = torch.where(target >= 133/255, torch.full_like(w, BMSE_WEIGHTS[3]), w)
    w = torch.where(target >= 160/255, torch.full_like(w, BMSE_WEIGHTS[4]), w)
    w = torch.where(target >= 181/255, torch.full_like(w, BMSE_WEIGHTS[5]), w)
    w = torch.where(target >= 219/255, torch.full_like(w, BMSE_WEIGHTS[6]), w)
    return (w * (pred - target) ** 2).mean()

# METRİKLER

def compute_csi(pred_np, true_np, threshold):
    p, t = pred_np >= threshold, true_np >= threshold
    tp, fp, fn = (p & t).sum(), (p & ~t).sum(), (~p & t).sum()
    d = tp + fp + fn
    return float(tp / d) if d > 0 else 0.0


def _max_pool_np(arr, pool_size):
    t = torch.from_numpy(arr).float().unsqueeze(1)
    pooled = F.max_pool2d(t, kernel_size=pool_size, stride=pool_size)
    return pooled.squeeze(1).numpy()


def compute_csi_pool(pred_flat, true_flat, threshold, pool_size):
    return compute_csi(_max_pool_np(pred_flat, pool_size),
                       _max_pool_np(true_flat, pool_size), threshold)


def evaluate_metrics(model, loader):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(DEVICE)).cpu().numpy()
            pred = np.clip(pred, 0.0, 1.0)
            all_pred.append(pred[:, :, 0])
            all_true.append(yb.numpy()[:, :, 0])
    pred_all = np.concatenate(all_pred)
    true_all = np.concatenate(all_true)
    N, T_pred, H, W = pred_all.shape

    ssim_matrix = np.zeros((N, T_pred), dtype=np.float64)
    for i in range(N):
        for t in range(T_pred):
            ssim_matrix[i, t] = sk_ssim(
                pred_all[i, t], true_all[i, t], data_range=1.0)

    per_lt_mse, per_lt_ssim, per_lt_csi_m, per_lt_csi_per_thr = [], [], [], []
    for t in range(T_pred):
        p_t, y_t = pred_all[:, t], true_all[:, t]
        mse_t = float(np.mean((p_t - y_t) ** 2))
        per_lt_mse.append(mse_t)
        per_lt_ssim.append(float(ssim_matrix[:, t].mean()))
        csi_t = [compute_csi(p_t, y_t, thr) for thr in CSI_THRESHOLDS]
        per_lt_csi_per_thr.append([round(c, 4) for c in csi_t])
        per_lt_csi_m.append(float(np.mean(csi_t)))

    pred_flat = pred_all.reshape(-1, H, W)
    true_flat = true_all.reshape(-1, H, W)
    csi_scores = [compute_csi(pred_flat, true_flat, thr) for thr in CSI_THRESHOLDS]
    csi_pool4  = [compute_csi_pool(pred_flat, true_flat, thr, 4)  for thr in CSI_THRESHOLDS]
    csi_pool16 = [compute_csi_pool(pred_flat, true_flat, thr, 16) for thr in CSI_THRESHOLDS]
    mse = float(np.mean((pred_all - true_all) ** 2))

    return {
        "mse": mse, "rmse": float(np.sqrt(mse)),
        "ssim": float(ssim_matrix.mean()),
        "csi_m": float(np.mean(csi_scores)),
        "csi_per_threshold": [round(c, 4) for c in csi_scores],
        "csi_pool4_m":  float(np.mean(csi_pool4)),
        "csi_pool4_per_threshold":  [round(c, 4) for c in csi_pool4],
        "csi_pool16_m": float(np.mean(csi_pool16)),
        "csi_pool16_per_threshold": [round(c, 4) for c in csi_pool16],
        "per_leadtime": {
            "mse":  [round(float(m), 6) for m in per_lt_mse],
            "rmse": [round(float(np.sqrt(m)), 6) for m in per_lt_mse],
            "ssim": [round(float(s), 4) for s in per_lt_ssim],
            "csi_m": [round(float(c), 4) for c in per_lt_csi_m],
            "csi_per_threshold": per_lt_csi_per_thr,
        },
    }

# EĞİTİM

def train_model(model, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=5, factor=0.5, min_lr=1e-6)
    criterion = bmse_loss
    scaler    = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))
    best_val   = float('inf')
    no_improve = 0
    history    = {"train_loss": [], "val_loss": [],
                  "train_rmse": [], "val_rmse": [], "ss_prob": []}

    train_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        # Linear scheduled sampling — 1.0 → 0.0 over SS_EPOCHS=60
        model.ss_prob = max(0.0, 1.0 - (epoch - 1) / SS_EPOCHS)

        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_batches = 0
        n_train_seen = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                pred = model(xb, y_true=yb)
                loss = criterion(pred, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * xb.size(0)
            n_train_seen += xb.size(0)
            n_batches += 1
            if SMOKE_TEST and n_batches >= 2:
                print(f"  Smoke test: {n_batches} batch tamamlandı, forward/backward OK")
                break
        train_loss /= max(n_train_seen, 1)

        model.eval()
        val_loss  = 0.0
        n_batches = 0
        n_val_seen = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                    # eval'da y_true verilmez → saf autoregressive
                    val_loss += criterion(model(xb), yb).item() * xb.size(0)
                n_val_seen += xb.size(0)
                n_batches += 1
                if SMOKE_TEST and n_batches >= 2:
                    break
        val_loss /= max(n_val_seen, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_rmse"].append(float(np.sqrt(train_loss)))
        history["val_rmse"].append(float(np.sqrt(val_loss)))
        history["ss_prob"].append(model.ss_prob)
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        total   = (time.time() - train_start) / 60
        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"ss={model.ss_prob:.2f}  "
              f"Train BMSE: {train_loss:.5f}  sqrt: {np.sqrt(train_loss):.4f}  |  "
              f"Val BMSE: {val_loss:.5f}  sqrt: {np.sqrt(val_loss):.4f}  "
              f"[{elapsed:.0f}s / {total:.1f}dk]")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"EarlyStopping tetiklendi ({PATIENCE} epoch).")
                break

    print(f"\nEn iyi Val BMSE: {best_val:.5f}  |  Toplam: {(time.time()-train_start)/60:.1f} dk")
    print(f"Not: train/val 'loss' B-MSE skalasında; plain MSE/RMSE evaluate_metrics çıktısında.")
    return history

# GÖRSELLEŞTİRME

def plot_training_history(history):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(history["train_loss"], label="Train BMSE")
    axes[0].plot(history["val_loss"],   label="Val BMSE")
    axes[0].set_title("B-MSE Loss"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(history["train_rmse"], label="Train sqrt(BMSE)")
    axes[1].plot(history["val_rmse"],   label="Val sqrt(BMSE)")
    axes[1].set_title("sqrt(B-MSE)"); axes[1].legend(); axes[1].grid(True)
    axes[2].plot(history["ss_prob"], label="p_teacher", color='purple')
    axes[2].set_title("Scheduled Sampling p_teacher (SS_EPOCHS=60)")
    axes[2].legend(); axes[2].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "training_history_s14_b1.png"), dpi=150)
    plt.close()


def save_comparison_gif(true_frames, pred_frames, fps=3):
    frames = []
    for i in range(len(true_frames)):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(true_frames[i], cmap=CMAP, vmin=0, vmax=1)
        axes[0].set_title(f"Gerçek t+{(i+1)*10}dk"); axes[0].axis("off")
        axes[1].imshow(pred_frames[i], cmap=CMAP, vmin=0, vmax=1)
        axes[1].set_title(f"Tahmin t+{(i+1)*10}dk"); axes[1].axis("off")
        plt.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3])
        plt.close(fig)
    path = os.path.join(FIG_DIR, "sevir_s14_b1_comparison.gif")
    imageio.mimsave(path, frames, fps=fps)
    print(f"GIF kaydedildi: {path}")

# ANA AKIŞ

def _seed_worker(worker_id):
    ws = 42 + worker_id
    np.random.seed(ws)
    random.seed(ws)


if __name__ == "__main__":

    assert os.path.exists(TRAIN_FILE), f"Veri bulunamadı: {TRAIN_FILE}"
    assert os.path.exists(VAL_FILE),   f"Veri bulunamadı: {VAL_FILE}"
    if not SMOKE_TEST:
        assert os.path.exists(TEST_FILE), f"Veri bulunamadı: {TEST_FILE}"

    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    g = torch.Generator()
    g.manual_seed(42)

    max_samples = 16 if SMOKE_TEST else None
    train_ds = SEVIRDataset(TRAIN_FILE, max_samples=max_samples)
    val_ds   = SEVIRDataset(VAL_FILE,   max_samples=max_samples)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=8, pin_memory=True,
                              worker_init_fn=_seed_worker, generator=g)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=8, pin_memory=True,
                              worker_init_fn=_seed_worker, generator=g)

    if not SMOKE_TEST:
        test_ds = SEVIRDataset(TEST_FILE, max_samples=max_samples)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=8, pin_memory=True,
                                  worker_init_fn=_seed_worker, generator=g)

    # Defansif shape kontrolü
    _xb, _yb = next(iter(train_loader))
    print(f"Batch shape kontrolü — x: {tuple(_xb.shape)}  y: {tuple(_yb.shape)}")
    assert _xb.shape[1:] == (N_INPUT, 1, IMG_SIZE, IMG_SIZE), (
        f"Beklenmedik x shape: {tuple(_xb.shape)} — .npy ham 25-kare mi?")
    assert _yb.shape[1:] == (N_PREDICT, 1, IMG_SIZE, IMG_SIZE), (
        f"Beklenmedik y shape: {tuple(_yb.shape)}")
    del _xb, _yb

    model = SEVIRUNetConvGRUGAT().to(DEVICE)

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Toplam parametre: {total:,}")
    print(f"S14 B1 — U-Net + ConvGRU bottleneck + Multi-hop GAT residual")
    print(f"  GAT_ALPHA={GAT_ALPHA}  HOPS={HOPS}  CH=[{CH_L0},{CH_L1},{CH_L2},{CH_L3}]  GRID_BN={GRID_BN}")
    print(f"  SS_EPOCHS={SS_EPOCHS} (v1: 40)  AR_NO_DETACH_STEPS={AR_NO_DETACH_STEPS} (v1: 0 → hep detach)")
    print(f"  BMSE_WEIGHTS={BMSE_WEIGHTS} (B1'de değişmedi)\n")

    history = train_model(model, train_loader, val_loader)

    if SMOKE_TEST:
        print("\n>>> SMOKE TEST BAŞARILI — ortam doğrulandı <<<")
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    CUDA OK, torch_geometric OK, AMP OK, forward/backward OK")
        print(f"    Multi-hop edges: {model.edge_index_single.shape[1]} kenar")
        print(f"    ss_prob ep1: {history['ss_prob'][0]:.4f} (beklenen 1.0)")
        print(f"    AR_NO_DETACH_STEPS aktif: model.ar_no_detach_steps={model.ar_no_detach_steps}")
    else:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
        print("\nMetrikler hesaplanıyor (val)...")
        val_metrics  = evaluate_metrics(model, val_loader)
        print("Metrikler hesaplanıyor (test)...")
        test_metrics = evaluate_metrics(model, test_loader)

        out = {
            "model":         "SEVIRUNetConvGRUGAT_S14_B1",
            "loss":          "B-MSE",
            "bmse_weights":  BMSE_WEIGHTS,
            "architecture":  "U-Net (3-level) + ConvGRU bottleneck + Multi-hop GAT residual branch",
            "b1_changes_vs_v1": {
                "SS_EPOCHS":          {"v1": 40, "b1": SS_EPOCHS},
                "AR_NO_DETACH_STEPS": {"v1": 0,  "b1": AR_NO_DETACH_STEPS},
            },
            "config": {
                "N_INPUT": N_INPUT, "N_PREDICT": N_PREDICT,
                "GRID_BN": GRID_BN, "HOPS": list(HOPS),
                "CH": [CH_L0, CH_L1, CH_L2, CH_L3],
                "GAT_HEADS": GAT_HEADS, "GAT_ALPHA": GAT_ALPHA,
                "BATCH_SIZE": BATCH_SIZE, "EPOCHS": EPOCHS, "PATIENCE": PATIENCE,
                "LR": LR, "SS_EPOCHS": SS_EPOCHS,
                "AR_NO_DETACH_STEPS": AR_NO_DETACH_STEPS, "SIGMA": SIGMA,
            },
            "epochs_ran":    len(history["train_loss"]),
            "best_val_bmse": round(min(history["val_loss"]), 6),
            "val":  val_metrics,
            "test": test_metrics,
        }

        with open(METRICS_PATH, "w") as f:
            json.dump(out, f, indent=2)

        print(f"\n--- S14 B1 (an HPC cluster full data) Sonuçlar ---")
        print(f"  Best Val BMSE : {out['best_val_bmse']:.6f}")
        print(f"  Epoch         : {out['epochs_ran']}")
        print(f"  -- Validation --")
        print(f"  MSE        : {val_metrics['mse']:.6f}")
        print(f"  RMSE       : {val_metrics['rmse']:.6f}")
        print(f"  SSIM       : {val_metrics['ssim']:.4f}")
        print(f"  CSI-M      : {val_metrics['csi_m']:.4f}")
        print(f"  CSI-pool4  : {val_metrics['csi_pool4_m']:.4f}")
        print(f"  CSI-pool16 : {val_metrics['csi_pool16_m']:.4f}")
        print(f"  -- Test (held-out, time_utc >= 2019-06-01) --")
        print(f"  MSE        : {test_metrics['mse']:.6f}")
        print(f"  RMSE       : {test_metrics['rmse']:.6f}")
        print(f"  SSIM       : {test_metrics['ssim']:.4f}")
        print(f"  CSI-M      : {test_metrics['csi_m']:.4f}  (Kaggle S14 v1: 0.3941)")
        print(f"  CSI-pool4  : {test_metrics['csi_pool4_m']:.4f}  (Kaggle S14 v1: 0.4544 | ConvLSTM lit: 0.4452)")
        print(f"  CSI-pool16 : {test_metrics['csi_pool16_m']:.4f}  (Kaggle S14 v1: 0.5283 | ConvLSTM lit: 0.5135)")
        print(f"  CSI@thr=181/255 : {test_metrics['csi_per_threshold'][4]:.4f}  (Kaggle S14 v1: 0.2699)")
        print(f"  CSI@thr=219/255 : {test_metrics['csi_per_threshold'][5]:.4f}  (Kaggle S14 v1: 0.1172)")

        plot_training_history(history)

        sample_x, sample_y = val_ds[0]
        model.eval()
        with torch.no_grad():
            pred = model(sample_x.unsqueeze(0).to(DEVICE))[0].cpu().numpy()
            pred = np.clip(pred, 0.0, 1.0)

        save_comparison_gif(sample_y[:, 0].numpy(), pred[:, 0])
        print(f"\nTamamlandı.\n  Model: {MODEL_SAVE_PATH}\n  Figürler: {FIG_DIR}")
