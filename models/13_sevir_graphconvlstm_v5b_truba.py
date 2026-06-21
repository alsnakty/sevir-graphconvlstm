# 13_sevir_graphconvlstm_v5b_truba.py
#
# an HPC cluster uyarlaması — 13_sevir_graphconvlstm_v5b_kaggle.ipynb'den farklar:
#   - Veri/çıktı yolları /path/to/ altında
#   - matplotlib Agg backend (plt.show() yok)
#   - num_workers=8 (V100 16GB)
#   - AMP (mixed precision), torch.cuda.amp.* API (an HPC cluster modül uyumu)
#   - SMOKE_TEST=1 env var ile 1 epoch + 2 batch → ortam doğrulama
#   - SEVIRDataset.__getitem__ içinde seq[::2]: an HPC cluster .npy ham 25-kare (5dk
#     aralık) tutuluyor; Kaggle subset .npy zaten 13-kareye indirilmiş halde
#     (06_sevir_subset_olustur.py içinde [::2] uygulandı). Bu satır iki tarafı
#     loop'a aynı (N, 13, 128, 128) shape ile sokar — PreDiff 10dk protokolü.
#
# Mimari + loss + hyperparam KAGGLE İLE BİREBİR:
#   - GroupNorm (BatchNorm değil)
#   - Decoder'da sigmoid YOK; raw output + autoregressive feedback'te clamp(0,1)
#   - B-MSE loss (6 kademeli ağırlık; CSI threshold'larıyla aligned)
#   - BATCH_SIZE=4, EPOCHS=100, PATIENCE=15, LR=1e-3
#   - Hem train hem val B-MSE; evaluate_metrics plain MSE/SSIM/CSI raporlar

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib
matplotlib.use('Agg')
import json
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

DATA_DIR  = "/path/to/sevir_data"
OUT_DIR   = "/path/to/outputs"
FIG_DIR   = "/path/to/figures/sevir_graphconvlstm_v5b"

TRAIN_FILE = os.path.join(DATA_DIR, "sevir_vil_train.npy")
VAL_FILE   = os.path.join(DATA_DIR, "sevir_vil_val.npy")
TEST_FILE  = os.path.join(DATA_DIR, "sevir_vil_test.npy")

MODEL_SAVE_PATH = os.path.join(OUT_DIR, "sevir_graphconvlstm_v5b_model.pt")
METRICS_PATH    = os.path.join(OUT_DIR, "sevir_metrics_graphconvlstm_v5b.json")

N_INPUT    = 7
N_PREDICT  = 6
IMG_SIZE   = 128
GRID_SIZE  = 32

FEAT_DIM   = 64
HIDDEN_DIM = 64
GAT_HEADS  = 4
N_LAYERS   = 2
SIGMA      = 1.0

EPOCHS     = 1 if SMOKE_TEST else 100
PATIENCE   = 15
LR         = 1e-3
BATCH_SIZE = 4

CSI_THRESHOLDS = [16/255, 74/255, 133/255, 160/255, 181/255, 219/255]

# B-MSE çarpanları — bin sınırları CSI_THRESHOLDS ile aynı.
# Index karşılığı: [<16, 16-74, 74-133, 133-160, 160-181, 181-219, >=219]
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

# GRAPH YAPISI

def build_grid_edges(H=GRID_SIZE, W=GRID_SIZE):
    edges = []
    for i in range(H):
        for j in range(W):
            node = i * W + j
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
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
    """
    w = exp(-||h_i - h_j||² / 2σ²)
    Her adımda encoder çıktısından hesaplanır → dinamik.
    """
    src, dst = edge_index[0], edge_index[1]
    diff    = x_nodes[src] - x_nodes[dst]
    dist_sq = (diff ** 2).sum(dim=-1, keepdim=True)
    return torch.exp(-dist_sq / (2 * sigma ** 2))

# CNN ENCODER / DECODER  — GroupNorm (v5/v5b)

class CNNEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=FEAT_DIM):
        super().__init__()
        mid = out_channels // 2  # 32
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1),
            nn.GroupNorm(8, mid), nn.LeakyReLU(0.2),
            nn.Conv2d(mid, mid, 3, padding=1),
            nn.GroupNorm(8, mid), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            nn.Conv2d(mid, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels), nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.net(x)


class CNNDecoder(nn.Module):
    def __init__(self, in_channels=HIDDEN_DIM + FEAT_DIM, out_channels=1):
        super().__init__()
        mid = HIDDEN_DIM // 2  # 32
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels, mid, 2, stride=2),
            nn.GroupNorm(8, mid), nn.LeakyReLU(0.2),
            nn.Conv2d(mid, mid, 3, padding=1),
            nn.GroupNorm(8, mid), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(mid, mid // 2, 2, stride=2),
            nn.GroupNorm(4, mid // 2), nn.LeakyReLU(0.2),
            nn.Conv2d(mid // 2, out_channels, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)

# GRAPHCONVLSTM HÜCRESİ

class GraphConvLSTMCell(nn.Module):
    def __init__(self, in_feat, hidden_dim=HIDDEN_DIM, heads=GAT_HEADS):
        super().__init__()
        assert in_feat == hidden_dim, (
            f"Residual connection h_next + x requires in_feat == hidden_dim, "
            f"got in_feat={in_feat}, hidden_dim={hidden_dim}"
        )
        self.hidden_dim = hidden_dim
        self.gat = GATConv(
            in_channels  = in_feat + hidden_dim,
            out_channels = (4 * hidden_dim) // heads,
            heads        = heads,
            concat       = True,
            edge_dim     = 1
        )

    def forward(self, x, h, c, edge_index, edge_attr):
        gates = self.gat(torch.cat([x, h], dim=-1), edge_index,
                         edge_attr=edge_attr)
        i, f, g, o = gates.chunk(4, dim=-1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        h_next = h_next + x   # residual
        return h_next, c_next

# MODEL — v5b: v5 mimarisi + B-MSE (mimari birebir)

class SEVIRGraphConvLSTMv5b(nn.Module):
    """
    v5b: v5 ile birebir aynı mimari. Tek fark eğitimde — B-MSE loss.
    """
    def __init__(self, in_channels=1, feat_dim=FEAT_DIM,
                 hidden_dim=HIDDEN_DIM, heads=GAT_HEADS,
                 n_layers=N_LAYERS, grid_size=GRID_SIZE):
        super().__init__()
        self.grid_size  = grid_size
        self.num_nodes  = grid_size * grid_size
        self.feat_dim   = feat_dim
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers

        self.encoder = CNNEncoder(in_channels, feat_dim)
        self.decoder = CNNDecoder(hidden_dim + feat_dim, in_channels)

        self.cells = nn.ModuleList([
            GraphConvLSTMCell(
                in_feat    = feat_dim if i == 0 else hidden_dim,
                hidden_dim = hidden_dim,
                heads      = heads
            )
            for i in range(n_layers)
        ])

        self.register_buffer('edge_index_single',
                             build_grid_edges(grid_size, grid_size))

    def _step(self, frame, h_list, c_list, B, edge_index):
        N    = self.num_nodes
        feat = self.encoder(frame)
        x    = feat.flatten(2).permute(0, 2, 1).reshape(B * N, self.feat_dim)

        # Dinamik Gaussian edge_attr
        edge_attr = compute_edge_attr(x, edge_index)

        new_h, new_c = [], []
        for i, cell in enumerate(self.cells):
            h, c = cell(x, h_list[i], c_list[i], edge_index, edge_attr)
            new_h.append(h)
            new_c.append(c)
            x = h
        return new_h, new_c, feat

    def _decode(self, h_last, enc_feat, B):
        # v5/v5b: sigmoid YOK — raw output. Inference'ta np.clip(0,1) yapılıyor.
        N    = self.num_nodes
        fmap = h_last.reshape(B, N, self.hidden_dim) \
                     .permute(0, 2, 1) \
                     .reshape(B, self.hidden_dim, self.grid_size, self.grid_size)
        return self.decoder(torch.cat([fmap, enc_feat], dim=1))

    def forward(self, x_in):
        B, T, C, H, W = x_in.shape
        N = self.num_nodes

        edge_index = batch_edge_index(
            self.edge_index_single, B, N).to(x_in.device)

        h_list = [torch.zeros(B * N, self.hidden_dim, device=x_in.device)
                  for _ in range(self.n_layers)]
        c_list = [torch.zeros(B * N, self.hidden_dim, device=x_in.device)
                  for _ in range(self.n_layers)]

        enc_feat = None
        for t in range(T):
            h_list, c_list, enc_feat = self._step(
                x_in[:, t], h_list, c_list, B, edge_index)

        pred_frame = x_in[:, -1]
        outputs = []
        for _ in range(N_PREDICT):
            h_list, c_list, enc_feat = self._step(
                pred_frame, h_list, c_list, B, edge_index)
            pred_raw = self._decode(h_list[-1], enc_feat, B)
            outputs.append(pred_raw.unsqueeze(1))
            # Autoregressive feedback: encoder valid range [0,1] görmeli.
            # Output raw kalıyor — loss gradient'i clamp boundary'sinde de flow ediyor.
            pred_frame = pred_raw.clamp(0, 1)

        return torch.cat(outputs, dim=1)

# LOSS — Balanced MSE

def bmse_loss(pred, target):
    """
    Balanced MSE — Shi 2017 inspired. Pixel intensity bin'ine göre kademeli ağırlık.
    Bin sınırları SEVIR/PreDiff CSI threshold'larıyla aynı.
    """
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
    # PreDiff/DGMR convention: max-pool then binarize for CSI-pool metrics
    # arr: (N, H, W) float32 → returns (N, H/pool, W/pool)
    t = torch.from_numpy(arr).float().unsqueeze(1)               # (N, 1, H, W)
    pooled = F.max_pool2d(t, kernel_size=pool_size, stride=pool_size)
    return pooled.squeeze(1).numpy()

def compute_csi_pool(pred_flat, true_flat, threshold, pool_size):
    pp = _max_pool_np(pred_flat, pool_size)
    tp = _max_pool_np(true_flat, pool_size)
    return compute_csi(pp, tp, threshold)

def evaluate_metrics(model, loader):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(DEVICE)).cpu().numpy()
            pred = np.clip(pred, 0.0, 1.0)   # v5/v5b: sigmoid yok -> output clip
            all_pred.append(pred[:, :, 0])
            all_true.append(yb.numpy()[:, :, 0])
    pred_all = np.concatenate(all_pred)                 # (N, T_pred, H, W)
    true_all = np.concatenate(all_true)
    N, T_pred, H, W = pred_all.shape

    ssim_matrix = np.zeros((N, T_pred), dtype=np.float64)
    for i in range(N):
        for t in range(T_pred):
            ssim_matrix[i, t] = sk_ssim(
                pred_all[i, t], true_all[i, t], data_range=1.0)

    per_lt_mse, per_lt_ssim, per_lt_csi_m, per_lt_csi_per_thr = [], [], [], []
    for t in range(T_pred):
        p_t = pred_all[:, t]
        y_t = true_all[:, t]
        mse_t = float(np.mean((p_t - y_t) ** 2))
        per_lt_mse.append(mse_t)
        per_lt_ssim.append(float(ssim_matrix[:, t].mean()))
        csi_t = [compute_csi(p_t, y_t, thr) for thr in CSI_THRESHOLDS]
        per_lt_csi_per_thr.append([round(c, 4) for c in csi_t])
        per_lt_csi_m.append(float(np.mean(csi_t)))

    pred_flat = pred_all.reshape(-1, H, W)
    true_flat = true_all.reshape(-1, H, W)
    csi_scores  = [compute_csi(pred_flat, true_flat, thr) for thr in CSI_THRESHOLDS]
    csi_pool4   = [compute_csi_pool(pred_flat, true_flat, thr, 4)  for thr in CSI_THRESHOLDS]
    csi_pool16  = [compute_csi_pool(pred_flat, true_flat, thr, 16) for thr in CSI_THRESHOLDS]
    mse = float(np.mean((pred_all - true_all) ** 2))

    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
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

# EĞİTİM — Kaggle ile birebir: hem train hem val B-MSE

def train_model(model, train_loader, val_loader):
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
                     optimizer, patience=5, factor=0.5, min_lr=1e-6)
    criterion  = bmse_loss
    scaler     = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))
    best_val   = float('inf')
    no_improve = 0
    history    = {"train_loss": [], "val_loss": [],
                  "train_rmse": [], "val_rmse": []}

    for epoch in range(1, EPOCHS + 1):
        t0 = __import__('time').time()
        model.train()
        train_loss = 0.0
        n_batches  = 0
        n_train_seen = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                pred = model(xb)
                loss = criterion(pred, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
        scheduler.step(val_loss)

        elapsed = __import__('time').time() - t0
        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"Train BMSE: {train_loss:.5f}  sqrt: {np.sqrt(train_loss):.4f}  |  "
              f"Val BMSE: {val_loss:.5f}  sqrt: {np.sqrt(val_loss):.4f}  "
              f"[{elapsed:.0f}s]")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"EarlyStopping tetiklendi ({PATIENCE} epoch).")
                break

    print(f"En iyi Val BMSE: {best_val:.5f}")
    print(f"Not: train/val 'loss' B-MSE skalasında; plain MSE/RMSE evaluate_metrics çıktısında.")
    return history

# GÖRSELLEŞTİRME

def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train BMSE")
    axes[0].plot(history["val_loss"],   label="Val BMSE")
    axes[0].set_title("B-MSE Loss"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(history["train_rmse"], label="Train sqrt(BMSE)")
    axes[1].plot(history["val_rmse"],   label="Val sqrt(BMSE)")
    axes[1].set_title("sqrt(B-MSE)"); axes[1].legend(); axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "training_history.png"), dpi=150)
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
    path = os.path.join(FIG_DIR, "sevir_graphconvlstm_v5b_comparison.gif")
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

    # Test loader sadece tam eğitimde gerek (SMOKE_TEST'te değerlendirme yok)
    if not SMOKE_TEST:
        test_ds  = SEVIRDataset(TEST_FILE, max_samples=max_samples)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=8, pin_memory=True,
                                  worker_init_fn=_seed_worker, generator=g)

    # Defansif shape kontrolü — an HPC cluster .npy formatı beklendiği gibi mi?
    # Beklenen: x=(B, 7, 1, 128, 128), y=(B, 6, 1, 128, 128)
    _xb, _yb = next(iter(train_loader))
    print(f"Batch shape kontrolü — x: {tuple(_xb.shape)}  y: {tuple(_yb.shape)}")
    assert _xb.shape[1:] == (N_INPUT, 1, IMG_SIZE, IMG_SIZE), (
        f"Beklenmedik x shape: {tuple(_xb.shape)} — .npy ham 25-kare mi?")
    assert _yb.shape[1:] == (N_PREDICT, 1, IMG_SIZE, IMG_SIZE), (
        f"Beklenmedik y shape: {tuple(_yb.shape)}")
    del _xb, _yb

    model = SEVIRGraphConvLSTMv5b(
        in_channels = 1,
        feat_dim    = FEAT_DIM,
        hidden_dim  = HIDDEN_DIM,
        heads       = GAT_HEADS,
        n_layers    = N_LAYERS,
        grid_size   = GRID_SIZE
    ).to(DEVICE)

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Toplam parametre: {total:,}")
    print(f"GraphConvLSTM v5b — v5 mimarisi (GroupNorm, sigmoid kaldırıldı) + B-MSE loss")
    print(f"BMSE_WEIGHTS: {BMSE_WEIGHTS}")
    print(f"Sigma: {SIGMA}\n")

    history = train_model(model, train_loader, val_loader)

    if SMOKE_TEST:
        print("\n>>> SMOKE TEST BAŞARILI — ortam doğrulandı <<<")
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    CUDA OK, torch_geometric OK, AMP OK, forward/backward OK")
    else:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
        print("\nMetrikler hesaplanıyor (val)...")
        val_metrics  = evaluate_metrics(model, val_loader)
        print("Metrikler hesaplanıyor (test)...")
        test_metrics = evaluate_metrics(model, test_loader)

        out = {
            "model":          "SEVIRGraphConvLSTMv5b",
            "loss":           "B-MSE",
            "bmse_weights":   BMSE_WEIGHTS,
            "epochs_ran":     len(history["train_loss"]),
            "best_val_bmse":  round(min(history["val_loss"]), 6),
            "val":  val_metrics,
            "test": test_metrics,
        }

        with open(METRICS_PATH, "w") as f:
            json.dump(out, f, indent=2)

        print(f"\n--- SEVIR GraphConvLSTM v5b Sonuçlar ---")
        print(f"  Best Val BMSE : {out['best_val_bmse']:.6f}")
        print(f"  Epoch         : {out['epochs_ran']}")
        print(f"  -- Validation (plain MSE on output) --")
        print(f"  MSE        : {val_metrics['mse']:.6f}")
        print(f"  RMSE       : {val_metrics['rmse']:.6f}")
        print(f"  SSIM       : {val_metrics['ssim']:.4f}")
        print(f"  CSI-M      : {val_metrics['csi_m']:.4f}")
        print(f"  CSI-pool4  : {val_metrics['csi_pool4_m']:.4f}")
        print(f"  CSI-pool16 : {val_metrics['csi_pool16_m']:.4f}")
        print(f"  -- Test (held-out, time_utc >= 2019-06-01, plain MSE on output) --")
        print(f"  MSE        : {test_metrics['mse']:.6f}")
        print(f"  RMSE       : {test_metrics['rmse']:.6f}")
        print(f"  SSIM       : {test_metrics['ssim']:.4f}")
        print(f"  CSI-M      : {test_metrics['csi_m']:.4f}  (PreDiff Tab.2 ConvLSTM: 0.4185 | Kaggle v5b: 0.3886)")
        print(f"  CSI-pool4  : {test_metrics['csi_pool4_m']:.4f}  (PreDiff Tab.2 ConvLSTM: 0.4452 | Kaggle v5b: 0.4495)")
        print(f"  CSI-pool16 : {test_metrics['csi_pool16_m']:.4f}  (PreDiff Tab.2 ConvLSTM: 0.5135 | Kaggle v5b: 0.5252)")
        print(f"  CSI@thr=181/255 : {test_metrics['csi_per_threshold'][4]:.4f}  (Kaggle v5b: 0.2676)")
        print(f"  CSI@thr=219/255 : {test_metrics['csi_per_threshold'][5]:.4f}  (Kaggle v5b: 0.1063)")

        plot_training_history(history)

        sample_x, sample_y = val_ds[0]
        model.eval()
        with torch.no_grad():
            pred = model(sample_x.unsqueeze(0).to(DEVICE))[0].cpu().numpy()
            pred = np.clip(pred, 0.0, 1.0)   # v5b: sigmoid yok -> sample clip

        save_comparison_gif(sample_y[:, 0].numpy(), pred[:, 0])
        print(f"\nTamamlandı.\n  Model: {MODEL_SAVE_PATH}\n  Figürler: {FIG_DIR}")
