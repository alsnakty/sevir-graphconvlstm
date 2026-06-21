"""
F.3 — "well-localised, not smoother" post-hoc spatial/spectral diagnostics.

Tahminlerin bulanık mı yoksa lokalize mi olduğunu
EĞİTİM GEREKTİRMEDEN, mevcut v5b checkpoint'i test seti üzerinde koşturarak ölç.

Üç tanı (hepsi online akümülasyonla — tahminler RAM'de saklanmaz):
  1. RAPSD  — radyal güç spektrumu (pred / GT / persistence / Gaussian-blur(GT)).
              Bulanıklık yüksek dalga-sayısı enerjisini düşürür. Tahmin GT'ye mi
              yoksa blur(GT)'ye mi yakın?
  2. FSS    — Fractions Skill Score, komşuluk × eşik × lead-time, persistence ref.
  3. FBI    — frekans bias (count_fcst / count_obs), eşik × lead-time.

Model sınıfları 13_sevir_graphconvlstm_v5b_truba.py ile birebir (o modül doğrudan
import edilemediği için sınıflar burada kopyalanmıştır).

Çıktı: analysis/f3_out/f3_metrics.json + f3_rapsd.png + f3_fss.png + f3_fbi.png
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, gaussian_filter
from torch_geometric.nn import GATConv

# YOLLAR
REPO      = r'/path/to/project'
CKPT      = os.path.join(REPO, 'results', 'maxpool_10dk_7-6',
                         's13v5b_graphconvlstm', 'sevir_graphconvlstm_v5b_model.pt')
TEST_FILE = r'/path/to/data_root\data\sevir_processed\sevir_vil_test.npy'
OUT_DIR   = os.path.join(REPO, 'analysis', 'f3_out')

# SABİTLER (v5b ile birebir)
N_INPUT, N_PREDICT = 7, 6
IMG_SIZE, GRID_SIZE = 128, 32
FEAT_DIM, HIDDEN_DIM, GAT_HEADS, N_LAYERS, SIGMA = 64, 64, 4, 2, 1.0
DEVICE = torch.device('cpu')

# FSS / FBI analiz parametreleri
# Eşikler SEVIR/PreDiff CSI ailesinden (light → very heavy)
FSS_THRESHOLDS = {'16': 16/255, '74': 74/255, '133': 133/255, '181': 181/255}
FSS_SCALES = [1, 3, 5, 9, 17, 33]          # komşuluk kenar uzunluğu (piksel)
BLUR_SIGMA = 1.0                            # blur(GT) referans eğrisi için

# MODEL (v5b birebir — make_qualitative_figure.py ile aynı)
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
    src, dst = edge_index[0], edge_index[1]
    diff = x_nodes[src] - x_nodes[dst]
    dist_sq = (diff ** 2).sum(dim=-1, keepdim=True)
    return torch.exp(-dist_sq / (2 * sigma ** 2))

class CNNEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=FEAT_DIM):
        super().__init__()
        mid = out_channels // 2
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
        mid = HIDDEN_DIM // 2
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

class GraphConvLSTMCell(nn.Module):
    def __init__(self, in_feat, hidden_dim=HIDDEN_DIM, heads=GAT_HEADS):
        super().__init__()
        assert in_feat == hidden_dim
        self.hidden_dim = hidden_dim
        self.gat = GATConv(in_channels=in_feat + hidden_dim,
                           out_channels=(4 * hidden_dim) // heads,
                           heads=heads, concat=True, edge_dim=1)
    def forward(self, x, h, c, edge_index, edge_attr):
        gates = self.gat(torch.cat([x, h], dim=-1), edge_index, edge_attr=edge_attr)
        i, f, g, o = gates.chunk(4, dim=-1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        h_next = h_next + x
        return h_next, c_next

class SEVIRGraphConvLSTMv5b(nn.Module):
    def __init__(self, in_channels=1, feat_dim=FEAT_DIM, hidden_dim=HIDDEN_DIM,
                 heads=GAT_HEADS, n_layers=N_LAYERS, grid_size=GRID_SIZE):
        super().__init__()
        self.grid_size = grid_size
        self.num_nodes = grid_size * grid_size
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.encoder = CNNEncoder(in_channels, feat_dim)
        self.decoder = CNNDecoder(hidden_dim + feat_dim, in_channels)
        self.cells = nn.ModuleList([
            GraphConvLSTMCell(in_feat=feat_dim if i == 0 else hidden_dim,
                              hidden_dim=hidden_dim, heads=heads)
            for i in range(n_layers)])
        self.register_buffer('edge_index_single', build_grid_edges(grid_size, grid_size))

    def _step(self, frame, h_list, c_list, B, edge_index):
        N = self.num_nodes
        feat = self.encoder(frame)
        x = feat.flatten(2).permute(0, 2, 1).reshape(B * N, self.feat_dim)
        edge_attr = compute_edge_attr(x, edge_index)
        new_h, new_c = [], []
        for i, cell in enumerate(self.cells):
            h, c = cell(x, h_list[i], c_list[i], edge_index, edge_attr)
            new_h.append(h); new_c.append(c); x = h
        return new_h, new_c, feat

    def _decode(self, h_last, enc_feat, B):
        N = self.num_nodes
        fmap = h_last.reshape(B, N, self.hidden_dim).permute(0, 2, 1) \
                     .reshape(B, self.hidden_dim, self.grid_size, self.grid_size)
        return self.decoder(torch.cat([fmap, enc_feat], dim=1))

    def forward(self, x_in):
        B, T, C, H, W = x_in.shape
        N = self.num_nodes
        edge_index = batch_edge_index(self.edge_index_single, B, N).to(x_in.device)
        h_list = [torch.zeros(B * N, self.hidden_dim, device=x_in.device) for _ in range(self.n_layers)]
        c_list = [torch.zeros(B * N, self.hidden_dim, device=x_in.device) for _ in range(self.n_layers)]
        enc_feat = None
        for t in range(T):
            h_list, c_list, enc_feat = self._step(x_in[:, t], h_list, c_list, B, edge_index)
        pred_frame = x_in[:, -1]
        outputs = []
        for _ in range(N_PREDICT):
            h_list, c_list, enc_feat = self._step(pred_frame, h_list, c_list, B, edge_index)
            pred_raw = self._decode(h_list[-1], enc_feat, B)
            outputs.append(pred_raw.unsqueeze(1))
            pred_frame = pred_raw.clamp(0, 1)
        return torch.cat(outputs, dim=1)

# RAPSD yardımcıları — radyal bin grid'i bir kez hesapla
def _radial_bins(H=IMG_SIZE, W=IMG_SIZE):
    cy, cx = H // 2, W // 2
    y, x = np.indices((H, W))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    kbin = np.round(r).astype(np.int64)          # (H,W)
    nbins = kbin.max() + 1
    pix_per_bin = np.bincount(kbin.ravel(), minlength=nbins).astype(np.float64)
    return kbin.ravel(), pix_per_bin, nbins

_KBIN_FLAT, _PIX_PER_BIN, _NBINS = _radial_bins()

def rapsd_sum(fields):
    """fields: (M,H,W) → her bin için psd toplamı (M alanı boyunca), shape (nbins,).
    Her alan ortalaması çıkarılır (DC bileşeni). M alanın bin-toplamı döner."""
    f = fields - fields.mean(axis=(1, 2), keepdims=True)
    F2 = np.fft.fftshift(np.fft.fft2(f, axes=(1, 2)), axes=(1, 2))
    psd = (np.abs(F2) ** 2)                       # (M,H,W)
    psd_sum_2d = psd.sum(axis=0).ravel()          # (H*W,)
    return np.bincount(_KBIN_FLAT, weights=psd_sum_2d, minlength=_NBINS)

# ANA ANALİZ
def run(limit=None, batch_size=8):
    os.makedirs(OUT_DIR, exist_ok=True)
    assert os.path.exists(CKPT), f'Checkpoint yok: {CKPT}'
    assert os.path.exists(TEST_FILE), f'Test verisi yok: {TEST_FILE}'

    data = np.load(TEST_FILE, mmap_mode='r')      # (N,25,128,128) float32 [0,1]
    N = data.shape[0] if limit is None else min(limit, data.shape[0])
    print(f'Test verisi: {data.shape}  kullanılacak: {N} dizi')

    model = SEVIRGraphConvLSTMv5b().to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f'Model param: {sum(p.numel() for p in model.parameters()):,}')

    # --- akümülatörler ---
    # RAPSD: lead-time başına pred/gt + global persistence/blur(gt) bin-toplamları
    rapsd = {k: np.zeros(_NBINS) for k in
             ['pred', 'gt', 'pers', 'blur']}
    rapsd_count = 0                                # toplam katkı veren alan sayısı

    # FSS: (thr, scale, lead) → numerator, denominator (model + persistence)
    fss_num   = {m: np.zeros((len(FSS_THRESHOLDS), len(FSS_SCALES), N_PREDICT))
                 for m in ['model', 'pers']}
    fss_den   = {m: np.zeros((len(FSS_THRESHOLDS), len(FSS_SCALES), N_PREDICT))
                 for m in ['model', 'pers']}

    # FBI: (thr, lead) → count_fcst, count_obs (model + persistence)
    fbi_fcst  = {m: np.zeros((len(FSS_THRESHOLDS), N_PREDICT)) for m in ['model', 'pers']}
    fbi_obs   = np.zeros((len(FSS_THRESHOLDS), N_PREDICT))

    thr_vals = list(FSS_THRESHOLDS.values())

    t0 = time.time()
    n_done = 0
    for start in range(0, N, batch_size):
        idx = list(range(start, min(start + batch_size, N)))
        seq = data[idx].astype(np.float32)[:, ::2]            # (B,13,128,128)
        x = torch.from_numpy(seq[:, :N_INPUT]).unsqueeze(2)   # (B,7,1,128,128)
        gt = seq[:, N_INPUT:]                                 # (B,6,128,128)
        pers_frame = seq[:, N_INPUT - 1]                      # (B,128,128) son input

        with torch.no_grad():
            pred = model(x.to(DEVICE))[:, :, 0].cpu().numpy() # (B,6,128,128)
        pred = np.clip(pred, 0.0, 1.0)
        B = pred.shape[0]
        pers = np.repeat(pers_frame[:, None], N_PREDICT, axis=1)  # (B,6,128,128)

        # ---- RAPSD (tüm lead-time'lar birlikte; bulanıklık global gösterilir) ----
        pf = pred.reshape(-1, IMG_SIZE, IMG_SIZE)
        gf = gt.reshape(-1, IMG_SIZE, IMG_SIZE)
        prf = pers.reshape(-1, IMG_SIZE, IMG_SIZE)
        blf = gaussian_filter(gf, sigma=(0, BLUR_SIGMA, BLUR_SIGMA))
        rapsd['pred'] += rapsd_sum(pf)
        rapsd['gt']   += rapsd_sum(gf)
        rapsd['pers'] += rapsd_sum(prf)
        rapsd['blur'] += rapsd_sum(blf)
        rapsd_count   += pf.shape[0]

        # ---- FSS + FBI ----
        for ti, thr in enumerate(thr_vals):
            obs_bin   = (gt   >= thr).astype(np.float32)      # (B,6,128,128)
            mod_bin   = (pred >= thr).astype(np.float32)
            pers_bin  = (pers >= thr).astype(np.float32)
            # FBI sayımları (lead başına)
            fbi_obs[ti]          += obs_bin.sum(axis=(0, 2, 3))
            fbi_fcst['model'][ti] += mod_bin.sum(axis=(0, 2, 3))
            fbi_fcst['pers'][ti]  += pers_bin.sum(axis=(0, 2, 3))
            for si, sc in enumerate(FSS_SCALES):
                Of = uniform_filter(obs_bin,  size=(1, 1, sc, sc), mode='constant')
                Mf = uniform_filter(mod_bin,  size=(1, 1, sc, sc), mode='constant')
                Pf = uniform_filter(pers_bin, size=(1, 1, sc, sc), mode='constant')
                # lead başına topla (axis 0=B,2,3=H,W)
                fss_num['model'][ti, si] += ((Mf - Of) ** 2).sum(axis=(0, 2, 3))
                fss_den['model'][ti, si] += (Mf ** 2 + Of ** 2).sum(axis=(0, 2, 3))
                fss_num['pers'][ti, si]  += ((Pf - Of) ** 2).sum(axis=(0, 2, 3))
                fss_den['pers'][ti, si]  += (Pf ** 2 + Of ** 2).sum(axis=(0, 2, 3))

        n_done += B
        if (start // batch_size) % 20 == 0 or n_done >= N:
            el = time.time() - t0
            rate = n_done / el if el > 0 else 0
            eta = (N - n_done) / rate if rate > 0 else 0
            print(f'  {n_done}/{N}  [{el:.0f}s, {rate:.1f} seq/s, ETA {eta:.0f}s]',
                  flush=True)

    # --- RAPSD ortalama: bin başına ortalama güç ---
    safe_pix = np.where(_PIX_PER_BIN > 0, _PIX_PER_BIN, 1)
    rapsd_avg = {k: v / (rapsd_count * safe_pix) for k, v in rapsd.items()}

    # --- FSS = 1 - num/den ---
    fss = {}
    for m in ['model', 'pers']:
        with np.errstate(divide='ignore', invalid='ignore'):
            fss[m] = 1.0 - np.where(fss_den[m] > 0, fss_num[m] / fss_den[m], 1.0)

    # --- FBI ---
    fbi = {}
    for m in ['model', 'pers']:
        with np.errstate(divide='ignore', invalid='ignore'):
            fbi[m] = np.where(fbi_obs > 0, fbi_fcst[m] / fbi_obs, np.nan)

    # ---- KAYDET (JSON) ----
    out = {
        'n_sequences': int(N),
        'rapsd': {
            'wavenumber': list(range(_NBINS)),
            'pred': rapsd_avg['pred'].tolist(),
            'gt':   rapsd_avg['gt'].tolist(),
            'pers': rapsd_avg['pers'].tolist(),
            'blur_gt_sigma{}'.format(BLUR_SIGMA): rapsd_avg['blur'].tolist(),
        },
        'fss': {
            'thresholds': list(FSS_THRESHOLDS.keys()),
            'scales': FSS_SCALES,
            'leadtimes_min': [(t + 1) * 10 for t in range(N_PREDICT)],
            'model': fss['model'].tolist(),
            'pers':  fss['pers'].tolist(),
        },
        'fbi': {
            'thresholds': list(FSS_THRESHOLDS.keys()),
            'leadtimes_min': [(t + 1) * 10 for t in range(N_PREDICT)],
            'model': fbi['model'].tolist(),
            'pers':  fbi['pers'].tolist(),
        },
    }
    jpath = os.path.join(OUT_DIR, 'f3_metrics.json')
    with open(jpath, 'w') as f:
        json.dump(out, f, indent=2)
    print('JSON:', jpath)

    _plot(rapsd_avg, fss, fbi)
    print(f'Bitti. Toplam süre {time.time() - t0:.0f}s')
    return out

# GRAFİKLER
def _plot(rapsd_avg, fss, fbi):
    thr_keys = list(FSS_THRESHOLDS.keys())

    # 1) RAPSD
    fig, ax = plt.subplots(figsize=(6, 4.5))
    k = np.arange(_NBINS)
    sel = k >= 1                                  # DC'yi atla
    ax.loglog(k[sel], rapsd_avg['gt'][sel],   'k-',  lw=2, label='Ground truth')
    ax.loglog(k[sel], rapsd_avg['pers'][sel], 'g--', lw=1.3, label='Persistence')
    ax.loglog(k[sel], rapsd_avg['blur'][sel], 'b:',  lw=1.3,
              label=f'Blurred GT ($\\sigma$={BLUR_SIGMA})')
    ax.loglog(k[sel], rapsd_avg['pred'][sel], 'r-',  lw=1.8, label='GraphConvLSTM')
    ax.set_xlabel('Radial wavenumber'); ax.set_ylabel('Radially averaged PSD')
    ax.set_title('Power spectrum — is the forecast smoother?')
    ax.legend(); ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'f3_rapsd.png'), dpi=150)
    plt.close(fig)

    # 2) FSS vs scale, eşik başına panel, +60 dk lead (son)
    fig, axes = plt.subplots(1, len(thr_keys), figsize=(3.2 * len(thr_keys), 3.4),
                             sharey=True)
    lead = N_PREDICT - 1
    for ti, tk in enumerate(thr_keys):
        ax = axes[ti]
        ax.plot(FSS_SCALES, fss['model'][ti, :, lead], 'r-o', label='GraphConvLSTM')
        ax.plot(FSS_SCALES, fss['pers'][ti, :, lead],  'g--s', label='Persistence')
        ax.set_title(f'thr={tk}/255 (+60 min)'); ax.set_xlabel('Neighborhood (px)')
        ax.grid(True, alpha=0.3)
        if ti == 0:
            ax.set_ylabel('FSS'); ax.legend(fontsize=8)
    fig.suptitle('Fractions Skill Score vs spatial scale')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'f3_fss.png'), dpi=150)
    plt.close(fig)

    # 3) FBI vs lead-time, eşik başına eğri
    fig, ax = plt.subplots(figsize=(6, 4.5))
    lt = [(t + 1) * 10 for t in range(N_PREDICT)]
    for ti, tk in enumerate(thr_keys):
        ax.plot(lt, fbi['model'][ti], '-o', label=f'thr={tk}/255')
    ax.axhline(1.0, color='k', lw=0.8, ls=':')
    ax.set_xlabel('Lead time (min)'); ax.set_ylabel('Frequency bias (fcst/obs)')
    ax.set_title('Frequency bias — under-forecasting of intense pixels?')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'f3_fbi.png'), dpi=150)
    plt.close(fig)
    print('Grafikler:', OUT_DIR)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None, help='kaç dizi (test için)')
    ap.add_argument('--batch', type=int, default=8)
    args = ap.parse_args()
    run(limit=args.limit, batch_size=args.batch)
