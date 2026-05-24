"""
models/architectures.py  —  ClimateAI Part 2
=============================================
FourCastNetLite  · GraphCastLite  · PanguLite
+ lat_weighted_mse  · train_one_epoch  · get_model
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def lat_weighted_mse(pred: Tensor, target: Tensor, lats: Tensor) -> Tensor:
    """
    Latitude-weighted MSE. lats: 1-D degrees tensor (H,).
    pred/target: (B, C, H, W).
    """
    w = torch.cos(torch.deg2rad(lats)).to(pred.device)   # (H,)
    w = w / w.mean()
    w = w[None, None, :, None]                            # (1,1,H,1)
    return (w * (pred - target) ** 2).mean()


def train_one_epoch(model, loader, optimizer, lats: Tensor,
                    ar_steps: int = 1, device: str = "cpu") -> float:
    """
    One training epoch with optional autoregressive rollout.
    Clips gradients at norm 32.
    """
    model.train()
    total, count = 0.0, 0
    lats = lats.to(device)
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        cur = xb
        loss = torch.tensor(0.0, device=device)
        for _ in range(ar_steps):
            cur = model(cur)
            loss = loss + lat_weighted_mse(cur, yb, lats)
        loss = loss / ar_steps
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 32.0)
        optimizer.step()
        total += loss.item()
        count += 1
    return total / max(count, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 1. FourCastNetLite  (Pathak et al. / NVIDIA, 2022)
#    patch_embed(4×4) → 6× AFNOBlock → linear_decoder
# ─────────────────────────────────────────────────────────────────────────────

class AFNOBlock(nn.Module):
    """
    Adaptive Fourier Neural Operator block.
    FFT → learned complex weights per freq → IFFT → LayerNorm → MLP
    Complexity O(N log N).
    """
    def __init__(self, embed_dim: int, mlp_ratio: int = 4, n_blocks: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_blocks  = n_blocks
        # Learnable complex weights in frequency domain
        self.w_re = nn.Parameter(torch.zeros(1, n_blocks, embed_dim // n_blocks, embed_dim // n_blocks))
        self.w_im = nn.Parameter(torch.zeros(1, n_blocks, embed_dim // n_blocks, embed_dim // n_blocks))
        nn.init.normal_(self.w_re, std=0.02)
        nn.init.normal_(self.w_im, std=0.02)
        # LayerNorm + MLP
        self.norm = nn.LayerNorm(embed_dim)
        hidden = embed_dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        residual = x
        # FFT over spatial dims
        xf = torch.fft.rfft2(x.permute(0, 3, 1, 2), norm="ortho")  # (B,C,H,W//2+1)
        # Split channels into blocks and apply learned weights
        block_size = C // self.n_blocks
        xf_r = xf.real.reshape(B, self.n_blocks, block_size, H, -1)
        xf_i = xf.imag.reshape(B, self.n_blocks, block_size, H, -1)
        # Weight application (broadcast over spatial)
        wr, wi = self.w_re, self.w_im             # (1,nb,bs,bs)
        xf_r_out = (torch.einsum("bnihw,bnij->bnjhw", xf_r, wr)
                  - torch.einsum("bnihw,bnij->bnjhw", xf_i, wi))
        xf_i_out = (torch.einsum("bnihw,bnij->bnjhw", xf_r, wi)
                  + torch.einsum("bnihw,bnij->bnjhw", xf_i, wr))
        xf_out = torch.complex(
            xf_r_out.reshape(B, C, H, -1),
            xf_i_out.reshape(B, C, H, -1),
        )
        # IFFT back to spatial
        x_ifft = torch.fft.irfft2(xf_out, s=(H, W), norm="ortho")  # (B,C,H,W)
        x = x_ifft.permute(0, 2, 3, 1)  # (B,H,W,C)
        x = residual + x
        # LayerNorm + MLP
        x = x + self.mlp(self.norm(x))
        return x


class FourCastNetLite(nn.Module):
    """
    FourCastNet-Lite: patch_embed(4×4) → 6×AFNOBlock → linear_decoder.
    In/Out: [B, C, H, W].
    """
    MODEL_NAME  = "FourCastNetLite"
    MODEL_COLOR = "#8b5cf6"   # purple

    def __init__(self, in_channels: int, out_channels: int,
                 img_h: int, img_w: int,
                 patch_size: int = 4, embed_dim: int = 256,
                 depth: int = 6, n_blocks: int = 8,
                 autoregressive_finetune: bool = False):
        super().__init__()
        self.patch_size  = patch_size
        self.embed_dim   = embed_dim
        self.autoregressive_finetune = autoregressive_finetune
        ph = img_h // patch_size
        pw = img_w // patch_size
        self.ph, self.pw = ph, pw
        self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, ph, pw, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([
            AFNOBlock(embed_dim, n_blocks=n_blocks) for _ in range(depth)
        ])
        self.norm    = nn.LayerNorm(embed_dim)
        # Decode patches back to full resolution
        self.decoder = nn.ConvTranspose2d(embed_dim, out_channels,
                                          kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W)
        x = self.patch_embed(x)                        # (B, embed, ph, pw)
        x = x.permute(0, 2, 3, 1) + self.pos_embed    # (B, ph, pw, embed)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)                     # (B, embed, ph, pw)
        x = self.decoder(x)                            # (B, out_ch, H, W)
        return x

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 2. GraphCastLite  (Lam et al. / DeepMind, 2023)
#    kNN graph (k=6) · GNNBlock: edge_mlp → scatter_mean → node_mlp+residual
#    encoder(2L) → processor(4L) → decoder
# ─────────────────────────────────────────────────────────────────────────────

def _build_knn_graph(lats: Tensor, lons: Tensor, k: int = 6):
    """
    Build kNN graph from lat/lon grid.
    Returns edge_index (2, E) and edge_feats (E, 3): [norm_dist, sin_bearing, cos_bearing].
    """
    lat_r  = torch.deg2rad(lats)
    lon_r  = torch.deg2rad(lons)
    # Cartesian coords on unit sphere
    x = torch.cos(lat_r) * torch.cos(lon_r)
    y = torch.cos(lat_r) * torch.sin(lon_r)
    z = torch.sin(lat_r)
    xyz = torch.stack([x, y, z], dim=-1)              # (N, 3)
    # Pairwise distances
    dist = torch.cdist(xyz, xyz)                       # (N, N)
    dist.fill_diagonal_(float("inf"))
    _, idx = dist.topk(k, dim=-1, largest=False)      # (N, k)
    src = torch.arange(len(lats)).unsqueeze(1).expand_as(idx).reshape(-1)
    dst = idx.reshape(-1)
    edge_index = torch.stack([src, dst], dim=0)       # (2, N*k)
    # Edge features: normalised distance, bearing sin/cos
    dlat = lat_r[dst] - lat_r[src]
    dlon = lon_r[dst] - lon_r[src]
    norm_dist = dist[src, dst] / (dist[src, dst].max() + 1e-8)
    sin_b = torch.sin(torch.atan2(dlon, dlat))
    cos_b = torch.cos(torch.atan2(dlon, dlat))
    edge_feats = torch.stack([norm_dist, sin_b, cos_b], dim=-1)   # (E, 3)
    return edge_index, edge_feats


class GNNBlock(nn.Module):
    """edge_mlp(src‖dst‖edge_feat) → scatter_mean → node_mlp + residual"""
    def __init__(self, node_dim: int, edge_dim: int = 3, hidden: int = 128):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, node_dim),
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, nodes: Tensor, edge_index: Tensor, edge_feats: Tensor) -> Tensor:
        src, dst = edge_index
        e_in  = torch.cat([nodes[src], nodes[dst], edge_feats], dim=-1)
        e_msg = self.edge_mlp(e_in)                                      # (E, hidden)
        # scatter_mean: aggregate messages to each destination node
        agg   = torch.zeros(nodes.shape[0], e_msg.shape[-1], device=nodes.device)
        count = torch.zeros(nodes.shape[0], 1, device=nodes.device)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(e_msg), e_msg)
        count.scatter_add_(0, dst.unsqueeze(-1), torch.ones(len(dst), 1, device=nodes.device))
        agg = agg / (count + 1e-8)
        n_in  = torch.cat([nodes, agg], dim=-1)
        delta = self.node_mlp(n_in)
        return self.norm(nodes + delta)


class GraphCastLite(nn.Module):
    """
    GraphCastLite: encoder(2L) → processor(4L) → decoder.
    Accepts grid input (B, C, H, W), flattens to nodes, processes on graph,
    reshapes back to (B, out_ch, H, W).
    """
    MODEL_NAME  = "GraphCastLite"
    MODEL_COLOR = "#06b6d4"   # cyan

    def __init__(self, in_channels: int, out_channels: int,
                 img_h: int, img_w: int, hidden: int = 128, k: int = 6,
                 autoregressive_finetune: bool = False):
        super().__init__()
        self.img_h, self.img_w = img_h, img_w
        self.out_channels = out_channels
        self.autoregressive_finetune = autoregressive_finetune
        # Build static graph from default lat/lon grid
        import numpy as np
        lats_1d = torch.tensor(np.linspace(-90, 90, img_h), dtype=torch.float32)
        lons_1d = torch.tensor(np.linspace(-180, 180, img_w), dtype=torch.float32)
        lat_g, lon_g = torch.meshgrid(lats_1d, lons_1d, indexing="ij")
        flat_lats = lat_g.reshape(-1)
        flat_lons = lon_g.reshape(-1)
        edge_index, edge_feats = _build_knn_graph(flat_lats, flat_lons, k=k)
        self.register_buffer("edge_index",  edge_index)
        self.register_buffer("edge_feats",  edge_feats)
        N = img_h * img_w
        # Encoder: project grid features → node embeddings
        self.node_encoder = nn.Linear(in_channels, hidden)
        self.enc_blocks = nn.ModuleList([GNNBlock(hidden, 3, hidden) for _ in range(2)])
        # Processor
        self.proc_blocks = nn.ModuleList([GNNBlock(hidden, 3, hidden) for _ in range(4)])
        # Decoder
        self.dec_blocks  = nn.ModuleList([GNNBlock(hidden, 3, hidden) for _ in range(2)])
        self.node_decoder = nn.Linear(hidden, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        # Flatten grid to nodes: (B, N, C)
        nodes_in = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        ei = self.edge_index
        ef = self.edge_feats
        outs = []
        for b in range(B):
            n = self.node_encoder(nodes_in[b])          # (N, hidden)
            for blk in self.enc_blocks:
                n = blk(n, ei, ef)
            for blk in self.proc_blocks:
                n = blk(n, ei, ef)
            for blk in self.dec_blocks:
                n = blk(n, ei, ef)
            outs.append(self.node_decoder(n))           # (N, out_ch)
        out = torch.stack(outs, dim=0)                  # (B, N, out_ch)
        return out.reshape(B, H, W, self.out_channels).permute(0, 3, 1, 2)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 3. PanguLite  (Bi et al. / Huawei, 2023)
#    3D input [B,Levels,H,W,Vars] · Earth Positional Bias
#    4× Swin window-attn(4×4) + rel-pos-bias · 3 lead-time heads: 6h/24h/72h
# ─────────────────────────────────────────────────────────────────────────────

class EarthPositionalBias(nn.Module):
    """Learnable bias table indexed by (lat_bin, lon_bin)."""
    def __init__(self, num_heads: int, lat_bins: int = 8, lon_bins: int = 16):
        super().__init__()
        self.lat_bins = lat_bins
        self.lon_bins = lon_bins
        self.table = nn.Parameter(torch.zeros(lat_bins, lon_bins, num_heads))
        nn.init.trunc_normal_(self.table, std=0.02)

    def forward(self, H: int, W: int) -> Tensor:
        lat_idx = (torch.arange(H).float() / H * self.lat_bins).long().clamp(0, self.lat_bins - 1)
        lon_idx = (torch.arange(W).float() / W * self.lon_bins).long().clamp(0, self.lon_bins - 1)
        bias = self.table[lat_idx[:, None], lon_idx[None, :]]   # (H, W, heads)
        return bias.permute(2, 0, 1)                             # (heads, H, W)


class SwinWindowAttn(nn.Module):
    """
    Window-based multi-head self-attention with relative position bias.
    Window size: win × win.
    """
    def __init__(self, dim: int, num_heads: int = 4, win: int = 4):
        super().__init__()
        self.win = win
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv   = nn.Linear(dim, dim * 3)
        self.proj  = nn.Linear(dim, dim)
        self.norm  = nn.LayerNorm(dim)
        # Relative position bias table: (2*win-1)^2 × num_heads
        self.rel_bias = nn.Parameter(torch.zeros((2 * win - 1) ** 2, num_heads))
        nn.init.trunc_normal_(self.rel_bias, std=0.02)
        coords = torch.arange(win)
        grid   = torch.stack(torch.meshgrid(coords, coords, indexing="ij"))  # (2,w,w)
        flat   = grid.reshape(2, -1)
        rel    = flat[:, :, None] - flat[:, None, :]                          # (2,w²,w²)
        rel[0] += win - 1
        rel[1] += win - 1
        rel[0] *= (2 * win - 1)
        idx    = rel.sum(0)                                                    # (w²,w²)
        self.register_buffer("rel_idx", idx)
        mlp_hidden = dim * 4
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        win = self.win
        # Pad to multiple of win
        pad_h = (win - H % win) % win
        pad_w = (win - W % win) % win
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[1], x.shape[2]
        # Partition into windows
        x_win = x.reshape(B, Hp // win, win, Wp // win, win, C)
        x_win = x_win.permute(0, 1, 3, 2, 4, 5).reshape(-1, win * win, C)  # (B*nW, w², C)
        res = x_win
        x_n = self.norm(x_win)
        nW = x_n.shape[0]
        qkv = self.qkv(x_n).reshape(nW, win * win, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                                              # (nW, heads, w², hd)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        rel  = self.rel_bias[self.rel_idx].permute(2, 0, 1).unsqueeze(0)    # (1,heads,w²,w²)
        attn = attn + rel
        attn = attn.softmax(dim=-1)
        x_win = (attn @ v).transpose(1, 2).reshape(nW, win * win, C)
        x_win = self.proj(x_win) + res
        x_win = x_win + self.mlp(x_win)
        # Merge windows
        x_win = x_win.reshape(B, Hp // win, Wp // win, win, win, C)
        x_out = x_win.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)
        # Remove padding
        return x_out[:, :H, :W, :]


class PanguLite(nn.Module):
    """
    Pangu-Lite: 3D pressure-level aware model with Earth positional bias.
    Input:  (B, C, H, W)  — channels represent Levels×Vars flattened.
    Output: (B, C, H, W)  — via 3 lead-time heads (6h / 24h / 72h).
    """
    MODEL_NAME  = "PanguLite"
    MODEL_COLOR = "#f97316"   # orange

    LEAD_TIMES = {"6h": 0, "24h": 1, "72h": 2}

    def __init__(self, in_channels: int, out_channels: int,
                 img_h: int, img_w: int,
                 embed_dim: int = 192, num_heads: int = 4,
                 win: int = 4, depth: int = 4,
                 autoregressive_finetune: bool = False):
        super().__init__()
        self.autoregressive_finetune = autoregressive_finetune
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        # Input projection
        self.input_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        # Earth positional bias
        self.earth_bias = EarthPositionalBias(num_heads)
        # Swin blocks
        self.blocks = nn.ModuleList([
            SwinWindowAttn(embed_dim, num_heads=num_heads, win=win)
            for _ in range(depth)
        ])
        # 3 lead-time heads
        self.heads = nn.ModuleDict({
            lt: nn.Conv2d(embed_dim, out_channels, kernel_size=1)
            for lt in self.LEAD_TIMES
        })
        self.active_head = "6h"

    def set_lead_time(self, lead: str):
        assert lead in self.LEAD_TIMES, f"lead must be one of {list(self.LEAD_TIMES)}"
        self.active_head = lead

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        x = self.input_proj(x)              # (B, embed, H, W)
        x = x.permute(0, 2, 3, 1)          # (B, H, W, embed)
        for blk in self.blocks:
            x = blk(x)
        x = x.permute(0, 3, 1, 2)          # (B, embed, H, W)
        return self.heads[self.active_head](x)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

_REGISTRY = {
    "fourcastnet": FourCastNetLite,
    "graphcast":   GraphCastLite,
    "pangu":       PanguLite,
}

LEAD_TIME_STEPS = {"6h": 1, "24h": 4, "72h": 12}


def get_model(name: str, in_ch: int, H: int, W: int,
              lead_time: str = "6h", **kwargs) -> nn.Module:
    """
    Instantiate a model by name.
    name: 'fourcastnet' | 'graphcast' | 'pangu'
    """
    key = name.lower().replace("lite", "").replace(" ", "").replace("-", "")
    if key not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(_REGISTRY)}")
    model = _REGISTRY[key](in_channels=in_ch, out_channels=in_ch, img_h=H, img_w=W, **kwargs)
    if hasattr(model, "set_lead_time"):
        model.set_lead_time(lead_time)
    return model


# Keep backward-compat alias used in Part 1
def build_model(name: str, in_channels: int, out_channels: int,
                img_h: int, img_w: int, **kwargs) -> nn.Module:
    key = (name.lower()
           .replace("lite", "")
           .replace("fourcastnet", "fourcastnet")
           .replace("graphcast", "graphcast")
           .replace("pangu", "pangu"))
    if key not in _REGISTRY:
        key = {
            "fourcastnetlite": "fourcastnet",
            "graphcastlite":   "graphcast",
            "pangulite":       "pangu",
        }.get(name.lower(), name.lower())
    return _REGISTRY[key](in_channels=in_channels, out_channels=out_channels,
                          img_h=img_h, img_w=img_w, **kwargs)
