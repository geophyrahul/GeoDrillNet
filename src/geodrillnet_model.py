"""GeoDrillNet (Experiment 008) — CNN + Transformer model for per-foot TVT regression.

Model classes are taken verbatim from horizontal_well_geology_cnn_transformer_fixed.ipynb;
the config values are the Optuna-selected ones in outputs_cnn_transformer/models/final_config.json.
"""
import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class Config:
    CNN_CHANNELS: Tuple[int, ...] = (32, 64, 128, 128)
    CNN_KERNEL_SIZE: int = 7
    CNN_DROPOUT: float = 0.04879845655289969
    GEOLOGY_EMBED_DIM: int = 16
    D_MODEL: int = 128
    N_TRANSFORMER_LAYERS: int = 1
    N_TRANSFORMER_HEADS: int = 2
    TRANSFORMER_FF_DIM: int = 256
    TRANSFORMER_DROPOUT: float = 0.21503639919897471
    HEAD_HIDDEN_DIM: int = 256
    HEAD_DROPOUT: float = 0.0045441726980578225
    WINDOW_SIZE: int = 256


CFG = Config()
N_CONTINUOUS_FEATURES = 17   # len(manifest["continuous_feature_cols"])
N_GEOLOGY_CLASSES = 36       # geology label-encoder vocabulary (incl. UNKNOWN)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.dropout(self.act(self.bn(self.conv(x))))
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int, channels: Tuple[int, ...], kernel_size: int, dropout: float):
        super().__init__()
        layers = []
        prev_c = in_channels
        for c in channels:
            layers.append(ConvBlock(prev_c, c, kernel_size, dropout))
            prev_c = c
        self.net = nn.Sequential(*layers)
        self.out_channels = prev_c

    def forward(self, x):  # x: [B, C_in, T]
        return self.net(x)  # [B, C_out, T]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        attn_out, attn_weights = self.attn(x, x, x, key_padding_mask=key_padding_mask,
                                            need_weights=True, average_attn_weights=True)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x, attn_weights


class TransformerStack(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([SelfAttentionLayer(d_model, n_heads, ff_dim, dropout) for _ in range(n_layers)])

    def forward(self, x, key_padding_mask=None):
        attn_weights = None
        for layer in self.layers:
            x, attn_weights = layer(x, key_padding_mask)
        return x, attn_weights  # returns the LAST layer's self-attention weights for viz


class SequenceModel(nn.Module):
    def __init__(self, n_continuous_features: int, geology_vocab: int, cfg: Config = CFG):
        super().__init__()
        self.cfg = cfg
        self.geo_embed = nn.Embedding(geology_vocab, cfg.GEOLOGY_EMBED_DIM)

        cnn_in_channels = n_continuous_features + cfg.GEOLOGY_EMBED_DIM
        self.cnn = CNNEncoder(cnn_in_channels, cfg.CNN_CHANNELS, cfg.CNN_KERNEL_SIZE, cfg.CNN_DROPOUT)

        self.input_proj = nn.Linear(self.cnn.out_channels, cfg.D_MODEL) if self.cnn.out_channels != cfg.D_MODEL else nn.Identity()
        self.pos_enc = PositionalEncoding(cfg.D_MODEL)
        self.transformer = TransformerStack(cfg.D_MODEL, cfg.N_TRANSFORMER_LAYERS, cfg.N_TRANSFORMER_HEADS,
                                             cfg.TRANSFORMER_FF_DIM, cfg.TRANSFORMER_DROPOUT)

        self.head = nn.Sequential(
            nn.Linear(cfg.D_MODEL, cfg.HEAD_HIDDEN_DIM), nn.GELU(), nn.Dropout(cfg.HEAD_DROPOUT),
            nn.Linear(cfg.HEAD_HIDDEN_DIM, 1),
        )

    def forward(self, x_cont, geo_idx, mask, return_attention: bool = False):
        geo_emb = self.geo_embed(geo_idx)
        x = torch.cat([x_cont, geo_emb], dim=-1).transpose(1, 2)  # [B, F+E, T]

        h = self.cnn(x).transpose(1, 2)  # [B, T, C_cnn]
        h = self.input_proj(h)
        h = self.pos_enc(h)

        key_padding_mask = (mask == 0)
        h, attn_weights = self.transformer(h, key_padding_mask)

        pred = self.head(h).squeeze(-1)
        if return_attention:
            return pred, attn_weights
        return pred


# ---- target scaling (the Exp 008 fix): the model predicts z-scored TVT
def fit_target_scaler(train_target: np.ndarray) -> Dict[str, float]:
    mean = float(np.mean(train_target))
    std = float(np.std(train_target, ddof=1))
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0
    return {"mean": mean, "std": std}


def inverse_transform_target(y_scaled: np.ndarray, target_stats: Dict[str, float]) -> np.ndarray:
    return y_scaled * target_stats["std"] + target_stats["mean"]


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    sq_err = (pred - target) ** 2 * mask
    return sq_err.sum() / mask.sum().clamp(min=1.0)


if __name__ == "__main__":
    model = SequenceModel(N_CONTINUOUS_FEATURES, N_GEOLOGY_CLASSES, CFG)
    print(f"trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    B, T = 4, CFG.WINDOW_SIZE
    x = torch.randn(B, T, N_CONTINUOUS_FEATURES)
    geo = torch.randint(0, N_GEOLOGY_CLASSES, (B, T))
    mask = torch.ones(B, T)
    pred, attn = model(x, geo, mask, return_attention=True)
    print("pred", tuple(pred.shape), "attention", tuple(attn.shape))
