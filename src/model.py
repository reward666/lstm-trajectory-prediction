import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Utility Blocks
# ─────────────────────────────────────────────
class ResidualMLP(nn.Module):
    """带 LayerNorm/GELU/Dropout 的残差 MLP，用于稳定深层预测头训练。"""

    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(self.norm(x))


# ─────────────────────────────────────────────
# Multi-Head Temporal Attention
# 对自身 / 邻居 LSTM 的所有时间步加权
# ─────────────────────────────────────────────
class MultiHeadTemporalAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.attn_heads = nn.Linear(hidden_dim, num_heads)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, lstm_out):
        """
        lstm_out: [B, T, hidden_dim]
        return:   [B, hidden_dim]
        """
        scores = self.attn_heads(lstm_out)  # [B, T, H]
        weights = F.softmax(scores, dim=1)

        # 向量化多头时间聚合，避免 Python for-loop，便于 GPU 并行。
        contexts = torch.einsum("bth,btd->bhd", weights, lstm_out)
        contexts = contexts.flatten(start_dim=1)  # [B, H * hidden_dim]
        return self.proj(contexts)


# ─────────────────────────────────────────────
# Multi-Head Social Attention
# 目标车辆 attend 到周围车辆，并显式处理无邻居样本
# ─────────────────────────────────────────────
class SocialAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** 0.5

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, ego_ctx, neighbor_ctx, mask=None):
        """
        ego_ctx:      [B, hidden_dim]
        neighbor_ctx: [B, K, hidden_dim]
        mask:         [B, K]  1=真实邻居 0=padding
        return:       [B, hidden_dim]
        """
        B, K, _ = neighbor_ctx.shape
        q = self.query(ego_ctx).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(neighbor_ctx).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(neighbor_ctx).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, H, 1, K]

        if mask is not None:
            valid_mask = mask.to(dtype=torch.bool).view(B, 1, 1, K)
            scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
            weights = F.softmax(scores, dim=-1).masked_fill(~valid_mask, 0.0)
            denom = weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
            weights = weights / denom
        else:
            weights = F.softmax(scores, dim=-1)

        social_ctx = torch.matmul(weights, v).transpose(1, 2).reshape(B, self.hidden_dim)
        social_ctx = self.out_proj(social_ctx)
        if mask is not None:
            social_ctx = social_ctx * mask.any(dim=1, keepdim=True).to(dtype=social_ctx.dtype)
        return social_ctx


# ─────────────────────────────────────────────
# Gated Ego-Social Fusion
# ─────────────────────────────────────────────
class GatedFusion(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, ego_ctx, social_ctx):
        gate = self.gate(torch.cat([ego_ctx, social_ctx], dim=-1))
        blended = gate * ego_ctx + (1.0 - gate) * social_ctx
        return self.fuse(torch.cat([blended, ego_ctx + social_ctx], dim=-1))


# ─────────────────────────────────────────────
# 完整模型
# ─────────────────────────────────────────────
class SocialLSTMWithAttention(nn.Module):
    def __init__(
        self,
        input_dim=2,
        hidden_dim=64,
        num_layers=2,
        output_len=50,
        num_heads=4,
        max_neighbors=5,
        num_modes=6,
        dropout=0.1,
        bidirectional=True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_len = output_len
        self.max_neighbors = max_neighbors
        self.num_modes = num_modes
        self.bidirectional = bidirectional
        self.encoder_dim = hidden_dim * (2 if bidirectional else 1)
        lstm_dropout = dropout if num_layers > 1 else 0.0

        # 先把 (x, y, v, a) 等原始特征投影到统一隐空间；input_dim=2 时也兼容旧数据。
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 自身历史编码
        self.ego_lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )

        # 邻居历史编码（共享权重，所有邻居用同一个 LSTM）
        self.neighbor_lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )

        # Multi-Head Temporal Attention（时间维度）
        self.temporal_attn = MultiHeadTemporalAttention(self.encoder_dim, num_heads, dropout)

        # Multi-Head Social Attention（跨车辆交互）
        self.social_attn = SocialAttention(self.encoder_dim, num_heads, dropout)
        self.fusion = GatedFusion(self.encoder_dim, dropout)
        self.context_refiner = ResidualMLP(self.encoder_dim, self.encoder_dim * 2, dropout)

        # 显式注入“预测模式”和“未来时刻”embedding，比一次性 flatten 输出更像可解释 decoder。
        self.mode_embedding = nn.Embedding(num_modes, self.encoder_dim)
        self.time_embedding = nn.Embedding(output_len, self.encoder_dim)

        decoder_dim = self.encoder_dim * 3
        self.traj_decoder = nn.Sequential(
            nn.Linear(decoder_dim, self.encoder_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualMLP(self.encoder_dim * 2, self.encoder_dim * 4, dropout),
            nn.Linear(self.encoder_dim * 2, 2),
        )

        # 概率预测头（每条轨迹的真实可能性）
        self.prob_head = nn.Sequential(
            nn.LayerNorm(self.encoder_dim),
            nn.Linear(self.encoder_dim, self.encoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.encoder_dim, num_modes),
        )

    def forward(self, x, x_social, mask=None):
        """
        x:        [B, history_len, input_dim]       自身历史
        x_social: [B, K, history_len, input_dim]    邻居历史
        mask:     [B, K]                            邻居有效掩码
        return:
            out_traj: [B, num_modes, output_len, 2]
            out_prob: [B, num_modes]  每个 mode 的 log-probability
        """
        B, K, T, D = x_social.shape

        # ── 自身 LSTM + Multi-Head Temporal Attention ──
        x_embed = self.input_encoder(x)
        ego_out, _ = self.ego_lstm(x_embed)
        ego_ctx = self.temporal_attn(ego_out)

        # ── 邻居 LSTM（共享权重）──
        x_nb = x_social.reshape(B * K, T, D)
        x_nb = self.input_encoder(x_nb)
        nb_out, _ = self.neighbor_lstm(x_nb)
        nb_ctx = self.temporal_attn(nb_out).view(B, K, self.encoder_dim)

        # Padding 邻居在进入 social attention 前置零，减少无效轨迹对 LayerNorm/投影的影响。
        if mask is not None:
            nb_ctx = nb_ctx * mask.unsqueeze(-1).to(dtype=nb_ctx.dtype)

        # ── Social Attention + Gated Fusion ──
        social_ctx = self.social_attn(ego_ctx, nb_ctx, mask)
        context = self.fusion(ego_ctx, social_ctx)
        context = self.context_refiner(context)

        # ── Horizon-aware Multi-modal Decoder ──
        mode_ids = torch.arange(self.num_modes, device=x.device)
        time_ids = torch.arange(self.output_len, device=x.device)
        mode_embed = self.mode_embedding(mode_ids).view(1, self.num_modes, 1, self.encoder_dim)
        time_embed = self.time_embedding(time_ids).view(1, 1, self.output_len, self.encoder_dim)
        context_embed = context.view(B, 1, 1, self.encoder_dim).expand(-1, self.num_modes, self.output_len, -1)

        decoder_input = torch.cat(
            [
                context_embed,
                mode_embed.expand(B, -1, self.output_len, -1),
                time_embed.expand(B, self.num_modes, -1, -1),
            ],
            dim=-1,
        )
        out_traj = self.traj_decoder(decoder_input)

        # ── Mode Confidence ──
        out_prob = F.log_softmax(self.prob_head(context), dim=-1)
        return out_traj, out_prob
