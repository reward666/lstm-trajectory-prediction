'''import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, output_len=50):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_len = output_len

        # LSTM 编码历史轨迹
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True
        )

        # 全连接层 → 输出未来轨迹
        self.fc = nn.Linear(hidden_dim, output_len * 2)

    def forward(self, x):
        """
        x: [B, 30, 2]
        """

        # LSTM 输出
        _, (h_n, _) = self.lstm(x)

        # 取最后一层 hidden state
        h = h_n[-1]   # [B, hidden_dim]

        out = self.fc(h)   # [B, 50*2]

        # reshape 成轨迹
        out = out.view(-1, self.output_len, 2)

        return out'''
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Multi-Head Temporal Attention
# 对自身 LSTM 的所有时间步加权
# ─────────────────────────────────────────────
class MultiHeadTemporalAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.attn_heads = nn.Linear(hidden_dim, num_heads)
        self.fc = nn.Linear(hidden_dim * num_heads, hidden_dim)

    def forward(self, lstm_out):
        """
        lstm_out: [B, T, hidden_dim]
        return:   [B, hidden_dim]
        """
        scores = self.attn_heads(lstm_out)       # [B, T, num_heads]
        weights = F.softmax(scores, dim=1)        # [B, T, num_heads]

        contexts = []
        for i in range(self.num_heads):
            w = weights[:, :, i:i+1]             # [B, T, 1]
            ctx = (w * lstm_out).sum(dim=1)       # [B, hidden_dim]
            contexts.append(ctx)

        out = torch.cat(contexts, dim=-1)         # [B, hidden_dim * num_heads]
        out = self.fc(out)                        # [B, hidden_dim]
        return out


# ─────────────────────────────────────────────
# Social Attention
# 目标车辆 attend 到周围车辆
# ─────────────────────────────────────────────
class SocialAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key   = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, ego_ctx, neighbor_ctx, mask=None):
        """
        ego_ctx:      [B, hidden_dim]
        neighbor_ctx: [B, K, hidden_dim]
        mask:         [B, K]  1=真实邻居 0=padding
        return:       [B, hidden_dim]
        """
        q = self.query(ego_ctx).unsqueeze(1)          # [B, 1, hidden_dim]
        k = self.key(neighbor_ctx)                     # [B, K, hidden_dim]
        v = self.value(neighbor_ctx)                   # [B, K, hidden_dim]

        scores = torch.bmm(q, k.transpose(1, 2))      # [B, 1, K]
        scores = scores / self.scale

        # 把 padding 位置的分数设为很小，softmax 后权重趋近 0
        if mask is not None:
            mask = mask.unsqueeze(1)                   # [B, 1, K]
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = F.softmax(scores, dim=-1)            # [B, 1, K]
        social_ctx = torch.bmm(weights, v).squeeze(1) # [B, hidden_dim]
        return social_ctx


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
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_len = output_len

        # 自身历史编码
        self.ego_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True
        )

        # 邻居历史编码（共享权重，所有邻居用同一个LSTM）
        self.neighbor_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True
        )

        # Multi-Head Temporal Attention（自身时间维度）
        self.temporal_attn = MultiHeadTemporalAttention(hidden_dim, num_heads)

        # Social Attention（跨车辆交互）
        self.social_attn = SocialAttention(hidden_dim)

        # 融合自身 + 社会 context 后预测
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_len * 2),
        )

    def forward(self, x, x_social, mask=None):
        """
        x:        [B, 30, 2]       自身历史
        x_social: [B, K, 30, 2]   邻居历史
        mask:     [B, K]           邻居有效掩码
        """
        B, K, T, D = x_social.shape

        # ── 自身 LSTM + Multi-Head Temporal Attention ──
        ego_out, _ = self.ego_lstm(x)             # [B, 30, hidden_dim]
        ego_ctx = self.temporal_attn(ego_out)     # [B, hidden_dim]

        # ── 邻居 LSTM（共享权重）──
        # 把 K 个邻居展平成 batch 维度一起过 LSTM
        x_nb = x_social.view(B * K, T, D)         # [B*K, 30, 2]
        nb_out, _ = self.neighbor_lstm(x_nb)      # [B*K, 30, hidden_dim]

        # 取最后一帧作为每个邻居的表示
        nb_ctx = nb_out[:, -1, :]                 # [B*K, hidden_dim]
        nb_ctx = nb_ctx.view(B, K, -1)            # [B, K, hidden_dim]

        # ── Social Attention ──
        social_ctx = self.social_attn(ego_ctx, nb_ctx, mask)  # [B, hidden_dim]

        # ── 融合 & 预测 ──
        combined = torch.cat([ego_ctx, social_ctx], dim=-1)   # [B, hidden_dim*2]
        out = self.fc(combined)                               # [B, 50*2]
        out = out.view(B, self.output_len, 2)                 # [B, 50, 2]

        return out