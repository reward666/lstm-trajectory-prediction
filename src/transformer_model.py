import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Transformer 本身不天然知道时间顺序，
    所以需要给每一帧加上位置编码。
    """

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [B, T, d_model]
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerTrajectoryModel(nn.Module):
    """
    输入：
        历史轨迹 [B, 30, 2]

    输出：
        未来轨迹 [B, 50, 2]
    """

    def __init__(
        self,
        input_dim=2,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        output_len=50,
    ):
        super().__init__()

        self.output_len = output_len

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.output_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, output_len * 2),
        )

    def forward(self, x):
        """
        x: [B, 30, 2]
        """

        x = self.input_proj(x)      # [B, 30, 64]
        x = self.pos_encoder(x)     # 加位置编码

        encoded = self.encoder(x)   # [B, 30, 64]

        last_token = encoded[:, -1, :]  # [B, 64]

        out = self.output_head(last_token)  # [B, 100]
        out = out.view(-1, self.output_len, 2)

        return out