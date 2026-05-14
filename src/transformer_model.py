import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for time-ordered trajectory tokens.
    """

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor with shape [B, T, d_model].
        """
        seq_len = x.size(1)
        return self.dropout(x + self.pe[:, :seq_len, :])


class TransformerTrajectoryModel(nn.Module):
    """
    Plain Transformer encoder-decoder baseline for trajectory prediction.

    The model intentionally uses only the target vehicle history and does not use
    social context, maneuver classes, or multi-modal heads. During training, pass
    the future ground-truth positions as ``target`` to use teacher forcing. During
    evaluation/inference, omit ``target`` and the decoder autoregressively feeds
    back its previous prediction.

    Input:
        history: [B, history_len, input_dim]
    Output:
        future positions: [B, output_len, output_dim]
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
        output_dim=2,
        num_encoder_layers=None,
        num_decoder_layers=None,
        max_len=500,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")

        self.output_len = output_len
        self.output_dim = output_dim
        encoder_layers = num_encoder_layers or num_layers
        decoder_layers = num_decoder_layers or num_layers

        self.input_proj = nn.Linear(input_dim, d_model)
        self.target_proj = nn.Linear(output_dim, d_model)
        self.encoder_pos = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.decoder_pos = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.output_head = nn.Linear(d_model, output_dim)

    @staticmethod
    def _causal_mask(seq_len, device):
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

    def _encode(self, history):
        memory = self.input_proj(history)
        memory = self.encoder_pos(memory)
        return self.encoder(memory)

    def _decode(self, decoder_input, memory):
        tgt = self.target_proj(decoder_input)
        tgt = self.decoder_pos(tgt)
        tgt_mask = self._causal_mask(tgt.size(1), tgt.device)
        decoded = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
        return self.output_head(decoded)

    def _build_teacher_forcing_input(self, history, target):
        start = history[:, -1:, : self.output_dim]
        return torch.cat([start, target[:, :-1, : self.output_dim]], dim=1)

    def forward(self, history, target=None):
        """
        Args:
            history: [B, history_len, input_dim]
            target: optional [B, output_len, output_dim] for teacher forcing.
        """
        memory = self._encode(history)

        if target is not None:
            decoder_input = self._build_teacher_forcing_input(history, target)
            return self._decode(decoder_input, memory)

        decoder_input = history[:, -1:, : self.output_dim]
        predictions = []

        for _ in range(self.output_len):
            step_predictions = self._decode(decoder_input, memory)
            next_step = step_predictions[:, -1:, :]
            predictions.append(next_step)
            decoder_input = torch.cat([decoder_input, next_step], dim=1)

        return torch.cat(predictions, dim=1)
