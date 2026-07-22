"""
Standard Transformer model for sequence prediction (encoder-decoder).
- Encoder: processes input sequence.
- Decoder: generates output sequence with masking.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]


class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = 4
        self.d_model = getattr(config, 'transformer_d_model', 256)
        self.nhead = getattr(config, 'transformer_nhead', 8)
        self.num_encoder_layers = getattr(config, 'transformer_num_encoder_layers', 3)
        self.num_decoder_layers = getattr(config, 'transformer_num_decoder_layers', 3)
        self.dim_feedforward = getattr(config, 'transformer_dim_feedforward', 512)
        self.dropout = getattr(config, 'transformer_dropout', 0.1)
        self.seq_len = config.init_seqlen
        self.pred_len = config.max_seqlen - config.init_seqlen

        self.embedding = nn.Linear(self.input_dim, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)
        self.pos_decoder = PositionalEncoding(self.d_model)

        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(self.d_model, self.input_dim)

    def forward(self, x, masks=None, with_targets=False, **kwargs):
        if with_targets:
            src = x[:, :self.seq_len, :]          # [B, seq_len, 4]
            tgt = x[:, self.seq_len:, :]          # [B, pred_len, 4]
        else:
            src = x
            tgt = None

        # Embed and add position
        src_emb = self.pos_encoder(self.embedding(src))    # [B, seq_len, d_model]
        # For decoder, we use a learnable start token, but here we use zeros as start
        # and shift target for training (teacher forcing). We'll create a causal mask.
        if with_targets and tgt is not None:
            # Use target embedding for decoder input (shifted right by one)
            tgt_input = torch.cat([torch.zeros(src.size(0), 1, self.d_model, device=x.device),
                                   self.embedding(tgt[:, :-1, :])], dim=1)  # [B, pred_len, d_model]
            tgt_input = self.pos_decoder(tgt_input)
            # Causal mask
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.pred_len).to(x.device)
            memory = self.transformer.encoder(src_emb)
            output = self.transformer.decoder(tgt_input, memory, tgt_mask=tgt_mask)
        else:
            # Inference: generate from scratch using empty decoder input (or zeros)
            # We'll generate using a loop or direct prediction with zero input (simplified)
            # For simplicity, we'll just use zeros as decoder input and no mask (autoregressive not used)
            tgt_input = torch.zeros(src.size(0), self.pred_len, self.d_model, device=x.device)
            tgt_input = self.pos_decoder(tgt_input)
            memory = self.transformer.encoder(src_emb)
            output = self.transformer.decoder(tgt_input, memory)

        pred = self.fc_out(output)  # [B, pred_len, 4]

        if with_targets and tgt is not None:
            loss = nn.MSELoss()(pred, tgt)
            return pred, loss
        else:
            return pred, None