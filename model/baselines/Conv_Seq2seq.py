"""
Convolutional Seq2seq using 1D convolutions over time.
- Encoder: stack of Conv1d layers with dilation to capture long-range dependencies.
- Decoder: transposed convolutions (or upsampling) to generate future sequence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvSeq2Seq(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = 4
        self.seq_len = config.init_seqlen
        self.pred_len = config.max_seqlen - config.init_seqlen
        self.hidden_channels = getattr(config, 'conv_channels', 64)
        self.kernel_size = getattr(config, 'conv_kernel', 3)

        # Encoder: Conv1d over time, output shape: [B, channels, seq_len]
        self.encoder = nn.Sequential(
            nn.Conv1d(self.input_dim, self.hidden_channels, kernel_size=self.kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size=self.kernel_size, padding=1),
            nn.ReLU(),
        )
        # Decoder: use ConvTranspose1d to expand time dimension
        # We need to go from seq_len to pred_len
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
        )
        # Adjust output length to exactly pred_len via adaptive pooling
        self.adapt_pool = nn.AdaptiveAvgPool1d(self.pred_len)
        self.fc_out = nn.Conv1d(self.hidden_channels, self.input_dim, kernel_size=1)

    def forward(self, x, masks=None, with_targets=False, **kwargs):
        # x: [B, L, 4], we take first seq_len as input
        if with_targets:
            src = x[:, :self.seq_len, :]   # [B, seq_len, 4]
            tgt = x[:, self.seq_len:, :]   # [B, pred_len, 4]
        else:
            src = x
            tgt = None

        # Permute to [B, 4, seq_len]
        src = src.permute(0, 2, 1)
        enc = self.encoder(src)            # [B, channels, seq_len]
        dec = self.decoder(enc)            # [B, channels, upsampled_len]
        dec = self.adapt_pool(dec)         # [B, channels, pred_len]
        pred = self.fc_out(dec)            # [B, 4, pred_len]
        pred = pred.permute(0, 2, 1)       # [B, pred_len, 4]

        if with_targets and tgt is not None:
            loss = nn.MSELoss()(pred, tgt)
            return pred, loss
        else:
            return pred, None