import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('affine_weight', None)
            self.register_parameter('affine_bias', None)

    def forward(self, x, mode:str):
        # x: [B, L, D]
        if mode == 'norm':
            self._get_statistics(x)
            x = (x - self.mean) / (self.std + self.eps)
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self.std + self.mean
            return x
        else:
            raise NotImplementedError

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.std = torch.std(x, dim=dim2reduce, keepdim=True).detach()


class TimeBridge(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_ff = config.d_ff
        self.num_ia_layers = config.ia_layers
        self.num_pd_layers = config.pd_layers
        self.num_ca_layers = config.ca_layers
        self.period = config.period
        self.stable_len = config.stable_len
        self.num_p = config.num_p if config.num_p is not None else config.seq_len // config.period
        self.dropout = config.dropout
        self.attn_dropout = config.attn_dropout
        self.activation = config.activation
        self.revin = config.revin

        self.input_proj = nn.Linear(4, self.d_model)
        self.pos_enc = nn.Parameter(torch.zeros(1, self.seq_len + self.pred_len, self.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_ia_layers)
        if self.num_pd_layers > 0:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=self.n_heads,
                dim_feedforward=self.d_ff,
                dropout=self.dropout,
                activation=self.activation,
                batch_first=True
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_pd_layers)
        else:
            self.decoder = None
        self.output_proj = nn.Linear(self.d_model, 4)
        if self.revin:
            self.revin_layer = RevIN(4)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, masks=None, with_targets=False, **kwargs):

        B, L, _ = x.shape
        if with_targets:
            src = x[:, :self.seq_len, :]          # [B, seq_len, 4]
            tgt = x[:, self.seq_len:, :]          # [B, pred_len, 4]
        else:
            src = x                               # [B, seq_len, 4]
            tgt = None

        if self.revin:
            src = self.revin_layer(src, 'norm')

        src_emb = self.input_proj(src)            # [B, seq_len, d_model]
        src_emb = src_emb + self.pos_enc[:, :self.seq_len, :]

        memory = self.encoder(src_emb)            # [B, seq_len, d_model]
        if self.decoder is not None:
            tgt_emb = torch.zeros(B, self.pred_len, self.d_model, device=x.device)
            tgt_emb = tgt_emb + self.pos_enc[:, self.seq_len:self.seq_len+self.pred_len, :]
            pred_emb = self.decoder(tgt_emb, memory)   # [B, pred_len, d_model]
        else:

            last_state = memory[:, -1, :]            # [B, d_model]
            pred_emb = last_state.unsqueeze(1).repeat(1, self.pred_len, 1)
            pred_emb = pred_emb + self.pos_enc[:, self.seq_len:self.seq_len+self.pred_len, :]
        pred = self.output_proj(pred_emb)            # [B, pred_len, 4]

        if self.revin:
            pred = self.revin_layer(pred, 'denorm')

        if with_targets:
            loss = self.time_freq_mae(tgt, pred, alpha=0.2)
            return pred, loss
        else:
            return pred, None

    def time_freq_mae(self, y_true, y_pred, alpha=0.2):
        t_loss = (y_pred - y_true).abs().mean()
        fft_true = torch.fft.rfft(y_true, dim=1)
        fft_pred = torch.fft.rfft(y_pred, dim=1)
        f_loss = (fft_pred - fft_true).abs().mean()
        return (1 - alpha) * t_loss + alpha * f_loss