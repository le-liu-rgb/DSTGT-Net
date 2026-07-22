"""
LSTM Encoder-Decoder for sequence prediction.
- Encoder: LSTM processes input sequence, final hidden state used as context.
- Decoder: LSTM autoregressively generates predictions (teacher forcing during training).
"""

import torch
import torch.nn as nn


class LSTMSeq2Seq(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = 4
        self.hidden_size = getattr(config, 'lstm_hidden_size', 128)
        self.num_layers = getattr(config, 'lstm_num_layers', 2)
        self.seq_len = config.init_seqlen
        self.pred_len = config.max_seqlen - config.init_seqlen

        self.encoder = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.decoder = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.fc_out = nn.Linear(self.hidden_size, self.input_dim)

    def forward(self, x, masks=None, with_targets=False, **kwargs):
        """
        x: [B, L, 4] 若 with_targets=True，则 L = seq_len + pred_len
           否则 L = seq_len
        """
        if with_targets:
            src = x[:, :self.seq_len, :]          # [B, seq_len, 4]
            tgt = x[:, self.seq_len:, :]          # [B, pred_len, 4]
        else:
            src = x
            tgt = None

        # Encoder
        enc_out, (h_n, c_n) = self.encoder(src)   # h_n: [num_layers, B, hidden]

        # Decoder: use last hidden state as initial
        dec_input = torch.zeros(src.size(0), 1, self.input_dim, device=x.device)  # start token (zeros)
        decoder_hidden = (h_n, c_n)
        outputs = []

        for t in range(self.pred_len):
            dec_out, decoder_hidden = self.decoder(dec_input, decoder_hidden)
            pred = self.fc_out(dec_out)           # [B, 1, 4]
            outputs.append(pred)
            dec_input = pred if with_targets else pred  # teacher forcing if targets exist else use own prediction

        pred_seq = torch.cat(outputs, dim=1)      # [B, pred_len, 4]

        if with_targets and tgt is not None:
            loss = nn.MSELoss()(pred_seq, tgt)
            return pred_seq, loss
        else:
            return pred_seq, None