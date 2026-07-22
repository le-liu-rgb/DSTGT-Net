"""
LSTM Encoder-Decoder with Bahdanau attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Bahdanau attention (additive)"""
    def __init__(self, hidden_size):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: [B, hidden]
        # encoder_outputs: [B, seq_len, hidden]
        hidden_expanded = decoder_hidden.unsqueeze(1)  # [B, 1, hidden]
        score = self.Va(torch.tanh(self.Wa(encoder_outputs) + self.Ua(hidden_expanded)))
        attn_weights = F.softmax(score.squeeze(-1), dim=1)  # [B, seq_len]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # [B, hidden]
        return context, attn_weights


class LSTMSeq2SeqAtt(nn.Module):
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
            batch_first=True,
            bidirectional=True
        )
        # Bidirectional: hidden size doubled
        self.decoder = nn.LSTM(
            input_size=self.input_dim + self.hidden_size * 2,  # concat context
            hidden_size=self.hidden_size * 2,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.attention = Attention(self.hidden_size * 2)
        self.fc_out = nn.Linear(self.hidden_size * 2, self.input_dim)

    def forward(self, x, masks=None, with_targets=False, **kwargs):
        if with_targets:
            src = x[:, :self.seq_len, :]
            tgt = x[:, self.seq_len:, :]
        else:
            src = x
            tgt = None

        # Encoder
        enc_out, (h_n, c_n) = self.encoder(src)   # enc_out: [B, seq_len, 2*hidden]
        # Use last layer hidden states (both directions) as initial decoder state
        # For simplicity, we use the concatenated final forward/backward states
        h_n = torch.cat((h_n[-2], h_n[-1]), dim=1).unsqueeze(0)  # [1, B, 2*hidden]
        c_n = torch.cat((c_n[-2], c_n[-1]), dim=1).unsqueeze(0)
        decoder_hidden = (h_n, c_n)

        dec_input = torch.zeros(src.size(0), 1, self.input_dim, device=x.device)
        outputs = []

        for t in range(self.pred_len):
            # Compute attention context
            context, _ = self.attention(decoder_hidden[0].squeeze(0), enc_out)  # [B, 2*hidden]
            dec_input_att = torch.cat([dec_input, context.unsqueeze(1)], dim=-1)  # [B, 1, input_dim+2*hidden]
            dec_out, decoder_hidden = self.decoder(dec_input_att, decoder_hidden)
            pred = self.fc_out(dec_out)          # [B, 1, 4]
            outputs.append(pred)
            dec_input = pred if with_targets else pred

        pred_seq = torch.cat(outputs, dim=1)

        if with_targets and tgt is not None:
            loss = nn.MSELoss()(pred_seq, tgt)
            return pred_seq, loss
        else:
            return pred_seq, None