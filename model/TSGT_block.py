"""
Temporal Encoder component for DSTGT.
Contains embedding layers, positional encoding, and Transformer blocks.
"""

import torch
import torch.nn as nn
from model.Causalself_Att_block import Block

class TemporalEncoder(nn.Module):
    """
    Encoder that processes the input sequence using Transformer blocks.
    It embeds each feature (lat, lon, sog, cog) separately, adds positional
    encoding, and passes through a stack of Blocks.
    """
    def __init__(self, config):
        super().__init__()
        self.lat_size = config.lat_size
        self.lon_size = config.lon_size
        self.sog_size = config.sog_size
        self.cog_size = config.cog_size
        self.n_lat_embd = config.n_lat_embd
        self.n_lon_embd = config.n_lon_embd
        self.n_sog_embd = config.n_sog_embd
        self.n_cog_embd = config.n_cog_embd
        self.n_embd = config.n_embd
        self.max_seqlen = config.max_seqlen

        # Embedding layers for each feature
        self.lat_emb = nn.Embedding(self.lat_size, self.n_lat_embd)
        self.lon_emb = nn.Embedding(self.lon_size, self.n_lon_embd)
        self.sog_emb = nn.Embedding(self.sog_size, self.n_sog_embd)
        self.cog_emb = nn.Embedding(self.cog_size, self.n_cog_embd)

        # Positional embedding (learnable)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.max_seqlen, self.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # Transformer blocks
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.n_embd)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def forward(self, idxs):
        """
        Args:
            idxs: Tensor of shape (batch, seq_len, 4) with indices for each feature.
        Returns:
            fea: Tensor of shape (batch, seq_len, n_embd) after Transformer encoding.
        """
        batchsize, seqlen, _ = idxs.size()
        assert seqlen <= self.max_seqlen, "Sequence too long"

        lat_emb = self.lat_emb(idxs[:, :, 0])
        lon_emb = self.lon_emb(idxs[:, :, 1])
        sog_emb = self.sog_emb(idxs[:, :, 2])
        cog_emb = self.cog_emb(idxs[:, :, 3])
        token_emb = torch.cat((lat_emb, lon_emb, sog_emb, cog_emb), dim=-1)

        pos_emb = self.pos_emb[:, :seqlen, :]
        fea = self.drop(token_emb + pos_emb)
        fea = self.blocks(fea)
        fea = self.ln_f(fea)   # (batch, seqlen, n_embd)
        return fea