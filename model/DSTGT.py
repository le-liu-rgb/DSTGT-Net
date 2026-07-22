"""
Full DSTGT model that combines TemporalEncoder and GraphSAGE.
"""

import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
import os

from model.TSGT_block import TemporalEncoder
from model.STGC_block import GraphSAGE

logger = logging.getLogger(__name__)

class DSTGT(nn.Module):
    def __init__(self, config, partition_model=None):
        super().__init__()
        self.lat_size = config.lat_size
        self.lon_size = config.lon_size
        self.sog_size = config.sog_size
        self.cog_size = config.cog_size
        self.full_size = config.full_size
        self.n_lat_embd = config.n_lat_embd
        self.n_lon_embd = config.n_lon_embd
        self.n_sog_embd = config.n_sog_embd
        self.n_cog_embd = config.n_cog_embd
        self.register_buffer("att_sizes", torch.tensor([config.lat_size, config.lon_size,
                                                        config.sog_size, config.cog_size]))
        self.register_buffer("emb_sizes", torch.tensor([config.n_lat_embd, config.n_lon_embd,
                                                        config.n_sog_embd, config.n_cog_embd]))

        self.partition_mode = getattr(config, "partition_mode", "uniform")
        self.partition_model = partition_model

        # Blur settings (for loss smoothing)
        self.blur = config.blur
        self.blur_learnable = config.blur_learnable
        self.blur_loss_w = config.blur_loss_w
        self.blur_n = config.blur_n
        if self.blur:
            self.blur_module = nn.Conv1d(1, 1, 3, padding=1, padding_mode='replicate', groups=1, bias=False)
            if not self.blur_learnable:
                for p in self.blur_module.parameters():
                    p.requires_grad = False
                    p.fill_(1/3)
        else:
            self.blur_module = None

        # ROI bounds (for Haversine distance)
        if hasattr(config, "lat_min"):
            self.lat_min = config.lat_min
            self.lat_max = config.lat_max
            self.lon_min = config.lon_min
            self.lon_max = config.lon_max
            self.lat_range = config.lat_max - config.lat_min
            self.lon_range = config.lon_max - config.lon_min
            self.sog_range = 30.

        self.mode = getattr(config, "mode", "pos")

        # Temporal encoder
        self.temporal_encoder = TemporalEncoder(config)

        # Classification head (outputs logits for each feature)
        if self.mode in ("mlp_pos", "mlp"):
            self.head = nn.Linear(config.n_embd, config.n_embd, bias=False)
        else:
            self.head = nn.Linear(config.n_embd, self.full_size, bias=False)

        self.max_seqlen = config.max_seqlen
        self.apply(self._init_weights)

        # Graph fusion module
        self.fea_proj = nn.Linear(config.n_embd, 4)
        self.graph_sage = GraphSAGE(4)   # input 4 features (lat, lon, sog, cog)

        self.graph_save_dir = getattr(config, "graph_save_dir", None)
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def set_graph_save_info(self, epoch, it):
        self.current_epoch = epoch
        self.current_iter = it

    def get_max_seqlen(self):
        return self.max_seqlen

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        decay = set()
        no_decay = set()
        whitelist = (nn.Linear, nn.Conv1d, GraphSAGE)
        blacklist = (nn.LayerNorm, nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist):
                    no_decay.add(fpn)
        no_decay.add('pos_emb')

        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)

    def to_indexes(self, x, mode="uniform"):
        """Convert continuous values to discrete indices."""
        bs, seqlen, _ = x.shape
        if mode == "uniform":
            idxs = (x * self.att_sizes).long()
            return idxs, idxs
        elif mode in ("freq", "freq_uniform"):
            idxs = (x * self.att_sizes).long()
            idxs_uniform = idxs.clone()
            _, _, lat_ids, lon_ids = self.partition_model(x[:, :, :2])
            idxs[:, :, 0] = torch.round(lat_ids.reshape(bs, seqlen)).long()
            idxs[:, :, 1] = torch.round(lon_ids.reshape(bs, seqlen)).long()
            return idxs, idxs_uniform
        else:
            raise ValueError(f"Unknown partition mode: {mode}")

    def count_parameters(self):
        total = 0
        for name, module in self.named_children():
            num = sum(p.numel() for p in module.parameters())
            total += num
            print(f"{name:20s} : {num:,} params")
        print(f"{'Total':20s} : {total:,} params")
        return total

    def forward(self, x, masks=None, with_targets=False, return_loss_tuple=False, time_stamps=None):
        """
        Args:
            x: input tensor (batch, seq_len, 4) in [0,1)
            masks: padding mask (batch, seq_len)
            with_targets: if True, use teacher forcing (inputs = x[:,:-1], targets = x[:,1:])
            return_loss_tuple: if True, return per-feature losses
            time_stamps: timestamps for graph construction
        Returns:
            logits, loss (and optionally loss_tuple)
        """
        # Convert to indices
        if self.mode in ("mlp_pos", "mlp"):
            idxs, idxs_uniform = x, x
        else:
            idxs, idxs_uniform = self.to_indexes(x, mode=self.partition_mode)

        if with_targets:
            inputs = idxs[:, :-1, :].contiguous()
            targets = idxs[:, 1:, :].contiguous()
        else:
            inputs = idxs
            targets = None

        batchsize, seqlen, _ = inputs.size()
        assert seqlen <= self.max_seqlen, "Sequence too long"

        # Temporal encoding
        fea = self.temporal_encoder(inputs)   # (batch, seqlen, n_embd)

        # Classification head
        logits = self.head(fea)               # (batch, seqlen, full_size)
        lat_logits, lon_logits, sog_logits, cog_logits = \
            torch.split(logits, (self.lat_size, self.lon_size, self.sog_size, self.cog_size), dim=-1)

        # Compute loss if targets are provided
        loss = None
        loss_tuple = None
        if targets is not None:
            sog_loss = F.cross_entropy(sog_logits.view(-1, self.sog_size), targets[:, :, 2].view(-1),
                                       reduction="none").view(batchsize, seqlen)
            cog_loss = F.cross_entropy(cog_logits.view(-1, self.cog_size), targets[:, :, 3].view(-1),
                                       reduction="none").view(batchsize, seqlen)
            lat_loss = F.cross_entropy(lat_logits.view(-1, self.lat_size), targets[:, :, 0].view(-1),
                                       reduction="none").view(batchsize, seqlen)
            lon_loss = F.cross_entropy(lon_logits.view(-1, self.lon_size), targets[:, :, 1].view(-1),
                                       reduction="none").view(batchsize, seqlen)

            if self.blur:
                lat_probs = F.softmax(lat_logits, dim=-1)
                lon_probs = F.softmax(lon_logits, dim=-1)
                sog_probs = F.softmax(sog_logits, dim=-1)
                cog_probs = F.softmax(cog_logits, dim=-1)
                for _ in range(self.blur_n):
                    lat_probs = self.blur_module(lat_probs.reshape(-1, 1, self.lat_size)).reshape(lat_probs.shape)
                    lon_probs = self.blur_module(lon_probs.reshape(-1, 1, self.lon_size)).reshape(lon_probs.shape)
                    sog_probs = self.blur_module(sog_probs.reshape(-1, 1, self.sog_size)).reshape(sog_probs.shape)
                    cog_probs = self.blur_module(cog_probs.reshape(-1, 1, self.cog_size)).reshape(cog_probs.shape)

                    lat_loss += self.blur_loss_w * F.nll_loss(lat_probs.view(-1, self.lat_size), targets[:, :, 0].view(-1),
                                                             reduction="none").view(batchsize, seqlen)
                    lon_loss += self.blur_loss_w * F.nll_loss(lon_probs.view(-1, self.lon_size), targets[:, :, 1].view(-1),
                                                             reduction="none").view(batchsize, seqlen)
                    sog_loss += self.blur_loss_w * F.nll_loss(sog_probs.view(-1, self.sog_size), targets[:, :, 2].view(-1),
                                                             reduction="none").view(batchsize, seqlen)
                    cog_loss += self.blur_loss_w * F.nll_loss(cog_probs.view(-1, self.cog_size), targets[:, :, 3].view(-1),
                                                             reduction="none").view(batchsize, seqlen)

            loss_tuple = (lat_loss, lon_loss, sog_loss, cog_loss)
            loss = sum(loss_tuple)
            if masks is not None:
                # masks shape: (batch, seqlen) but we need to align with seqlen
                # masks passed may be for full sequence; we use the same length as inputs
                loss = (loss * masks[:, :seqlen]).sum(dim=1) / masks[:, :seqlen].sum(dim=1)
            loss = loss.mean()

        # Graph convolution branch (if time_stamps provided)
        if time_stamps is not None:
            graph_load_path = None
            graph_save_path = None
            if self.graph_save_dir is not None:
                it = getattr(self, "current_iter", 0) + 1
                epoch = getattr(self, "current_epoch", 0)
                graph_load_path = os.path.join(self.graph_save_dir, f"graph_epoch{epoch}_iter{it}.pt")
                graph_save_path = graph_load_path
            x_graph = self.graph_sage(x, time_stamps, load_path=graph_load_path, save_path=graph_save_path)
            x_graph = x_graph[:, :120, :]   # limit to 120 steps (as in original)
            fea = self.fea_proj(fea) + x_graph   # fusion

        if return_loss_tuple:
            return logits, loss, loss_tuple
        else:
            return logits, loss