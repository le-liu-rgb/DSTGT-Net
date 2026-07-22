"""
The complete code, data preprocessing process and pre-trained model have all been open-sourced in the official repository.
This implementation supports learning probability representations from AIS trajectories and includes an a contrario anomaly detection module. Specific implementation details and training commands are provided.
For more details on the dependencies, please refer to
https://github.com/CIA-Oceanix/TrAISformer.
"""

import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
from model.Causalself_Att_block import Block   # 复用注意力块

logger = logging.getLogger(__name__)

class TrAISformer(nn.Module):
    """Transformer for AIS trajectories (baseline)"""
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

        if hasattr(config, "lat_min"):
            self.lat_min = config.lat_min
            self.lat_max = config.lat_max
            self.lon_min = config.lon_min
            self.lon_max = config.lon_max
            self.lat_range = config.lat_max - config.lat_min
            self.lon_range = config.lon_max - config.lon_min
            self.sog_range = 30.

        self.mode = getattr(config, "mode", "pos")

        self.lat_emb = nn.Embedding(self.lat_size, config.n_lat_embd)
        self.lon_emb = nn.Embedding(self.lon_size, config.n_lon_embd)
        self.sog_emb = nn.Embedding(self.sog_size, config.n_sog_embd)
        self.cog_emb = nn.Embedding(self.cog_size, config.n_cog_embd)

        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_seqlen, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)

        if self.mode in ("mlp_pos", "mlp"):
            self.head = nn.Linear(config.n_embd, config.n_embd, bias=False)
        else:
            self.head = nn.Linear(config.n_embd, self.full_size, bias=False)

        self.max_seqlen = config.max_seqlen
        self.apply(self._init_weights)
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

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
        whitelist = (nn.Linear, nn.Conv1d)
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

    def forward(self, x, masks=None, with_targets=False, return_loss_tuple=False, time_stamps=None):
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
        assert seqlen <= self.max_seqlen

        lat_emb = self.lat_emb(inputs[:, :, 0])
        lon_emb = self.lon_emb(inputs[:, :, 1])
        sog_emb = self.sog_emb(inputs[:, :, 2])
        cog_emb = self.cog_emb(inputs[:, :, 3])
        token_emb = torch.cat((lat_emb, lon_emb, sog_emb, cog_emb), dim=-1)

        pos_emb = self.pos_emb[:, :seqlen, :]
        fea = self.drop(token_emb + pos_emb)
        fea = self.blocks(fea)
        fea = self.ln_f(fea)
        logits = self.head(fea)

        lat_logits, lon_logits, sog_logits, cog_logits = \
            torch.split(logits, (self.lat_size, self.lon_size, self.sog_size, self.cog_size), dim=-1)

        loss = None
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

            loss = lat_loss + lon_loss + sog_loss + cog_loss
            if masks is not None:
                loss = (loss * masks).sum(dim=1) / masks.sum(dim=1)
            loss = loss.mean()

        if return_loss_tuple:
            return logits, loss, (lat_loss, lon_loss, sog_loss, cog_loss)
        else:
            return logits, loss