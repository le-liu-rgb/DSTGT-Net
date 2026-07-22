import torch
from torch.nn import functional as F
from tools import tensor_ops

@torch.no_grad()
def sample(model, seqs, steps, temperature=1.0, sample=False,
           sample_mode="pos_vicinity", r_vicinity=20, top_k=None):

    max_seqlen = model.get_max_seqlen()
    model.eval()
    for _ in range(steps):
        seqs_cond = seqs if seqs.size(1) <= max_seqlen else seqs[:, -max_seqlen:]
        logits, _ = model(seqs_cond)
        d2inf_pred = torch.zeros((logits.shape[0], 4), device=seqs.device) + 0.5

        logits = logits[:, -1, :] / temperature
        lat_logits, lon_logits, sog_logits, cog_logits = \
            torch.split(logits, (model.lat_size, model.lon_size, model.sog_size, model.cog_size), dim=-1)

        if sample_mode in ("pos_vicinity",):
            idxs, idxs_uniform = model.to_indexes(seqs_cond[:, -1:, :])
            lat_idxs, lon_idxs = idxs_uniform[:, 0, 0:1], idxs_uniform[:, 0, 1:2]
            lat_logits = tensor_ops.top_k_nearest_idx(lat_logits, lat_idxs, r_vicinity)
            lon_logits = tensor_ops.top_k_nearest_idx(lon_logits, lon_idxs, r_vicinity)

        if top_k is not None:
            lat_logits = tensor_ops.top_k_logits(lat_logits, top_k)
            lon_logits = tensor_ops.top_k_logits(lon_logits, top_k)
            sog_logits = tensor_ops.top_k_logits(sog_logits, top_k)
            cog_logits = tensor_ops.top_k_logits(cog_logits, top_k)

        lat_probs = F.softmax(lat_logits, dim=-1)
        lon_probs = F.softmax(lon_logits, dim=-1)
        sog_probs = F.softmax(sog_logits, dim=-1)
        cog_probs = F.softmax(cog_logits, dim=-1)

        if sample:
            lat_ix = torch.multinomial(lat_probs, num_samples=1)
            lon_ix = torch.multinomial(lon_probs, num_samples=1)
            sog_ix = torch.multinomial(sog_probs, num_samples=1)
            cog_ix = torch.multinomial(cog_probs, num_samples=1)
        else:
            _, lat_ix = torch.topk(lat_probs, k=1, dim=-1)
            _, lon_ix = torch.topk(lon_probs, k=1, dim=-1)
            _, sog_ix = torch.topk(sog_probs, k=1, dim=-1)
            _, cog_ix = torch.topk(cog_probs, k=1, dim=-1)

        ix = torch.cat((lat_ix, lon_ix, sog_ix, cog_ix), dim=-1)
        x_sample = (ix.float() + d2inf_pred) / model.att_sizes
        seqs = torch.cat((seqs, x_sample.unsqueeze(1)), dim=1)

    return seqs