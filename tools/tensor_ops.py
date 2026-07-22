import torch

def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(-1)
    return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)

def top_k_nearest_idx(logits, ref_idxs, r):
    # ref_idxs: (batch, 1)
    # logits: (batch, class_size)
    class_size = logits.size(-1)
    idx_range = torch.arange(class_size, device=logits.device).view(1, -1)
    dist = torch.abs(idx_range - ref_idxs)
    mask = dist <= r
    return torch.where(mask, logits, torch.full_like(logits, float('-inf')))