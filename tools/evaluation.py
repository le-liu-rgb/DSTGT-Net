import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from trainers.sample import sample
from tools.geo_utils import haversine

def evaluate_model(model, dataloader, config, device, init_seqlen, n_samples=16):
    model.eval()
    v_ranges = torch.tensor([2, 3, 0, 0]).to(device)
    v_roi_min = torch.tensor([model.lat_min, -7, 0, 0]).to(device)
    max_seqlen = init_seqlen + 6 * 15

    l_min_errors, l_mean_errors, l_masks = [], [], []
    seqs_list, masks_list, seqlens_list = [], [], []

    with torch.no_grad():
        for seqs, masks, seqlens, mmsis, time_starts in tqdm(dataloader):
            seqs_init = seqs[:, :init_seqlen, :].to(device)
            masks = masks[:, :max_seqlen].to(device)
            batchsize = seqs.shape[0]
            error_ens = torch.zeros((batchsize, max_seqlen - init_seqlen, n_samples)).to(device)

            for i_sample in range(n_samples):
                preds = sample(model, seqs_init, max_seqlen - init_seqlen,
                               temperature=1.0, sample=True,
                               sample_mode=config.sample_mode,
                               r_vicinity=config.r_vicinity,
                               top_k=config.top_k)
                inputs = seqs[:, :max_seqlen, :].to(device)
                input_coords = (inputs * v_ranges + v_roi_min) * torch.pi / 180
                pred_coords = (preds * v_ranges + v_roi_min) * torch.pi / 180
                d = haversine(input_coords, pred_coords) * masks
                error_ens[:, :, i_sample] = d[:, init_seqlen:]

            l_min_errors.append(error_ens.min(dim=-1))
            l_mean_errors.append(error_ens.mean(dim=-1))
            l_masks.append(masks[:, init_seqlen:])
            seqs_list.append(seqs)
            masks_list.append(masks)
            seqlens_list.extend(seqlens.tolist())

    min_errors = torch.cat([x.values for x in l_min_errors], dim=0) * torch.cat(l_masks, dim=0)
    masks_all = torch.cat(l_masks, dim=0)
    mae = (min_errors * masks_all).sum() / masks_all.sum()
    mae = mae.item()

    # MRE
    seqs_all = torch.cat(seqs_list, dim=0)
    masks_all_for_mre = torch.cat(masks_list, dim=0)
    true_lengths = []
    for seq, mask, seqlen in zip(seqs_all, masks_all_for_mre, seqlens_list):
        valid_traj = seq[:seqlen]
        coords = (valid_traj[:, :2] * v_ranges[:2] + v_roi_min[:2]) * torch.pi / 180
        dists = haversine(coords[:-1], coords[1:])
        true_lengths.append(dists.sum().item())
    true_lengths = torch.tensor(true_lengths).to(device)
    rel_errors = min_errors.sum(dim=1) / (masks_all.sum(dim=1) * true_lengths + 1e-8)
    mre = rel_errors.mean().item()
    return mae, mre