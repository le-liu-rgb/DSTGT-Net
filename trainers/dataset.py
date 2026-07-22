import numpy as np
import torch
from torch.utils.data import Dataset

class AISDataset(Dataset):
    def __init__(self, l_data, max_seqlen=96, dtype=torch.float32, device=torch.device("cpu")):
        self.max_seqlen = max_seqlen
        self.device = device
        self.l_data = l_data

    def __len__(self):
        return len(self.l_data)

    def __getitem__(self, idx):
        V = self.l_data[idx]
        m_v = V["traj"][:, :4]
        m_v[m_v == 1] = 0.9999
        seqlen = min(len(m_v), self.max_seqlen)
        seq = np.zeros((self.max_seqlen, 4))
        seq[:seqlen, :] = m_v[:seqlen, :]
        seq = torch.tensor(seq, dtype=torch.float32)

        mask = torch.zeros(self.max_seqlen)
        mask[:seqlen] = 1.

        seqlen = torch.tensor(seqlen, dtype=torch.int)
        mmsi = torch.tensor(V["mmsi"], dtype=torch.int)
        time_stamps = torch.tensor(V["traj"][:self.max_seqlen, 4], dtype=torch.int32)
        if len(time_stamps) < self.max_seqlen:
            time_stamps = torch.cat(
                [time_stamps, torch.zeros(self.max_seqlen - len(time_stamps), dtype=torch.int32)])
        return seq, mask, seqlen, mmsi, time_stamps


class AISDataset_grad(Dataset):
    def __init__(self, l_data, dlat_max=0.04, dlon_max=0.04, max_seqlen=96,
                 dtype=torch.float32, device=torch.device("cpu")):
        self.dlat_max = dlat_max
        self.dlon_max = dlon_max
        self.dpos_max = np.array([dlat_max, dlon_max])
        self.max_seqlen = max_seqlen
        self.device = device
        self.l_data = l_data

    def __len__(self):
        return len(self.l_data)

    def __getitem__(self, idx):
        V = self.l_data[idx]
        m_v = V["traj"][:, :4]
        m_v[m_v == 1] = 0.9999
        seqlen = min(len(m_v), self.max_seqlen)
        seq = np.zeros((self.max_seqlen, 4))
        seq[:seqlen, :2] = m_v[:seqlen, :2]
        dpos = (m_v[1:, :2] - m_v[:-1, :2] + self.dpos_max) / (2 * self.dpos_max)
        dpos = np.concatenate((dpos[:1, :], dpos), axis=0)
        dpos[dpos >= 1] = 0.9999
        dpos[dpos <= 0] = 0.0
        seq[:seqlen, 2:] = dpos[:seqlen, :2]
        seq = torch.tensor(seq, dtype=torch.float32)

        mask = torch.zeros(self.max_seqlen)
        mask[:seqlen] = 1.

        seqlen = torch.tensor(seqlen, dtype=torch.int)
        mmsi = torch.tensor(V["mmsi"], dtype=torch.int)
        time_start = torch.tensor(V["traj"][0, 4], dtype=torch.int)
        return seq, mask, seqlen, mmsi, time_start