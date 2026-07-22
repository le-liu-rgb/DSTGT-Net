import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import os

class GraphSAGE(nn.Module):
    def __init__(self, full_size):
        super().__init__()
        self.conv1 = SAGEConv(full_size, 256)
        self.conv2 = SAGEConv(256, full_size)
        self.full_size = full_size

    def build_spatiotemporal_graph(self, x, time_stamps, save_path=None):
        batch_size, seq_len, _ = x.shape
        edges = []
        for b in range(batch_size):
            for i in range(seq_len):
                for j in range(seq_len):
                    if i != j:
                        time_diff = abs(time_stamps[b, i] - time_stamps[b, j])
                        if time_diff <= 15 * 60:   # 15 minutes
                            lat_i, lon_i = x[b, i, 0], x[b, i, 1]
                            lat_j, lon_j = x[b, j, 0], x[b, j, 1]
                            distance = torch.sqrt((lat_i - lat_j)**2 + (lon_i - lon_j)**2) * 111000
                            if distance < 1000:
                                edges.append([i, j])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(x.device)
        if save_path is not None:
            torch.save(edge_index, save_path)
        return edge_index

    def forward(self, x, time_stamps, edge_index=None, load_path=None, save_path=None):
        batch_size, seq_len, _ = x.shape
        if load_path is not None and os.path.exists(load_path):
            edge_index = torch.load(load_path).to(x.device)
        else:
            edge_index = self.build_spatiotemporal_graph(x, time_stamps, save_path=save_path)
        x_graph = x.view(-1, x.shape[-1])
        graph_data = Data(x=x_graph, edge_index=edge_index)
        x_graph = self.conv1(graph_data.x, graph_data.edge_index)
        x_graph = F.relu(x_graph)
        x_graph = self.conv2(x_graph, graph_data.edge_index)
        x_graph = x_graph.view(batch_size, seq_len, -1)
        return x_graph