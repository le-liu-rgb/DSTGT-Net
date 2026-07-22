import torch

def haversine(coord1, coord2):
    lat1, lon1 = coord1[..., 0], coord1[..., 1]
    lat2, lon2 = coord2[..., 0], coord2[..., 1]
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
    c = 2 * torch.asin(torch.sqrt(a.clamp(0, 1)))
    R = 6371.0
    return R * c