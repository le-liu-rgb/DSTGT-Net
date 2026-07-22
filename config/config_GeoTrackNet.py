import os
import torch

class Config():
    retrain = True
    tb_log = False
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    max_epochs = 50
    batch_size = 32
    n_samples = 16

    init_seqlen = 18
    max_seqlen = 120
    min_seqlen = 36

    dataset_name = "ct_dma"

    if dataset_name == "ct_dma":
        lat_size = 250
        lon_size = 270
        sog_size = 30
        cog_size = 72
        lat_min = 55.5
        lat_max = 58.0
        lon_min = 10.3
        lon_max = 13

    geotracknet_latent_size = 32
    geotracknet_rnn_hidden = 64
    geotracknet_feat_size = 32

    learning_rate = 1e-3
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1
    lr_decay = True
    warmup_tokens = 512 * 20
    final_tokens = 260e9
    num_workers = 4

    filename = f"{dataset_name}_geotracknet" \
               + f"-bs-{batch_size}" \
               + f"-seqlen-{init_seqlen}-{max_seqlen}" \
               + f"-latent-{geotracknet_latent_size}-rnn-{geotracknet_rnn_hidden}"
    savedir = "./results/" + filename + "/"
    ckpt_path = os.path.join(savedir, "model.pt")