
import os
import torch

class Config():
    retrain = True
    tb_log = False
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    max_epochs = 20
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

        n_lat_embd = 256
        n_lon_embd = 256
        n_sog_embd = 128
        n_cog_embd = 128

        lat_min = 55.5
        lat_max = 58.0
        lon_min = 10.3
        lon_max = 13

    mode = "pos"
    sample_mode = "pos_vicinity"
    top_k = 10
    r_vicinity = 40

    blur = True
    blur_learnable = False
    blur_loss_w = 1.0
    blur_n = 2
    if not blur:
        blur_n = 0
        blur_loss_w = 0

    datadir = f"./data/{dataset_name}/"
    trainset_name = f"{dataset_name}_train.pkl"
    validset_name = f"{dataset_name}_valid.pkl"
    testset_name = f"{dataset_name}_test.pkl"

    n_head = 8
    n_layer = 8
    full_size = lat_size + lon_size + sog_size + cog_size
    n_embd = n_lat_embd + n_lon_embd + n_sog_embd + n_cog_embd

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1
    lr_decay = True
    warmup_tokens = 512 * 20
    final_tokens = 260e9
    num_workers = 4

    filename = f"{dataset_name}" \
               + f"-{mode}-{sample_mode}-{top_k}-{r_vicinity}" \
               + f"-blur-{blur}-{blur_learnable}-{blur_n}-{blur_loss_w}" \
               + f"-data_size-{lat_size}-{lon_size}-{sog_size}-{cog_size}" \
               + f"-embd_size-{n_lat_embd}-{n_lon_embd}-{n_sog_embd}-{n_cog_embd}" \
               + f"-head-{n_head}-{n_layer}" \
               + f"-bs-{batch_size}" \
               + f"-seqlen-{init_seqlen}-{max_seqlen}" \
               + f"-GraphSAGE-{getattr(self, 'time', 15)}min"

    savedir = "./results/" + filename + "/"
    ckpt_path = os.path.join(savedir, "model.pt")


    graph_save_dir = os.path.join(savedir, "graphs")