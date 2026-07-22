
import os  # 导入操作系统接口模块
import pickle  # 导入序列化/反序列化模块
import torch  # 导入PyTorch深度学习框架


class Config():
    retrain = True
    tb_log = False
    device = torch.device("cuda:0")
    #     device = torch.device("cpu")
    #time = 15

    max_epochs = 20
    batch_size = 32
    n_samples = 16

    init_seqlen = 18
    max_seqlen = 120
    min_seqlen = 36

    dataset_name = "ct_dma"

    if dataset_name == "ct_dma":
    # ==============================

        # When mode == "grad" or "pos_grad", sog and cog are actually dlat and dlon  # 当mode为"grad"或"pos_grad"时，sog和cog实际表示纬度和经度的变化量
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

    # ===========================================================================
    mode = "pos"
    sample_mode = "pos_vicinity"
    top_k = 10
    r_vicinity = 40

    # ===================================================
    blur = True
    blur_learnable = False
    blur_loss_w = 1.0
    blur_n = 2
    if not blur:
        blur_n = 0
        blur_loss_w = 0

    # ===================================================
    datadir = f"./data/{dataset_name}/"
    trainset_name = f"{dataset_name}_train.pkl"
    validset_name = f"{dataset_name}_valid.pkl"
    testset_name = f"{dataset_name}_test.pkl"

    # ===================================================
    n_head = 8
    n_layer = 8
    full_size = lat_size + lon_size + sog_size + cog_size
    n_embd = n_lat_embd + n_lon_embd + n_sog_embd + n_cog_embd

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    # ===================================================
    learning_rate = 0.00095
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1
    lr_decay = True
    warmup_tokens = 512 * 20
    final_tokens = 260e9
    num_workers = 4

    # ========== TimeBridge ==========
    seq_len = init_seqlen
    pred_len = max_seqlen - init_seqlen
    d_model = 512
    n_heads = 8
    d_ff = 2048
    ia_layers = 1
    pd_layers = 1
    ca_layers = 0
    period = 6
    stable_len = 6
    num_p = None
    dropout = 0.1
    attn_dropout = 0.15
    activation = 'gelu'
    revin = True

    alpha = 0.2

    filename = f"{dataset_name}" \
               + f"-{mode}-{sample_mode}-{top_k}-{r_vicinity}" \
               + f"-blur-{blur}-{blur_learnable}-{blur_n}-{blur_loss_w}" \
               + f"-data_size-{lat_size}-{lon_size}-{sog_size}-{cog_size}" \
               + f"-embd_size-{n_lat_embd}-{n_lon_embd}-{n_sog_embd}-{n_cog_embd}" \
               + f"-head-{n_head}-{n_layer}" \
               + f"-bs-{batch_size}" \
               + f"-seqlen-{init_seqlen}-{max_seqlen}" \
               + f"-GraphSAGE-{15}min"

    savedir = "./results/" + filename + "/"

    ckpt_path = os.path.join(savedir, "model.pt")