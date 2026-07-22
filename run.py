import os
import sys
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader


from config.config_DSTGT import Config as Config_STGT
from config.config_TrAISformer import Config as Config_TrAISformer
from config.config_TimeBridgeais import Config as Config_TimeBridge
from config.config_LSTM_Seq2seq import Config as Config_LSTM
from config.config_lstm_seq2seq_att import Config as Config_LSTM_Att
from config.config_conv_seq2seq import Config as Config_Conv
from config.config_transformer import Config as Config_Transformer
from config.config_geotracknet import Config as Config_GeoTrackNet

from model.DSTGT import DSTGT
from model.baselines.LSTM_Seq2seq import LSTMSeq2Seq
from model.baselines.LSTM_Seq2seq_att import LSTMSeq2SeqAtt
from model.baselines.Conv_Seq2seq import ConvSeq2Seq
from model.baselines.Transformer import TransformerModel
from model.baselines.GeoTrackNet import GeoTrackNet

from trainers.dataset import AISDataset, AISDataset_grad
from trainers.trainer import Trainer

from tools.random_seed import set_seed
from tools.logging_utils import new_log
from tools.evaluation import evaluate_model

def main():
    cf = Config()
    set_seed(42)
    device = cf.device
    init_seqlen = cf.init_seqlen

    if not os.path.isdir(cf.savedir):
        os.makedirs(cf.savedir)
        print(f"Create directory: {cf.savedir}")
    new_log(cf.savedir, "log")

    moving_threshold = 0.05
    l_pkl_filenames = [cf.trainset_name, cf.validset_name, cf.testset_name]
    Data, aisdatasets, aisdls = {}, {}, {}
    for phase, filename in zip(("train", "valid", "test"), l_pkl_filenames):
        datapath = os.path.join(cf.datadir, filename)
        print(f"Loading {datapath}...")
        with open(datapath, "rb") as f:
            raw = pickle.load(f)
        for V in raw:
            try:
                moving_idx = np.where(V["traj"][:, 2] > moving_threshold)[0][0]
            except:
                moving_idx = len(V["traj"]) - 1
            V["traj"] = V["traj"][moving_idx:, :]
        Data[phase] = [x for x in raw if not np.isnan(x["traj"]).any() and len(x["traj"]) > cf.min_seqlen]
        print(f"{phase}: {len(raw)} -> {len(Data[phase])}")

        if cf.mode in ("pos_grad", "grad"):
            aisdatasets[phase] = AISDataset_grad(Data[phase], max_seqlen=cf.max_seqlen+1, device=device)
        else:
            aisdatasets[phase] = AISDataset(Data[phase], max_seqlen=cf.max_seqlen+1, device=device)
        shuffle = phase != "test"
        aisdls[phase] = DataLoader(aisdatasets[phase], batch_size=cf.batch_size, shuffle=shuffle)

    model = STGTNet(cf)
    model.count_parameters()

    trainer = Trainer(model, aisdatasets["train"], aisdatasets["valid"], cf,
                      savedir=cf.savedir, device=device, aisdls=aisdls, INIT_SEQLEN=init_seqlen)

    if cf.retrain:
        trainer.train()

    model.load_state_dict(torch.load(cf.ckpt_path))
    test_loader = aisdls["test"]
    mae, mre = evaluate_model(model, test_loader, cf, device, init_seqlen, n_samples=cf.n_samples)
    print(f"MAE: {mae:.4f} km")
    print(f"MRE: {mre*100:.4f} %")

if __name__ == "__main__":
    main()