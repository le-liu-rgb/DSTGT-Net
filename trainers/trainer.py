import os
import math
import logging
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from trainers.sample import sample
from tools.logging_utils import new_log

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config, savedir=None,
                 device=torch.device("cpu"), aisdls={}, INIT_SEQLEN=0):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.savedir = savedir
        self.device = device
        self.model = model.to(device)
        self.aisdls = aisdls
        self.INIT_SEQLEN = INIT_SEQLEN
        self.tokens = 0

        if hasattr(model, "graph_sage") and hasattr(model, "set_graph_save_info"):
            model.graph_save_dir = os.path.join(savedir, 'graphs')
            if not os.path.exists(model.graph_save_dir):
                os.makedirs(model.graph_save_dir)

        self.csv_file_path = os.path.join(savedir, 'training_loss.csv')
        import csv
        with open(self.csv_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Iteration', 'Loss', 'Learning Rate'])

    def save_checkpoint(self, best_epoch):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logging.info(f"Best epoch: {best_epoch:03d}, saving model to {self.config.ckpt_path}")
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config, aisdls, INIT_SEQLEN = self.model, self.config, self.aisdls, self.INIT_SEQLEN
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split, epoch=0):
            is_train = split == 'Training'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=is_train, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)
            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            d_loss, d_n = 0, 0
            for it, (seqs, masks, seqlens, mmsis, time_starts) in pbar:
                seqs = seqs.to(self.device)
                masks = masks[:, :-1].to(self.device)
                time_starts = time_starts.to(self.device)

                if is_train and hasattr(model, "set_graph_save_info"):
                    model.set_graph_save_info(epoch, it)

                with torch.set_grad_enabled(is_train):
                    logits, loss = model(seqs, masks=masks, with_targets=True, time_stamps=time_starts)
                    loss = loss.mean()
                    losses.append(loss.item())

                d_loss += loss.item() * seqs.shape[0]
                d_n += seqs.shape[0]

                if is_train:
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()
                    if config.lr_decay:
                        self.tokens += (seqs >= 0).sum()
                        if self.tokens < config.warmup_tokens:
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            progress = float(self.tokens - config.warmup_tokens) / float(
                                max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    pbar.set_description(f"epoch {epoch+1} iter {it}: loss {loss.item():.5f}. lr {lr:e}")
                    import csv
                    with open(self.csv_file_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch+1, it+1, loss.item(), lr])

            logging.info(f"{split}, epoch {epoch+1}, loss {d_loss/d_n:.5f}")
            if not is_train:
                return float(np.mean(losses))

        best_loss = float('inf')
        best_epoch = 0
        for epoch in range(config.max_epochs):
            run_epoch('Training', epoch=epoch)
            if self.test_dataset is not None:
                test_loss = run_epoch('Valid', epoch=epoch)
                if test_loss < best_loss:
                    best_loss = test_loss
                    best_epoch = epoch
                    self.save_checkpoint(best_epoch + 1)

        # 保存最终模型
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        save_path = config.ckpt_path.replace("model.pt", f"best_model_epoch_{best_epoch+1}_loss_{best_loss:.4f}.pt")
        torch.save(raw_model.state_dict(), save_path)
        logging.info(f"Last epoch: {epoch:03d}, saving final model to {save_path}")