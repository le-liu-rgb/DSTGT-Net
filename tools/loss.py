import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryLoss(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss_type = getattr(config, 'loss_type', 'ce')
        self.use_blur = getattr(config, 'blur', False)
        self.blur_loss_w = getattr(config, 'blur_loss_w', 1.0)
        self.blur_n = getattr(config, 'blur_n', 2)
        self.blur_learnable = getattr(config, 'blur_learnable', False)
        self.alpha = getattr(config, 'alpha', 0.2)

        if self.use_blur:

            pass

    def _blur_probs(self, logits, num_classes):

        blur_module = nn.Conv1d(1, 1, 3, padding=1, padding_mode='replicate', groups=1, bias=False)
        if not self.blur_learnable:
            with torch.no_grad():
                blur_module.weight.fill_(1/3)
        else:

            blur_module = blur_module.to(logits.device)

        probs = F.softmax(logits, dim=-1)
        blurred = blur_module(probs.view(-1, 1, num_classes))
        return blurred.view(probs.shape)

    def cross_entropy_loss(self, logits, targets, masks=None, feature_name='all'):

        if logits.dim() == 3:
            B, T, C = logits.shape
            logits_flat = logits.view(-1, C)
            targets_flat = targets.view(-1)
            if masks is not None:
                masks_flat = masks.view(-1)
            else:
                masks_flat = None
        else:
            logits_flat = logits
            targets_flat = targets
            masks_flat = masks

        loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')  # [B*T]

        if self.use_blur and self.blur_n > 0:

            num_classes = logits_flat.shape[-1]
            blurred_probs = self._blur_probs(logits_flat, num_classes)  # [B*T, C]

            blur_loss = F.nll_loss(blurred_probs.log(), targets_flat, reduction='none')
            loss = loss + self.blur_loss_w * blur_loss

        # 应用掩码
        if masks_flat is not None:
            loss = loss * masks_flat

            return loss
        else:
            return loss

    def mse_loss(self, pred, target, masks=None):

        loss = (pred - target).pow(2)
        if masks is not None:
            loss = loss * masks.unsqueeze(-1)
            seq_mask = masks.sum(dim=1, keepdim=True)  # [B, 1]
            loss = loss.sum(dim=1) / (seq_mask + 1e-8)  # [B, 4]
            loss = loss.mean()
        else:
            loss = loss.mean()
        return loss

    def time_freq_mae(self, pred, target, alpha=None):

        if alpha is None:
            alpha = self.alpha
        t_loss = (pred - target).abs().mean()
        fft_pred = torch.fft.rfft(pred, dim=1)
        fft_target = torch.fft.rfft(target, dim=1)
        f_loss = (fft_pred - fft_target).abs().mean()
        return (1 - alpha) * t_loss + alpha * f_loss

    def forward(self, logits, targets, masks=None, loss_type=None, **kwargs):

        if loss_type is None:
            loss_type = self.loss_type

        if loss_type == 'ce':

            if isinstance(logits, (list, tuple)):
                losses = []
                for l, t in zip(logits, targets):
                    loss = self.cross_entropy_loss(l, t, masks)
                    losses.append(loss)
                return losses
            else:
                return self.cross_entropy_loss(logits, targets, masks)
        elif loss_type == 'mse':
            return self.mse_loss(logits, targets, masks)
        elif loss_type == 'time_freq':
            return self.time_freq_mae(logits, targets, **kwargs)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def compute_loss_tuple(self, lat_logits, lon_logits, sog_logits, cog_logits,
                           lat_targets, lon_targets, sog_targets, cog_targets,
                           masks=None):

        lat_loss = self.cross_entropy_loss(lat_logits, lat_targets, masks)
        lon_loss = self.cross_entropy_loss(lon_logits, lon_targets, masks)
        sog_loss = self.cross_entropy_loss(sog_logits, sog_targets, masks)
        cog_loss = self.cross_entropy_loss(cog_logits, cog_targets, masks)
        return lat_loss, lon_loss, sog_loss, cog_loss