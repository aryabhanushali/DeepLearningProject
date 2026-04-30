import torch.nn as nn

from .structured_loss import _build_neighbor_pairs


class SmoothOnlyLoss(nn.Module):
    # spatial smoothness loss with no competition term
    # simpler than StructuredLoss -- just penalizes neighboring channels being different

    def __init__(self, grid_h, grid_w, lambda_smooth=0.01):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.lambda_smooth = lambda_smooth
        self._neighbor_i = self._neighbor_j = None
        self._cached_device = None

    def _ensure_cached(self, device):
        if self._cached_device == device:
            return
        self._neighbor_i, self._neighbor_j = _build_neighbor_pairs(
            self.grid_h, self.grid_w, device
        )
        self._cached_device = device

    def forward(self, ce_loss, activations, **kwargs):
        a = activations.mean(dim=(2, 3)) if activations.dim() == 4 else activations
        self._ensure_cached(a.device)
        diff = a[:, self._neighbor_i] - a[:, self._neighbor_j]
        l_smooth = (diff ** 2).mean()
        total = ce_loss + self.lambda_smooth * l_smooth
        return total, {
            "loss/ce": ce_loss.item(),
            "loss/smooth": l_smooth.item(),
            "loss/comp": 0.0,
            "loss/total": total.item(),
        }
