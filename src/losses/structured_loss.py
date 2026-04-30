# structured loss for spatial organization in CNNs
# total loss = CE + lambda_smooth * L_smooth + lambda_comp * L_comp
# L_smooth: neighboring channels on the grid should activate similarly
# L_comp: channels far apart on the grid should learn different things

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_neighbor_pairs(grid_h, grid_w, device):
    # get all 4-connected neighbor pairs (right and down only, no duplicates)
    pairs = []
    for r in range(grid_h):
        for c in range(grid_w):
            idx = r * grid_w + c
            if c + 1 < grid_w:
                pairs.append((idx, r * grid_w + (c + 1)))
            if r + 1 < grid_h:
                pairs.append((idx, (r + 1) * grid_w + c))
    i_idx = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device)
    j_idx = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)
    return i_idx, j_idx


class StructuredLoss(nn.Module):
    # grid_h * grid_w should equal the number of channels in the target layer
    # for ResNet18 layer4 (512 channels) we use a 16x32 grid

    def __init__(self, grid_h, grid_w, lambda_smooth=0.01, lambda_comp=0.001):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_channels = grid_h * grid_w
        self.lambda_smooth = lambda_smooth
        self.lambda_comp = lambda_comp
        self._neighbor_i = None
        self._neighbor_j = None
        self._dist_weights = None
        self._cached_device = None

    def _ensure_cached(self, device):
        if self._cached_device == device:
            return
        self._neighbor_i, self._neighbor_j = _build_neighbor_pairs(
            self.grid_h, self.grid_w, device
        )
        # precompute normalized grid distances between all channel pairs
        c = self.num_channels
        rows = torch.arange(c, device=device) // self.grid_w
        cols = torch.arange(c, device=device) % self.grid_w
        dist = ((rows[:, None] - rows[None, :]).float() ** 2 +
                (cols[:, None] - cols[None, :]).float() ** 2).sqrt()
        self._dist_weights = dist / dist.max()
        self._cached_device = device

    def smooth_loss(self, activations):
        a = activations.mean(dim=(2, 3)) if activations.dim() == 4 else activations
        self._ensure_cached(a.device)
        diff = a[:, self._neighbor_i] - a[:, self._neighbor_j]
        return (diff ** 2).mean()

    def competition_loss(self, activations):
        a = activations.mean(dim=(2, 3)) if activations.dim() == 4 else activations
        self._ensure_cached(a.device)
        # use eps=1e-4 to avoid exploding gradients when channels have near-zero activation
        a_norm = F.normalize(a.T, dim=1, eps=1e-4)
        sim_matrix = a_norm @ a_norm.T
        mask = ~torch.eye(self.num_channels, dtype=torch.bool, device=a.device)
        return (sim_matrix * self._dist_weights)[mask].mean()

    def forward(self, ce_loss, activations, **kwargs):
        l_smooth = self.smooth_loss(activations)
        l_comp = self.competition_loss(activations)
        total = ce_loss + self.lambda_smooth * l_smooth + self.lambda_comp * l_comp
        return total, {
            "loss/ce": ce_loss.item(),
            "loss/smooth": l_smooth.item(),
            "loss/comp": l_comp.item(),
            "loss/total": total.item(),
        }
