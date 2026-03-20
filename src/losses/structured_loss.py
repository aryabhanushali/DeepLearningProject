# Structured loss for encouraging spatial organization in CNNs.
# how we defined total loss:
#   L_total = L_CE + lambda_smooth * L_smooth + lambda_comp * L_comp
# L_smooth: neighboring channels on the grid should have similar activations
# L_comp:   channels far apart on the grid should not learn the same features

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_neighbor_pairs(grid_h, grid_w, device):
    # collect all 4-connected neighbor pairs (right and down only to avoid duplicates)
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
    # grid_h * grid_w must equal the number of channels in the target layer
    # (512 for ResNet18 layer4, so we use a 16x32 grid)

    def __init__(self, grid_h, grid_w, lambda_smooth=0.01, lambda_comp=0.001):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_channels = grid_h * grid_w
        self.lambda_smooth = lambda_smooth
        self.lambda_comp = lambda_comp

        # these get built the first time forward is called and cached after that
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
        # precompute a (C, C) matrix of normalized grid distances between channels
        # used to weight the competition loss so distant pairs are penalized more
        c = self.num_channels
        rows = torch.arange(c, device=device) // self.grid_w
        cols = torch.arange(c, device=device) % self.grid_w
        dist = ((rows[:, None] - rows[None, :]).float() ** 2 +
                (cols[:, None] - cols[None, :]).float() ** 2).sqrt()
        self._dist_weights = dist / dist.max()
        self._cached_device = device

    def smooth_loss(self, activations):
        # average-pool spatial dims so each channel is a single scalar per image
        a = activations.mean(dim=(2, 3)) if activations.dim() == 4 else activations
        self._ensure_cached(a.device)
        diff = a[:, self._neighbor_i] - a[:, self._neighbor_j]
        return (diff ** 2).mean()

    def competition_loss(self, activations):
        # penalize cosine similarity between channels, scaled by their distance
        # on the grid so far-apart channels are pushed to learn different things
        a = activations.mean(dim=(2, 3)) if activations.dim() == 4 else activations
        self._ensure_cached(a.device)

        a_norm = F.normalize(a.T, dim=1)       # (C, B) -- unit vectors per channel
        sim_matrix = a_norm @ a_norm.T          # (C, C) pairwise cosine similarities

        mask = ~torch.eye(self.num_channels, dtype=torch.bool, device=a.device)
        return (sim_matrix * self._dist_weights)[mask].mean()

    def forward(self, ce_loss, activations):
        l_smooth = self.smooth_loss(activations)
        l_comp = self.competition_loss(activations)
        total = ce_loss + self.lambda_smooth * l_smooth + self.lambda_comp * l_comp
        metrics = {
            "loss/ce": ce_loss.item(),
            "loss/smooth": l_smooth.item(),
            "loss/comp": l_comp.item(),
            "loss/total": total.item(),
        }
        return total, metrics
