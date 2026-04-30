# adaptive spatial loss -- same smooth loss as SmoothOnlyLoss but the grid
# topology gets updated every few epochs based on which channels are actually
# similar to each other (inspired by self-organizing maps)

import numpy as np
import torch
import torch.nn as nn

from .structured_loss import _build_neighbor_pairs


def _spectral_assignment(activations_np, grid_h, grid_w):
    # assign channels to grid positions so that similar channels end up near each other
    # uses PCA on the channel similarity matrix to get 2D coordinates, then sorts onto grid
    C = grid_h * grid_w
    a = activations_np - activations_np.mean(0, keepdims=True)
    norms = np.linalg.norm(a, axis=0, keepdims=True) + 1e-8
    a_norm = a / norms
    sim = (a_norm.T @ a_norm) / max(len(a), 1)

    # top 2 eigenvectors give 2D coordinates for each channel
    _, eigenvectors = np.linalg.eigh(sim)
    coords = eigenvectors[:, -2:]  # (C, 2)

    # sort by first eigenvector to assign rows, second to assign columns within each row
    row_order = np.argsort(coords[:, 0])
    assignment = np.empty(C, dtype=np.int64)
    for row in range(grid_h):
        start, end = row * grid_w, (row + 1) * grid_w
        row_channels = row_order[start:end]
        col_order = row_channels[np.argsort(coords[row_channels, 1])]
        for col, ch in enumerate(col_order):
            assignment[ch] = row * grid_w + col

    return assignment


def _assignment_to_neighbor_pairs(assignment, grid_h, grid_w, device):
    # convert channel->grid_position assignment into 4-connected neighbor pairs
    C = grid_h * grid_w
    inverse = np.empty(C, dtype=np.int64)
    inverse[assignment] = np.arange(C)

    pairs = []
    for r in range(grid_h):
        for c in range(grid_w):
            ch = inverse[r * grid_w + c]
            if c + 1 < grid_w:
                pairs.append((ch, inverse[r * grid_w + (c + 1)]))
            if r + 1 < grid_h:
                pairs.append((ch, inverse[(r + 1) * grid_w + c]))

    i_idx = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device)
    j_idx = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)
    return i_idx, j_idx


class AdaptiveSpatialLoss(nn.Module):
    # smooth loss where the grid topology gets reassigned every reassign_every epochs

    def __init__(self, grid_h, grid_w, lambda_smooth=0.1, reassign_every=2):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_channels = grid_h * grid_w
        self.lambda_smooth = lambda_smooth
        self.reassign_every = reassign_every
        self._neighbor_i = None
        self._neighbor_j = None
        self._cached_device = None
        self._assignment = None
        self._activation_buffer = []

    def _ensure_cached(self, device):
        if self._cached_device == device and self._neighbor_i is not None:
            return
        if self._assignment is None:
            self._neighbor_i, self._neighbor_j = _build_neighbor_pairs(
                self.grid_h, self.grid_w, device
            )
        else:
            self._neighbor_i, self._neighbor_j = _assignment_to_neighbor_pairs(
                self._assignment, self.grid_h, self.grid_w, device
            )
        self._cached_device = device

    def maybe_reassign(self, epoch):
        # call at the end of each epoch to flush the buffer and maybe update the grid
        if not self._activation_buffer:
            return
        all_acts = np.concatenate(self._activation_buffer, axis=0)
        self._activation_buffer.clear()
        if epoch % self.reassign_every == 0:
            self._assignment = _spectral_assignment(all_acts, self.grid_h, self.grid_w)
            self._cached_device = None  # force neighbor pair recompute
            print(f"  [adaptive] grid reassigned at epoch {epoch}")

    def forward(self, ce_loss, activations, **kwargs):
        a = activations.mean(dim=(2, 3)) if activations.dim() == 4 else activations
        self._activation_buffer.append(a.detach().cpu().numpy())
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
