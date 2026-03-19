"""
Structured regularization loss that encourages spatial organization among
feature channels arranged on a 2-D grid.

  L_total = L_CE + lambda_smooth * L_smooth + lambda_comp * L_comp

  L_smooth  — local smoothness: penalize squared activation differences
              between channels that are adjacent on the grid.
  L_comp    — global competition: penalize cosine similarity between
              channels that are far apart on the grid.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_neighbor_pairs(grid_h: int, grid_w: int, device: torch.device):
    """Return (i, j) index pairs for all 4-connected neighbors on an H x W grid."""
    pairs = []
    for r in range(grid_h):
        for c in range(grid_w):
            idx = r * grid_w + c
            if c + 1 < grid_w:
                pairs.append((idx, r * grid_w + (c + 1)))   # right
            if r + 1 < grid_h:
                pairs.append((idx, (r + 1) * grid_w + c))   # down
    i_idx = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device)
    j_idx = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)
    return i_idx, j_idx


class StructuredLoss(nn.Module):
    """
    Args:
        grid_h, grid_w : dimensions of the virtual channel grid.
                         grid_h * grid_w must equal the number of channels
                         in the hooked activation tensor.
        lambda_smooth  : weight for the local smoothness term.
        lambda_comp    : weight for the global competition term.
    """

    def __init__(
        self,
        grid_h: int,
        grid_w: int,
        lambda_smooth: float = 0.01,
        lambda_comp: float = 0.001,
    ):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_channels = grid_h * grid_w
        self.lambda_smooth = lambda_smooth
        self.lambda_comp = lambda_comp

        self._neighbor_i = None
        self._neighbor_j = None

    def _ensure_neighbors(self, device: torch.device):
        if self._neighbor_i is None or self._neighbor_i.device != device:
            self._neighbor_i, self._neighbor_j = _build_neighbor_pairs(
                self.grid_h, self.grid_w, device
            )

    def smooth_loss(self, activations: torch.Tensor) -> torch.Tensor:
        """
        activations : (B, C, H, W) or (B, C) — spatial dims are avg-pooled away.
        """
        if activations.dim() == 4:
            a = activations.mean(dim=(2, 3))   # (B, C)
        else:
            a = activations                     # (B, C)

        self._ensure_neighbors(a.device)
        diff = a[:, self._neighbor_i] - a[:, self._neighbor_j]   # (B, num_pairs)
        return (diff ** 2).mean()

    def competition_loss(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Penalize cosine similarity between all pairs of channels (not just neighbors).
        activations : (B, C, H, W) or (B, C)
        """
        if activations.dim() == 4:
            a = activations.mean(dim=(2, 3))   # (B, C)
        else:
            a = activations

        # a : (B, C) — treat each channel vector across the batch dimension
        # Normalise along batch dimension to get unit vectors per channel
        a_t = a.T                              # (C, B)
        a_norm = F.normalize(a_t, dim=1)       # (C, B)
        sim_matrix = a_norm @ a_norm.T         # (C, C)

        # Exclude self-similarity (diagonal)
        mask = ~torch.eye(self.num_channels, dtype=torch.bool, device=a.device)
        return sim_matrix[mask].mean()

    def forward(
        self,
        ce_loss: torch.Tensor,
        activations: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
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
