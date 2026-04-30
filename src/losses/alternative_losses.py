# Baseline structured loss hurts accuracy due to unstable gradients from normalizing near-zero activations early on.
# Applying constraints to layer4 is harmful since it’s the most task-specific and needs feature specialization.
# Competition loss conflicts with CE: it increases while being minimized and pulls features away from optimal solutions.
# The imposed channel grid is arbitrary, so smoothing adds noise instead of meaningful structure.
# Prior work succeeds by building structure into the model or separating it from classification features.
# SmoothOnlyLoss tests if removing competition reduces harm while still organizing features.
# ClassConditionalSpatialLoss applies structure to class prototypes instead of individual samples.
# CurriculumStructuredLoss ramps λ slowly so the model learns CE first, then spatial structure.
import torch
import torch.nn as nn
import torch.nn.functional as F

from .structured_loss import StructuredLoss, _build_neighbor_pairs


# Smooth-only loss


class SmoothOnlyLoss(nn.Module):


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

    def forward(self, ce_loss, activations, labels=None):
        a = activations.mean(dim=(2, 3)) if activations.dim() == 4 else activations
        self._ensure_cached(a.device)
        diff = a[:, self._neighbor_i] - a[:, self._neighbor_j]
        l_smooth = (diff ** 2).mean()
        total = ce_loss + self.lambda_smooth * l_smooth
        metrics = {
            "loss/ce": ce_loss.item(),
            "loss/smooth": l_smooth.item(),
            "loss/comp": 0.0,
            "loss/total": total.item(),
        }
        return total, metrics


# Class-conditional spatial loss


class ClassConditionalSpatialLoss(nn.Module):


    def __init__(self, grid_h, grid_w, lambda_smooth=0.01, num_classes=10):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_channels = grid_h * grid_w
        self.lambda_smooth = lambda_smooth
        self.num_classes = num_classes
        self._neighbor_i = self._neighbor_j = None
        self._cached_device = None

    def _ensure_cached(self, device):
        if self._cached_device == device:
            return
        self._neighbor_i, self._neighbor_j = _build_neighbor_pairs(
            self.grid_h, self.grid_w, device
        )
        self._cached_device = device

    def forward(self, ce_loss, activations, labels=None):
        if labels is None:
            raise ValueError("ClassConditionalSpatialLoss requires labels")

        a = activations.mean(dim=(2, 3)) if activations.dim() == 4 else activations
        self._ensure_cached(a.device)

        # Build class-conditional channel prototypes (num_present_classes, C)
        prototypes = []
        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() > 0:
                prototypes.append(a[mask].mean(dim=0))

        if not prototypes:
            return ce_loss, {"loss/ce": ce_loss.item(), "loss/smooth": 0.0,
                             "loss/comp": 0.0, "loss/total": ce_loss.item()}

        proto = torch.stack(prototypes, dim=0)          # (K, C)
        diff = proto[:, self._neighbor_i] - proto[:, self._neighbor_j]
        l_smooth = (diff ** 2).mean()

        total = ce_loss + self.lambda_smooth * l_smooth
        metrics = {
            "loss/ce": ce_loss.item(),
            "loss/smooth": l_smooth.item(),
            "loss/comp": 0.0,
            "loss/total": total.item(),
        }
        return total, metrics


# Curriculum wrapper


class CurriculumStructuredLoss(nn.Module):


    def __init__(self, grid_h, grid_w, lambda_smooth=0.01, lambda_comp=0.001,
                 warmup_epochs=5):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self._base_lambda_smooth = lambda_smooth
        self._base_lambda_comp = lambda_comp
        self.inner = StructuredLoss(
            grid_h=grid_h, grid_w=grid_w,
            lambda_smooth=0.0, lambda_comp=0.0,  # set dynamically
        )
        self.current_scale = 0.0

    def set_epoch(self, epoch):
        """Update λ scale based on current epoch (call before each epoch)."""
        if self.warmup_epochs <= 0:
            self.current_scale = 1.0
        else:
            self.current_scale = min((epoch - 1) / self.warmup_epochs, 1.0)
        self.inner.lambda_smooth = self._base_lambda_smooth * self.current_scale
        self.inner.lambda_comp = self._base_lambda_comp * self.current_scale

    def forward(self, ce_loss, activations, labels=None):
        return self.inner(ce_loss, activations)
