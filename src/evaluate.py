"""
Evaluation script: reports test accuracy and spatial organization metrics.

Usage:
    python src/evaluate.py --config configs/default.yaml \
        --checkpoint experiments/structured/best_model.pth
"""

import argparse
import yaml
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

from data.cifar10 import get_cifar10_loaders
from models.resnet import get_resnet18


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def spatial_organization_score(activations: np.ndarray, grid_h: int, grid_w: int) -> float:
    """
    Measures how well channel similarity correlates with grid proximity.

    Returns the Pearson correlation between pairwise cosine similarity and
    negative grid distance (higher = more organized).

    activations : (N, C) array of per-sample channel activations.
    """
    # Mean activations per channel across samples
    mean_act = activations.mean(axis=0)   # (C,)
    C = mean_act.shape[0]

    # Pairwise cosine similarity
    normed = mean_act / (np.linalg.norm(mean_act) + 1e-8)
    # Per-channel norms
    norms = np.linalg.norm(activations, axis=0, keepdims=True)   # (1, C)
    act_norm = activations / (norms + 1e-8)
    sim_matrix = (act_norm.T @ act_norm) / len(activations)       # (C, C)

    # Grid distance matrix
    rows = np.arange(C) // grid_w
    cols = np.arange(C) % grid_w
    dist_matrix = np.sqrt(
        (rows[:, None] - rows[None, :]) ** 2 +
        (cols[:, None] - cols[None, :]) ** 2
    )

    # Upper triangle (exclude diagonal)
    idx = np.triu_indices(C, k=1)
    sims = sim_matrix[idx]
    dists = dist_matrix[idx]

    corr = np.corrcoef(sims, -dists)[0, 1]
    return float(corr)


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, test_loader = get_cifar10_loaders(
        data_dir=cfg["data"]["data_dir"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        augment=False,
    )

    model = get_resnet18(
        num_classes=cfg["model"]["num_classes"],
        pretrained=False,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # Hook activations
    all_activations = []
    def hook_fn(module, input, output):
        feat = output.mean(dim=(2, 3)).detach().cpu().numpy()
        all_activations.append(feat)

    target_layer = cfg["loss"]["target_layer"]
    getattr(model, target_layer).register_forward_hook(hook_fn)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = 100.0 * np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"Test accuracy: {acc:.2f}%")

    activations_np = np.concatenate(all_activations, axis=0)
    score = spatial_organization_score(
        activations_np,
        grid_h=cfg["loss"]["grid_h"],
        grid_w=cfg["loss"]["grid_w"],
    )
    print(f"Spatial organization score (similarity-distance correlation): {score:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion matrix:")
    print("         " + " ".join(f"{c:>10}" for c in CIFAR10_CLASSES))
    for i, row in enumerate(cm):
        print(f"{CIFAR10_CLASSES[i]:>10} " + " ".join(f"{v:>10}" for v in row))


if __name__ == "__main__":
    main()
