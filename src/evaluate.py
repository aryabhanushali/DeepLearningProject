import argparse

import numpy as np
import torch
import yaml
from sklearn.metrics import confusion_matrix

from data.cifar10 import get_cifar10_loaders, CLASSES
from models.resnet import get_resnet18


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def spatial_organization_score(activations, grid_h, grid_w):
    # measure how much channel similarity correlates with grid proximity
    # if nearby channels are more similar than distant ones, the score is higher
    # we use pearson correlation between cosine similarity and negative grid distance
    C = activations.shape[1]

    norms = np.linalg.norm(activations, axis=0, keepdims=True)
    act_norm = activations / (norms + 1e-8)
    sim_matrix = act_norm.T @ act_norm   # (C, C) cosine similarities

    rows = np.arange(C) // grid_w
    cols = np.arange(C) % grid_w
    dist_matrix = np.sqrt(
        (rows[:, None] - rows[None, :]) ** 2 +
        (cols[:, None] - cols[None, :]) ** 2
    )

    # only look at upper triangle to avoid counting each pair twice
    idx = np.triu_indices(C, k=1)
    sims = sim_matrix[idx]
    dists = dist_matrix[idx]

    return float(np.corrcoef(sims, -dists)[0, 1])


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

    model = get_resnet18(num_classes=cfg["model"]["num_classes"]).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # collect layer4 activations over the full test set
    all_activations = []
    def save_activations(module, input, output):
        all_activations.append(output.mean(dim=(2, 3)).detach().cpu().numpy())

    getattr(model, cfg["loss"]["target_layer"]).register_forward_hook(save_activations)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images.to(device))
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = 100.0 * np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"test accuracy: {acc:.2f}%")

    activations_np = np.concatenate(all_activations, axis=0)
    score = spatial_organization_score(
        activations_np,
        grid_h=cfg["loss"]["grid_h"],
        grid_w=cfg["loss"]["grid_w"],
    )
    print(f"spatial organization score: {score:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    print("\nconfusion matrix:")
    print("            " + "  ".join(f"{c:>10}" for c in CLASSES))
    for i, row in enumerate(cm):
        print(f"{CLASSES[i]:>12}  " + "  ".join(f"{v:>10}" for v in row))


if __name__ == "__main__":
    main()
