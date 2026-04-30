

import os, sys
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data.cifar10 import get_cifar10_loaders
from models.resnet import get_resnet18
from vis import plot_activation_grid, _compute_sim_and_dist, _bin_by_distance

OUT_DIR = "experiments/figures"
os.makedirs(OUT_DIR, exist_ok=True)


#helper function to load a model, run the test set through it, and collect the mean channel activations for all images
def collect_activations(checkpoint, config_path, device, loader):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model = get_resnet18(num_classes=10).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    all_acts = []
    def hook(_m, _i, out):
        all_acts.append(out.mean(dim=(2, 3)).detach().cpu().numpy())
    h = getattr(model, cfg["loss"]["target_layer"]).register_forward_hook(hook)

    with torch.no_grad():
        for images, _ in loader:
            model(images.to(device))
    h.remove()

    acts = np.concatenate(all_acts, axis=0)  # (N, C)
    return acts, cfg["loss"]["grid_h"], cfg["loss"]["grid_w"]


# Figure 1: Similarity vs Distance — baseline, original, best (3 subplots)

def fig_similarity_vs_distance(cache):
    experiments = [
        ("baseline",           "Baseline (CE only)\nscore ≈ 0.001"),
        ("structured",         "Original: L4 smooth+comp\nscore ≈ 0.004"),
        ("layer3_smooth_only", "Best: L3 smooth only\nscore ≈ 0.056"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)

    for ax, (exp, title) in zip(axes, experiments):
        if exp not in cache:
            ax.set_title(f"{title}\n(missing)")
            continue

        acts, grid_h, grid_w = cache[exp]
        sims, dists = _compute_sim_and_dist(acts, grid_w)
        centers, means = _bin_by_distance(sims, dists)

        ax.scatter(dists, sims, alpha=0.015, s=1, color="steelblue", rasterized=True)
        ax.plot(centers, means, color="red", linewidth=2, label="bin mean")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("grid distance")
        ax.set_ylabel("cosine similarity")
        ax.legend(fontsize=8)

    fig.suptitle("Channel Similarity vs. Grid Distance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig1_similarity_vs_distance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")


# Figure 2: Activation grid heatmaps: base vs best

def fig_activation_grids(cache):
    experiments = [
        ("baseline",           "Baseline (CE only)"),
        ("layer3_smooth_only", "Best: L3 smooth only"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, (exp, title) in zip(axes, experiments):
        if exp not in cache:
            ax.set_title(f"{title}\n(missing)")
            continue

        acts, grid_h, grid_w = cache[exp]
        mean_act = acts.mean(axis=0).reshape(grid_h, grid_w)

        im = ax.imshow(mean_act, cmap="viridis", aspect="auto")
        plt.colorbar(im, ax=ax)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("grid column")
        ax.set_ylabel("grid row")

    fig.suptitle("Mean Channel Activation on Virtual Grid", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig2_activation_grids.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")


# Figure 3: Accuracy vs spatial score scatter  for all experiments


def fig_accuracy_vs_score():
    import csv

    experiments = [
        ("baseline",           "configs/default.yaml",           "Baseline",          0.0010),
        ("structured",         "configs/default.yaml",           "L4 smooth+comp",    0.0040),
        ("smooth_only",        "configs/smooth_only.yaml",       "L4 smooth only",   -0.0144),
        ("layer3",             "configs/layer3.yaml",            "L3 smooth+comp",    0.0081),
        ("curriculum",         "configs/curriculum.yaml",        "L4 curriculum",     0.0190),
        ("class_conditional",  "configs/class_conditional.yaml", "L4 class-cond.",    0.0115),
        ("layer3_smooth_only", "configs/layer3_smooth_only.yaml","L3 smooth only",    0.0558),
    ]

    accs, scores, labels = [], [], []
    for exp, _, label, score in experiments:
        log = f"experiments/{exp}/log.csv"
        if not os.path.exists(log):
            continue
        rows = [r for r in csv.DictReader(open(log)) if r["test_acc"]]
        best = max(float(r["test_acc"]) for r in rows)
        accs.append(best)
        scores.append(score)
        labels.append(label)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#d62728" if "Baseline" in l else
              "#ff7f0e" if "L3 smooth only" in l else "#1f77b4"
              for l in labels]
    ax.scatter(scores, accs, c=colors, s=100, zorder=3)

    for x, y, lbl in zip(scores, accs, labels):
        ax.annotate(lbl, (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)

    ax.axhline(y=accs[labels.index("Baseline")], color="#d62728",
               linestyle="--", linewidth=1, alpha=0.5, label="baseline accuracy")
    ax.set_xlabel("Spatial Organization Score", fontsize=11)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_title("Accuracy vs. Spatial Organization — All Variants", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    path = os.path.join(OUT_DIR, "fig3_accuracy_vs_score.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")




def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2000 samples is enough for stable similarity statistics
    _, test_loader = get_cifar10_loaders(
        data_dir="./data", batch_size=128,
        num_workers=0, augment=False, subset_frac=0.2,
    )

    # collect activations once per model so nothing is run twice
    print("collecting activations...")
    cache = {}
    for exp, cfg_path in [
        ("baseline",           "configs/default.yaml"),
        ("structured",         "configs/default.yaml"),
        ("layer3_smooth_only", "configs/layer3_smooth_only.yaml"),
    ]:
        ckpt = f"experiments/{exp}/best_model.pth"
        if os.path.exists(ckpt):
            print(f"  {exp}...")
            cache[exp] = collect_activations(ckpt, cfg_path, device, test_loader)

    print("generating figures...")
    fig_similarity_vs_distance(cache)
    fig_activation_grids(cache)
    fig_accuracy_vs_score()
    print(f"\nall figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
