import os, sys, csv
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data.cifar10 import get_cifar10_loaders
from models.resnet import get_resnet18
from vis import _compute_sim_and_dist, _bin_by_distance

OUT_DIR = "experiments/figures"
os.makedirs(OUT_DIR, exist_ok=True)

ALL_EXPERIMENTS = [
    ("baseline",             "configs/default.yaml",              "Baseline (CE only)"),
    ("structured",           "configs/default.yaml",              "Original (L4 smooth+comp)"),
    ("layer3_smooth_only",   "configs/layer3_smooth_only.yaml",   "L3 smooth only"),
    ("high_lambda",          "configs/high_lambda.yaml",          "L3 high-λ (0.5)"),
    ("adaptive",             "configs/adaptive.yaml",             "L3 adaptive grid"),
    ("layer3_medium_lambda", "configs/layer3_medium_lambda.yaml", "L3 medium-λ (0.1)"),
]


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
    return np.concatenate(all_acts, axis=0), cfg["loss"]["grid_h"], cfg["loss"]["grid_w"]


def spatial_org_score(acts, grid_h, grid_w):
    C = acts.shape[1]
    norms = np.linalg.norm(acts, axis=0, keepdims=True)
    acts_n = acts / (norms + 1e-8)
    sim = acts_n.T @ acts_n
    rows = np.arange(C) // grid_w
    cols = np.arange(C) % grid_w
    dist = np.sqrt((rows[:, None] - rows[None, :])**2 + (cols[:, None] - cols[None, :])**2)
    idx = np.triu_indices(C, k=1)
    return float(np.corrcoef(sim[idx], -dist[idx])[0, 1])


# figure 1: similarity vs distance for all experiments
def fig_similarity_vs_distance(cache):
    n = len(ALL_EXPERIMENTS)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)

    for ax, (exp, _, title) in zip(axes, ALL_EXPERIMENTS):
        if exp not in cache:
            ax.set_title(f"{title}\n(missing)")
            continue
        acts, grid_h, grid_w = cache[exp]
        score = spatial_org_score(acts, grid_h, grid_w)
        sims, dists = _compute_sim_and_dist(acts, grid_w)
        centers, means = _bin_by_distance(sims, dists)
        ax.scatter(dists, sims, alpha=0.015, s=1, color="steelblue", rasterized=True)
        ax.plot(centers, means, color="red", linewidth=2, label="bin mean")
        ax.set_title(f"{title}\nscore={score:.4f}", fontsize=9)
        ax.set_xlabel("grid distance")
        ax.set_ylabel("cosine similarity")
        ax.legend(fontsize=7)

    fig.suptitle("Channel Similarity vs. Grid Distance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig1_similarity_vs_distance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")


# figure 2: mean activation heatmap on the virtual grid
def fig_activation_grids(cache):
    n = sum(1 for exp, _, _ in ALL_EXPERIMENTS if exp in cache)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    ax_idx = 0
    for exp, _, title in ALL_EXPERIMENTS:
        if exp not in cache:
            continue
        acts, grid_h, grid_w = cache[exp]
        mean_act = acts.mean(axis=0).reshape(grid_h, grid_w)
        im = axes[ax_idx].imshow(mean_act, cmap="viridis", aspect="auto")
        plt.colorbar(im, ax=axes[ax_idx])
        axes[ax_idx].set_title(title, fontsize=10)
        axes[ax_idx].set_xlabel("grid column")
        axes[ax_idx].set_ylabel("grid row")
        ax_idx += 1

    fig.suptitle("Mean Channel Activation on Virtual Grid", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig2_activation_grids.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")


# figure 3: scatter of accuracy vs spatial score for each experiment
def fig_accuracy_vs_score(cache):
    accs, scores, labels, colors = [], [], [], []
    color_map = {
        "baseline": "#d62728",
        "structured": "#1f77b4",
        "layer3_smooth_only": "#ff7f0e",
        "high_lambda": "#9467bd",
        "adaptive": "#2ca02c",
        "layer3_medium_lambda": "#8c564b",
    }

    for exp, cfg_path, label in ALL_EXPERIMENTS:
        log = f"experiments/{exp}/log.csv"
        if not os.path.exists(log):
            continue
        rows = [r for r in csv.DictReader(open(log)) if r["test_acc"]]
        if not rows:
            continue
        best_acc = max(float(r["test_acc"]) for r in rows)

        if exp in cache:
            acts, grid_h, grid_w = cache[exp]
            score = spatial_org_score(acts, grid_h, grid_w)
        else:
            score = 0.0

        accs.append(best_acc)
        scores.append(score)
        labels.append(label)
        colors.append(color_map.get(exp, "#7f7f7f"))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(scores, accs, c=colors, s=120, zorder=3)
    for x, y, lbl in zip(scores, accs, labels):
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)

    if "Baseline (CE only)" in labels:
        baseline_acc = accs[labels.index("Baseline (CE only)")]
        ax.axhline(y=baseline_acc, color="#d62728", linestyle="--",
                   linewidth=1, alpha=0.5, label="baseline accuracy")

    ax.set_xlabel("Spatial Organization Score", fontsize=11)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_title("Accuracy vs. Spatial Organization", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig3_accuracy_vs_score.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, test_loader = get_cifar10_loaders(
        data_dir="./data", batch_size=128,
        num_workers=0, augment=False, subset_frac=0.2,
    )

    print("collecting activations...")
    cache = {}
    for exp, cfg_path, _ in ALL_EXPERIMENTS:
        ckpt = f"experiments/{exp}/best_model.pth"
        if os.path.exists(ckpt):
            print(f"  {exp}...")
            cache[exp] = collect_activations(ckpt, cfg_path, device, test_loader)
        else:
            print(f"  {exp} -- missing, skipping")

    print("generating figures...")
    fig_similarity_vs_distance(cache)
    fig_activation_grids(cache)
    fig_accuracy_vs_score(cache)
    print(f"\nall figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
