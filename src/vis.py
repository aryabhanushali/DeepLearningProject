import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def _compute_sim_and_dist(activations, grid_w):
    # returns upper-triangle pairwise cosine similarities and grid distances
    C = activations.shape[1]
    norms = np.linalg.norm(activations, axis=0, keepdims=True)
    act_norm = activations / (norms + 1e-8)
    sim_matrix = act_norm.T @ act_norm

    rows = np.arange(C) // grid_w
    cols = np.arange(C) % grid_w
    dist_matrix = np.sqrt(
        (rows[:, None] - rows[None, :]) ** 2 +
        (cols[:, None] - cols[None, :]) ** 2
    )

    idx = np.triu_indices(C, k=1)
    return sim_matrix[idx], dist_matrix[idx]


def _bin_by_distance(sims, dists, n_bins=20):
    # bin similarities by distance, return (bin centers, bin means)
    bins = np.linspace(0, dists.max(), n_bins + 1)
    centers, means = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (dists >= lo) & (dists < hi)
        if mask.sum() > 0:
            centers.append((lo + hi) / 2)
            means.append(sims[mask].mean())
    return centers, means


def plot_activation_grid(activations, grid_h, grid_w, title="mean channel activations", save_path=None):
    mean_act = activations.mean(axis=0).reshape(grid_h, grid_w)
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(mean_act, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("grid column")
    ax.set_ylabel("grid row")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_similarity_matrix(activations, title="channel cosine similarity", save_path=None):
    norms = np.linalg.norm(activations, axis=0, keepdims=True)
    act_norm = activations / (norms + 1e-8)
    sim = act_norm.T @ act_norm

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(sim, cmap="coolwarm", center=0, ax=ax, square=True,
                xticklabels=False, yticklabels=False)
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_similarity_vs_distance(activations, grid_h, grid_w, label="model", save_path=None):
    # if spatial organization worked, similarity should decrease as distance increases
    sims, dists = _compute_sim_and_dist(activations, grid_w)
    centers, means = _bin_by_distance(sims, dists)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(dists, sims, alpha=0.02, s=1, color="steelblue")
    ax.plot(centers, means, color="red", linewidth=2, label="bin mean")
    ax.set_xlabel("grid distance")
    ax.set_ylabel("cosine similarity")
    ax.set_title(f"similarity vs. distance ({label})")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def compare_models(activations_baseline, activations_structured, grid_h, grid_w, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for ax, acts, name in zip(axes,
                               [activations_baseline, activations_structured],
                               ["Baseline", "Structured"]):
        sims, dists = _compute_sim_and_dist(acts, grid_w)
        centers, means = _bin_by_distance(sims, dists)
        ax.scatter(dists, sims, alpha=0.02, s=1, color="steelblue")
        ax.plot(centers, means, color="red", linewidth=2)
        ax.set_title(name)
        ax.set_xlabel("grid distance")
        ax.set_ylabel("cosine similarity")

    plt.suptitle("similarity vs. grid distance", fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
