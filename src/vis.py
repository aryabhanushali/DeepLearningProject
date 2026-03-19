"""
Visualization utilities for analyzing spatial organization of channel activations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch


def plot_activation_grid(activations: np.ndarray, grid_h: int, grid_w: int,
                         title: str = "Mean channel activations",
                         save_path: str | None = None):
    """
    Plot mean per-channel activation as a heatmap on the virtual grid.

    activations : (N, C) array
    """
    mean_act = activations.mean(axis=0).reshape(grid_h, grid_w)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mean_act, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Grid column")
    ax.set_ylabel("Grid row")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_similarity_matrix(activations: np.ndarray,
                           title: str = "Channel cosine similarity",
                           save_path: str | None = None):
    """
    Plot pairwise cosine similarity between channels.

    activations : (N, C)
    """
    norms = np.linalg.norm(activations, axis=0, keepdims=True)
    act_norm = activations / (norms + 1e-8)
    sim = (act_norm.T @ act_norm) / len(activations)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(sim, cmap="coolwarm", center=0, ax=ax, square=True,
                xticklabels=False, yticklabels=False)
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_similarity_vs_distance(activations: np.ndarray, grid_h: int, grid_w: int,
                                label: str = "model",
                                save_path: str | None = None):
    """
    Scatter / bin plot of pairwise cosine similarity vs. grid distance.
    A spatially organized model should show decreasing similarity with distance.

    activations : (N, C)
    """
    C = activations.shape[1]
    norms = np.linalg.norm(activations, axis=0, keepdims=True)
    act_norm = activations / (norms + 1e-8)
    sim = (act_norm.T @ act_norm) / len(activations)

    rows = np.arange(C) // grid_w
    cols = np.arange(C) % grid_w
    dist = np.sqrt((rows[:, None] - rows[None, :]) ** 2 +
                   (cols[:, None] - cols[None, :]) ** 2)

    idx = np.triu_indices(C, k=1)
    sims = sim[idx]
    dists = dist[idx]

    # Bin by distance
    max_dist = dists.max()
    bins = np.linspace(0, max_dist, 20)
    bin_means, bin_edges = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (dists >= lo) & (dists < hi)
        if mask.sum() > 0:
            bin_means.append(sims[mask].mean())
            bin_edges.append((lo + hi) / 2)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(dists, sims, alpha=0.02, s=1, color="steelblue")
    ax.plot(bin_edges, bin_means, color="red", linewidth=2, label="bin mean")
    ax.set_xlabel("Grid distance")
    ax.set_ylabel("Cosine similarity")
    ax.set_title(f"Similarity vs. distance — {label}")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def compare_models(activations_baseline: np.ndarray,
                   activations_structured: np.ndarray,
                   grid_h: int, grid_w: int,
                   save_path: str | None = None):
    """Side-by-side similarity-vs-distance for baseline and structured model."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for ax, acts, name in zip(axes,
                               [activations_baseline, activations_structured],
                               ["Baseline", "Structured"]):
        C = acts.shape[1]
        norms = np.linalg.norm(acts, axis=0, keepdims=True)
        act_norm = acts / (norms + 1e-8)
        sim = (act_norm.T @ act_norm) / len(acts)

        rows = np.arange(C) // grid_w
        cols = np.arange(C) % grid_w
        dist = np.sqrt((rows[:, None] - rows[None, :]) ** 2 +
                       (cols[:, None] - cols[None, :]) ** 2)
        idx = np.triu_indices(C, k=1)
        sims, dists = sim[idx], dist[idx]

        bins = np.linspace(0, dists.max(), 20)
        bm, be = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (dists >= lo) & (dists < hi)
            if mask.sum() > 0:
                bm.append(sims[mask].mean())
                be.append((lo + hi) / 2)

        ax.scatter(dists, sims, alpha=0.02, s=1, color="steelblue")
        ax.plot(be, bm, color="red", linewidth=2)
        ax.set_title(name)
        ax.set_xlabel("Grid distance")
        ax.set_ylabel("Cosine similarity")

    plt.suptitle("Similarity vs. grid distance", fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
