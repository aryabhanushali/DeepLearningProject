"""
Train any missing experiments, evaluate all of them, and print the ablation table.

Usage:
    python3 ablation.py
"""

import os, sys, csv
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data.cifar10 import get_cifar10_loaders
from models.resnet import get_resnet18
from losses.alternative_losses import SmoothOnlyLoss
from losses.adaptive_loss import AdaptiveSpatialLoss


ALL_EXPERIMENTS = [
    ("baseline",             "configs/default.yaml",              "Baseline (CE only)"),
    ("structured",           "configs/default.yaml",              "Original (L4 smooth+comp)"),
    ("layer3_smooth_only",   "configs/layer3_smooth_only.yaml",   "L3 smooth only"),
    ("high_lambda",          "configs/high_lambda.yaml",          "L3 high-λ (0.5)"),
    ("adaptive",             "configs/adaptive.yaml",             "L3 adaptive grid"),
    ("layer3_medium_lambda", "configs/layer3_medium_lambda.yaml", "L3 medium-λ (0.1)"),
]

# only train these if they don't have checkpoints yet
TO_TRAIN = ["layer3_medium_lambda"]


def _build_criterion(cfg):
    loss_type = cfg["loss"].get("type", "smooth_only")
    gh, gw, ls = cfg["loss"]["grid_h"], cfg["loss"]["grid_w"], cfg["loss"]["lambda_smooth"]
    if loss_type == "adaptive":
        return AdaptiveSpatialLoss(gh, gw, lambda_smooth=ls,
                                   reassign_every=cfg["loss"].get("reassign_every", 2))
    return SmoothOnlyLoss(gh, gw, lambda_smooth=ls)


def train(cfg, exp_name):
    exp_dir = os.path.join(cfg["output"]["exp_dir"], exp_name)
    checkpoint = os.path.join(exp_dir, "best_model.pth")
    if os.path.exists(checkpoint):
        rows = [r for r in csv.DictReader(open(os.path.join(exp_dir, "log.csv")))
                if r["test_acc"]]
        if len(rows) >= cfg["train"]["epochs"]:
            print(f"[{exp_name}] already trained -- skipping")
            return

    os.makedirs(exp_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[{exp_name}] training on {device}")

    train_loader, test_loader = get_cifar10_loaders(
        data_dir=cfg["data"]["data_dir"], batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"], augment=cfg["data"]["augment"],
        subset_frac=cfg["data"].get("subset_frac", 1.0),
    )

    model = get_resnet18(num_classes=cfg["model"]["num_classes"]).to(device)

    activations = {}
    def save_act(_m, _i, out):
        activations["feat"] = out
    getattr(model, cfg["loss"]["target_layer"]).register_forward_hook(save_act)

    ce_crit = nn.CrossEntropyLoss()
    spatial_crit = _build_criterion(cfg)
    base_lambda = cfg["loss"]["lambda_smooth"]
    warmup = cfg["loss"].get("warmup_epochs", 0)

    opt = torch.optim.SGD(model.parameters(), lr=cfg["train"]["lr"],
                          momentum=cfg["train"]["momentum"],
                          weight_decay=cfg["train"]["weight_decay"])
    scheduler = CosineAnnealingLR(opt, T_max=cfg["train"]["epochs"])

    log_file = open(os.path.join(exp_dir, "log.csv"), "w", newline="")
    logger = csv.DictWriter(log_file,
        fieldnames=["epoch", "train_acc", "test_acc", "loss_total",
                    "loss_ce", "loss_smooth", "loss_comp"])
    logger.writeheader()

    best_acc = 0.0
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        if warmup > 0:
            spatial_crit.lambda_smooth = base_lambda * min((epoch - 1) / warmup, 1.0)

        model.train()
        running = {"total": 0.0, "ce": 0.0, "smooth": 0.0}
        correct = total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch:2d}", leave=False):
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad()
            logits = model(images)
            ce_loss = ce_crit(logits, labels)
            loss, metrics = spatial_crit(ce_loss, activations["feat"])
            running["total"] += loss.item()
            running["ce"]    += metrics["loss/ce"]
            running["smooth"] += metrics["loss/smooth"]
            loss.backward()
            opt.step()
            correct += (logits.argmax(1) == labels).sum().item()
            total   += labels.size(0)

        train_acc = 100.0 * correct / total
        test_acc = _eval(model, test_loader, device)
        scheduler.step()

        if hasattr(spatial_crit, "maybe_reassign"):
            spatial_crit.maybe_reassign(epoch)

        n = len(train_loader)
        print(f"  Epoch {epoch:2d} | train {train_acc:.2f}% | test {test_acc:.2f}% "
              f"| ce {running['ce']/n:.3f} sm {running['smooth']/n:.3f}")
        logger.writerow({"epoch": epoch, "train_acc": round(train_acc, 4),
                         "test_acc": round(test_acc, 4),
                         "loss_total": round(running["total"]/n, 6),
                         "loss_ce": round(running["ce"]/n, 6),
                         "loss_smooth": round(running["smooth"]/n, 6),
                         "loss_comp": 0.0})
        log_file.flush()
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), checkpoint)

    log_file.close()
    print(f"[{exp_name}] done -- best test acc: {best_acc:.2f}%")


def _eval(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def spatial_org_score(model, loader, device, target_layer, grid_h, grid_w):
    all_acts = []
    def hook(_m, _i, out):
        all_acts.append(out.mean(dim=(2, 3)).detach().cpu().numpy())
    h = getattr(model, target_layer).register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            model(images.to(device))
    h.remove()
    acts = np.concatenate(all_acts, axis=0)
    C = acts.shape[1]
    norms = np.linalg.norm(acts, axis=0, keepdims=True)
    sim = (acts / (norms + 1e-8)).T @ (acts / (norms + 1e-8))
    rows = np.arange(C) // grid_w
    cols = np.arange(C) % grid_w
    dist = np.sqrt((rows[:, None] - rows[None, :])**2 + (cols[:, None] - cols[None, :])**2)
    idx = np.triu_indices(C, k=1)
    return float(np.corrcoef(sim[idx], -dist[idx])[0, 1])


def main():
    for exp_name in TO_TRAIN:
        cfg_path = next(c for n, c, _ in ALL_EXPERIMENTS if n == exp_name)
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        train(cfg, exp_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = get_cifar10_loaders(
        data_dir="./data", batch_size=128, num_workers=0,
        augment=False, subset_frac=1.0,
    )

    results = []
    for exp_name, config_path, label in ALL_EXPERIMENTS:
        checkpoint = f"experiments/{exp_name}/best_model.pth"
        log_path   = f"experiments/{exp_name}/log.csv"
        if not os.path.exists(checkpoint):
            print(f"[{exp_name}] checkpoint missing -- skipping")
            continue
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        model = get_resnet18(num_classes=10).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        rows = [r for r in csv.DictReader(open(log_path)) if r["test_acc"]]
        best_acc = max(float(r["test_acc"]) for r in rows)
        score = spatial_org_score(model, test_loader, device,
                                  cfg["loss"]["target_layer"],
                                  cfg["loss"]["grid_h"], cfg["loss"]["grid_w"])
        results.append((label, best_acc, score))
        print(f"  evaluated {exp_name}")

    print("\n" + "="*65)
    print(f"{'Experiment':<32} {'Test Acc':>9}  {'Spatial Score':>13}")
    print("-"*65)
    baseline_acc = next(acc for lbl, acc, _ in results if "Baseline" in lbl)
    for label, acc, score in results:
        gap = f"({acc - baseline_acc:+.1f}pp)" if "Baseline" not in label else "        "
        print(f"{label:<32} {acc:>7.2f}%  {gap:>9}  {score:>8.4f}")
    print("="*65)


if __name__ == "__main__":
    main()
