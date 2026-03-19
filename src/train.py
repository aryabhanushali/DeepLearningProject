"""
Training script for baseline and structured ResNet18 on CIFAR-10.

Usage:
    # Baseline (cross-entropy only)
    python src/train.py --config configs/default.yaml --no-structured

    # Structured loss
    python src/train.py --config configs/default.yaml
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data.cifar10 import get_cifar10_loaders
from models.resnet import get_resnet18
from losses.structured_loss import StructuredLoss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--no-structured", action="store_true",
                        help="Train baseline without structured loss")
    parser.add_argument("--exp-name", default=None,
                        help="Override experiment directory name")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    use_structured = not args.no_structured
    exp_name = args.exp_name or ("structured" if use_structured else "baseline")
    exp_dir = os.path.join(cfg["output"]["exp_dir"], exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  |  experiment: {exp_name}")

    # Data
    train_loader, test_loader = get_cifar10_loaders(
        data_dir=cfg["data"]["data_dir"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        augment=cfg["data"]["augment"],
    )

    # Model
    model = get_resnet18(
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
    ).to(device)

    # Register forward hook on target layer to capture activations
    activations = {}
    target_layer = cfg["loss"]["target_layer"]

    def hook_fn(module, input, output):
        activations["feat"] = output

    getattr(model, target_layer).register_forward_hook(hook_fn)

    # Loss
    ce_criterion = nn.CrossEntropyLoss()
    structured_criterion = None
    if use_structured:
        structured_criterion = StructuredLoss(
            grid_h=cfg["loss"]["grid_h"],
            grid_w=cfg["loss"]["grid_w"],
            lambda_smooth=cfg["loss"]["lambda_smooth"],
            lambda_comp=cfg["loss"]["lambda_comp"],
        )

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["train"]["lr"],
        momentum=cfg["train"]["momentum"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])

    best_acc = 0.0
    log_interval = cfg["output"]["log_interval"]

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        running = {k: 0.0 for k in ["total", "ce", "smooth", "comp"]}
        correct = total = 0

        for step, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            logits = model(images)
            ce_loss = ce_criterion(logits, labels)

            if use_structured:
                loss, metrics = structured_criterion(ce_loss, activations["feat"])
                running["ce"] += metrics["loss/ce"]
                running["smooth"] += metrics["loss/smooth"]
                running["comp"] += metrics["loss/comp"]
            else:
                loss = ce_loss
                running["ce"] += ce_loss.item()

            running["total"] += loss.item()
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (step + 1) % log_interval == 0:
                n = step + 1
                print(
                    f"  step {n:4d} | total {running['total']/n:.4f} "
                    f"| ce {running['ce']/n:.4f} "
                    + (f"| smooth {running['smooth']/n:.4f} | comp {running['comp']/n:.4f}"
                       if use_structured else "")
                )

        train_acc = 100.0 * correct / total
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()

        print(f"Epoch {epoch:3d} | train acc {train_acc:.2f}% | test acc {test_acc:.2f}%")

        if cfg["output"]["save_best"] and test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pth"))

    torch.save(model.state_dict(), os.path.join(exp_dir, "final_model.pth"))
    print(f"Done. Best test acc: {best_acc:.2f}%")


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


if __name__ == "__main__":
    main()
