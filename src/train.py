import argparse
import csv
import os
import shutil
import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from data.cifar10 import get_cifar10_loaders
from losses.structured_loss import StructuredLoss
from models.resnet import get_resnet18


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--no-structured", action="store_true",
                        help="train baseline with cross-entropy only")
    parser.add_argument("--exp-name", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    use_structured = not args.no_structured
    exp_name = args.exp_name or ("structured" if use_structured else "baseline")
    exp_dir = os.path.join(cfg["output"]["exp_dir"], exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # save config so we can reproduce this exact run later
    shutil.copy(args.config, os.path.join(exp_dir, "config.yaml"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | run: {exp_name}")

    train_loader, test_loader = get_cifar10_loaders(
        data_dir=cfg["data"]["data_dir"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        augment=cfg["data"]["augment"],
        subset_frac=cfg["data"].get("subset_frac", 1.0),
    )

    model = get_resnet18(
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
    ).to(device)

    # grab layer4 activations via a forward hook so we can pass them to
    # the structured loss without modifying the model architecture at all
    activations = {}
    if use_structured:
        def save_activations(module, input, output):
            activations["feat"] = output
        getattr(model, cfg["loss"]["target_layer"]).register_forward_hook(save_activations)

    ce_criterion = nn.CrossEntropyLoss()

    if use_structured:
        structured_criterion = StructuredLoss(
            grid_h=cfg["loss"]["grid_h"],
            grid_w=cfg["loss"]["grid_w"],
            lambda_smooth=cfg["loss"]["lambda_smooth"],
            lambda_comp=cfg["loss"]["lambda_comp"],
        )

    # SGD with momentum and cosine LR decay -- standard for CIFAR-10
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["train"]["lr"],
        momentum=cfg["train"]["momentum"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])

    # write per-epoch stats to CSV so we can plot and compare runs later
    log_path = os.path.join(exp_dir, "log.csv")
    log_fields = ["epoch", "train_acc", "test_acc", "loss_total",
                  "loss_ce", "loss_smooth", "loss_comp"]
    log_file = open(log_path, "w", newline="")
    logger = csv.DictWriter(log_file, fieldnames=log_fields)
    logger.writeheader()

    best_acc = 0.0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        running = {"total": 0.0, "ce": 0.0, "smooth": 0.0, "comp": 0.0}
        correct = total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch:3d}", leave=False):
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

            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_acc = 100.0 * correct / total
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()

        n = len(train_loader)
        loss_str = f"loss {running['total']/n:.4f}"
        if use_structured:
            loss_str += (f" (ce {running['ce']/n:.3f}"
                         f" sm {running['smooth']/n:.3f}"
                         f" comp {running['comp']/n:.3f})")
        print(f"Epoch {epoch:3d} | train {train_acc:.2f}% | test {test_acc:.2f}% | {loss_str}")

        logger.writerow({
            "epoch": epoch,
            "train_acc": round(train_acc, 4),
            "test_acc": round(test_acc, 4),
            "loss_total": round(running["total"] / n, 6),
            "loss_ce": round(running["ce"] / n, 6),
            "loss_smooth": round(running["smooth"] / n, 6),
            "loss_comp": round(running["comp"] / n, 6),
        })
        log_file.flush()

        if cfg["output"]["save_best"] and test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pth"))

    log_file.close()
    torch.save(model.state_dict(), os.path.join(exp_dir, "final_model.pth"))
    print(f"done. best test acc: {best_acc:.2f}% | results in {exp_dir}")


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


if __name__ == "__main__":
    main()
