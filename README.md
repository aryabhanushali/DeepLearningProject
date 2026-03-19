# Encouraging Spatial Organization in CNNs Through a Structured Loss

**CS 4644/7643 — Arya Bhanushali, Aryan Thakkar**

Explore if a modified loss function can encourage spatially organized internal representations in CNNs without changing the architecture.
Channels are arranged on a 2-D grid and regularized with two terms:
- **Local smoothness** — neighboring channels should activate similarly.
- **Global competition** — distant channels should not learn the same features.


```
DeepLearningProject/
├── configs/
│   └── default.yaml          # hyperparameters & loss weights
├── src/
│   ├── data/
│   │   └── cifar10.py        # CIFAR-10
│   ├── models/
│   │   └── resnet.py         # ResNet18 adapted for CIFAR-10
│   ├── losses/
│   │   └── structured_loss.py  # L_smooth + L_comp regularization
│   ├── train.py              # training script (baseline & structured)
│   ├── evaluate.py           # accuracy + spatial organization metrics
│   └── vis.py                # visualization utilities
└── experiments/              # saved checkpoints & logs (gitignored)
```
