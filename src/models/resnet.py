import torch.nn as nn
import torchvision.models as tv_models


def get_resnet18(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = tv_models.resnet18(weights=weights)
    # Replace classifier head for CIFAR-10
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # CIFAR-10 images are 32x32; shrink the initial conv & remove maxpool
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model
