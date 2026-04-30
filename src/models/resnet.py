import torch.nn as nn
import torchvision.models as tv_models


def get_resnet18(num_classes=10, pretrained=False):
    weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = tv_models.resnet18(weights=weights)

    # replace the 7x7 stride-2 conv and maxpool with a smaller 3x3 conv
    # the default setup is designed for 224x224 ImageNet images and downsamples
    # too aggressively for CIFAR-10's 32x32 inputs
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
