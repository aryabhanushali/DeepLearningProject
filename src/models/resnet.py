import torch.nn as nn
import torchvision.models as tv_models


def get_resnet18(num_classes=10, pretrained=False):
    weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = tv_models.resnet18(weights=weights)

    # The default ResNet18 uses a 7x7 conv with stride 2 and a maxpool, which
    # is designed for 224x224 ImageNet images. For CIFAR-10 (32x32), this
    # aggressively downsamples the input too early. We replace conv1 with a
    # smaller 3x3 stride-1 conv and remove the maxpool so that layer4 still
    # produces 4x4 spatial feature maps instead of collapsing to 1x1.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    # swap out the final classifier for CIFAR-10 (10 classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
