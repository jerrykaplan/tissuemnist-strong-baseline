"""
ResNet-18 architecture adapted for 28x28 single-channel inputs, plus
TissueMNIST dataset wrapper with optional geometric augmentation.

The architecture adaptation follows standard MedMNIST practice for
low-resolution inputs: the initial 7x7 stride-2 convolution is replaced
with a 3x3 stride-1 convolution, and the initial max-pool is removed.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class BasicBlock(nn.Module):
    """Standard ResNet basic block (used by ResNet-18 and ResNet-34)."""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return F.relu(out)


class ResNet18_28(nn.Module):
    """
    ResNet-18 adapted for 28x28 single-channel inputs.

    Differences from standard ResNet-18:
      - First conv is 3x3 stride 1 (not 7x7 stride 2)
      - Initial max-pool is removed
      - Input is single channel by default (grayscale)
    """
    def __init__(self, num_classes=8, in_channels=1):
        super().__init__()
        self.in_channels = 64

        # Modified stem for 28x28 inputs
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # (no initial maxpool)

        # Standard ResNet-18 body: 4 stages of 2 blocks each
        self.layer1 = self._make_layer(64,  2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # Standard init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                         nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, out_channels, n_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [BasicBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def features(self, x):
        """Penultimate-layer features (for LDA-on-features experiments)."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class TissueDS(Dataset):
    """
    TissueMNIST dataset wrapper with optional geometric augmentation.

    Augmentation (applied per sample at each epoch when augment=True):
      - Rotation drawn uniformly from [-15, +15] degrees
      - Translation up to +/- 2 pixels in each axis
      - Horizontal flip with probability 0.5

    Normalization: inputs are scaled from [0, 255] uint8 to [-1, 1] float32.
    """
    def __init__(self, X, y, augment=False,
                 rotation_range=15.0, translation_pixels=2, flip_prob=0.5):
        self.X = X  # uint8 array, shape (N, 28, 28)
        self.y = y.flatten().astype(np.int64)
        self.augment = augment
        self.rotation_range = rotation_range
        self.translation_pixels = translation_pixels
        self.flip_prob = flip_prob

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx].astype(np.float32) / 255.0  # [0, 1]
        img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1, 28, 28)

        if self.augment:
            # Rotation
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            # Translation
            dx = np.random.randint(-self.translation_pixels,
                                    self.translation_pixels + 1)
            dy = np.random.randint(-self.translation_pixels,
                                    self.translation_pixels + 1)
            img_tensor = TF.affine(img_tensor, angle=angle,
                                    translate=[dx, dy], scale=1.0, shear=[0.0, 0.0])
            # Horizontal flip
            if np.random.random() < self.flip_prob:
                img_tensor = TF.hflip(img_tensor)

        # Normalize to [-1, 1]
        img_tensor = (img_tensor - 0.5) / 0.5
        return img_tensor, torch.tensor(self.y[idx])
