import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import math

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2):
        super(GhostModule, self).__init__()
        self.oup = oup  # Define the oup attribute
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride=1, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, 3, stride=1, padding=1, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]

class AttentionPathway(nn.Module):
    def __init__(self, inp, oup):
        super(AttentionPathway, self).__init__()
        self.conv = nn.Conv2d(inp, oup, 1, 1, 0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv(x))

class LightweightModule(nn.Module):
    def __init__(self, num_classes):
        super(LightweightModule, self).__init__()
        self.ghost_module = GhostModule(inp=2048, oup=2048)
        self.attention_pathway = AttentionPathway(inp=2048, oup=2048)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        ghost = self.ghost_module(x)
        attention = self.attention_pathway(x)
        x = ghost * attention
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x