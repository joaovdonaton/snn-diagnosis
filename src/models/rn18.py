import torch
from torch import nn
from torch.nn import Linear
from torchvision import models

model_18 = models.resnet18(weights=None)
model_18.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False) # change to 1 input channel (resnet18 is originally for 3-channel RGB)
model_18.fc = Linear(in_features=512, out_features=2) # binary output