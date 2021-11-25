import os
import sys

from core import *

import torch

data_transforms = {
    'train': transforms.Compose([
        # transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(225),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

model = torch.load("res50_3.pth")
generate_hw_submission(model, data_transforms['val'])
