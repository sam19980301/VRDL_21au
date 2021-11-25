import os
import sys

from core import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
from torchvision.datasets import ImageFolder

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

# train_dataset = BirdDataset(
#     annotations_file = os.path.join(DATA_PATH,'sub_train.txt'),
#     img_dir = os.path.join(DATA_PATH,'training_images/'),
#     transform = data_transforms['train'],
#     target_transform = None)

# val_dataset = BirdDataset(
#     annotations_file = os.path.join(DATA_PATH,'sub_test.txt'),
#     img_dir = os.path.join(DATA_PATH,'training_images/'),
#     transform = data_transforms['val'],
#     target_transform = None)

train_dataset = ImageFolder(
    root=os.path.join(
        DATA_PATH,
        'training_images'),
    transform=data_transforms['train'])
val_dataset = ImageFolder(
    root=os.path.join(
        DATA_PATH,
        'testing_images'),
    transform=data_transforms['val'])

image_datasets = {'train': train_dataset, 'val': val_dataset}

dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0) for x in [
            'train',
        'val']}

# model_conv = models.resnet34(pretrained=True)
# model_conv = models.resnet50(pretrained=True)
model_conv = models.resnet101(pretrained=True)
# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, N_CLASSES)
model_conv = model_conv.to(device)


# for param in model_conv.parameters():
#     param.requires_grad = False
criterion = nn.CrossEntropyLoss()
# optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
# # optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=2.5e-3, momentum=0.9, weight_decay=2.25e-3)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=10, gamma=0.1)
# model_conv = train_model(model_conv, dataloaders, criterion, optimizer_conv,
#                          exp_lr_scheduler, num_epochs=30) # 30

for param in model_conv.parameters():
    param.requires_grad = True
optimizer_ft = optim.SGD(
    model_conv.parameters(),
    lr=1e-3,
    momentum=0.9,
    weight_decay=1e-3)  # LR: 2.5e-3 WD: 1e-4, 5e-4
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)
exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=150)
model_conv = train_model(model_conv, dataloaders, criterion, optimizer_ft,
                         exp_lr_scheduler, num_epochs=150)  # 200

torch.save(model_conv, 'testtt.pth')
