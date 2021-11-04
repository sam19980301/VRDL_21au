import os
import torch

DATA_PATH = r'../CUB200'  # Image path after converting into pytorch.Imagefolder style
# Image path of the given homework datasets
DATA_PATH_ = r'../2021VRDL_HW1_datasets'

N_CLASSES = 200
BATCH_SIZE = 8

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
