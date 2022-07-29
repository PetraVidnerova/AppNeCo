import numpy as np
from tqdm import tqdm
import torch
from torch.nn import Dropout
import torchvision.models as models 
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, CIFAR10
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

from random import randrange


import sys 

from utils import test

NOISE_LEVEL = 0.2

VARIANT="from_scratch"

NETWORK_FILE = f"alexnet_{VARIANT}_cifar10.pt"
#OUTPUT_FILE = f"from_scratch_{SAMPLE}_variance_{NOISE_LEVEL}"

print("Loading network ... ")
net = torch.load(NETWORK_FILE)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = net.to(device=device)

print("Loading  data ...")
transform = transforms.Compose([
    transforms.Resize((227,227)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


print("Prepare dataset ... ", end="", flush=True)
#test_ds = ImageFolder(DATA_ROOT, transform=transform)
test_ds = CIFAR10("../example/data/", train=False, download=True,
                  transform=transform)
test_dl = DataLoader(test_ds, batch_size=256, num_workers=16, shuffle=False)
test(net, test_dl, device)

class NoiseDataset(Dataset):
    def __init__(self, base_data_set, noise_level):
        self.base_data_set = base_data_set 
        self.noise_level = noise_level

    def __len__(self):
        return 1000*len(self.base_data_set)

    def __getitem__(self, idx):
        point, target = self.base_data_set[idx // 1000]
        noisy_point =  point + self.noise_level * torch.randn(point.size())
        return noisy_point, target

class NoiseDatasetSingle(Dataset):
    def __init__(self, base_data_set, noise_level):
        self.base_data_set = base_data_set 
        self.noise_level = noise_level
        self.point = randrange(len(base_data_set))
        self.first = True
        
    def __len__(self):
        return 10000 #*len(self.base_data_set)

    def __getitem__(self, idx):
        point, target = self.base_data_set[self.point]
        if self.first:
            print("===>", target)
            self.first = False
        noisy_point =  point + self.noise_level * torch.randn(point.size())
        return noisy_point, target

    
#noisy_data = NoiseDataset(test_ds, NOISE_LEVEL)
#test_dl = DataLoader(noisy_data, batch_size=256, num_workers=16, shuffle=False)
noisy_data = NoiseDatasetSingle(test_ds, NOISE_LEVEL)
test_dl = DataLoader(noisy_data, batch_size=1, num_workers=16, shuffle=False)

test(
    net, test_dl, device,
    save_outputs=True,
    filename=f"alexnet_{NOISE_LEVEL}_{VARIANT}"
)
