import numpy as np
from tqdm import tqdm
import torch
from torch.nn import Dropout
import torchvision.models as models 
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, CIFAR10
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

import sys 

SAMPLE = int(sys.argv[1])
NOISE_LEVEL = float(sys.argv[2])

NETWORK_FILE = "alexnet_from_scratch_cifar10.pt"
OUTPUT_FILE = f"from_scratch_{SAMPLE}_variance_{NOISE_LEVEL}"

print("Loading network ... ")
net = torch.load(NETWORK_FILE)

def flatten(x):
    return torch.flatten(x, 1)

layer_outputs = []
layer_names = [] 
print(net)

layer_outputs.append(lambda x: x)
layer_names.append("Input")
for layer in net.features:
    layer_outputs.append(layer)
    layer_names.append(layer.__class__.__name__)
layer_outputs.append(net.avgpool)
layer_names.append("avgpool")
layer_outputs.append(flatten)
layer_names.append("flatten")
for layer in net.classifier:
    if isinstance(layer, Dropout):
        continue
    layer_outputs.append(layer)
    layer_names.append(layer.__class__.__name__)
print(layer_names)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = net.to(device=device)

print("Loading ImageNet validation data ...")
transform = transforms.Compose([
    transforms.Resize((227,227)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


print("Prepare dataset ... ", end="", flush=True)
#test_ds = ImageFolder(DATA_ROOT, transform=transform)
test_ds = CIFAR10("../example/data/", train=False, download=True,
                  transform=transform)
#test_dl = DataLoader(test_ds, batch_size=64, num_workers=8, shuffle=False)

single_data_point = test_ds[SAMPLE]
print(single_data_point)

class NoiseDataset(Dataset):
    def __init__(self, data_point, noise_level):
        self.point, self.target = data_point
        self.noise_level = noise_level

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        noisy_point =  self.point + self.noise_level * torch.randn(self.point.size())
        return noisy_point, self.target

noisy_data = NoiseDataset(single_data_point, NOISE_LEVEL)
test_dl = DataLoader(noisy_data, batch_size=1, num_workers=8, shuffle=False)

        
output_dict = {
    i: []
    for i, _ in enumerate(layer_outputs)
}


print("Evaluating network ...")
net.eval()
num_correct = 0
num_samples = 0
with torch.no_grad():
    for data, targets in tqdm(test_dl):
        data = data.to(device=device)
        targets = targets.to(device=device)
        # Forward Pass
        #scores = net(data)
        output = data
        for i, layer in enumerate(layer_outputs):
            output = layer(output)
            output_dict[i].append(output.cpu().numpy().flatten())
        scores = output
        # geting predictions
        _, predictions = scores.max(1)
        num_correct += (predictions == targets).sum()
        num_samples += predictions.size(0)

print(
    f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
)

def compute_variance(xlist):
    a = np.vstack(xlist)
    v = np.var(a, axis=0)
    return v

output = 0, 2, 5, 8, 10, 12, 13

variances = []

for o in output:
    outputs = output_dict[o]
    variances.append(compute_variance(outputs))

import pickle
with open(f'{OUTPUT_FILE}.pickle', 'wb') as f:
    pickle.dump(variances, f, protocol=pickle.HIGHEST_PROTOCOL)
    
