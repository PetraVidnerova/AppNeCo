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

NET_TYPE = sys.argv[1] 

NETWORK_FILE = f"alexnet_{NET_TYPE}_cifar10.pt"

#
# Prepare network 
#
print("Loading network ... ")
net = torch.load(NETWORK_FILE)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = net.to(device=device)

print(net)

#
# Track outputs 
#
def flatten(x):
    return torch.flatten(x, 1)

#layer_outputs=[layer for layer in net.features]
#layer_outputs.append(net.avgpool)
#layer_outputs.append(flatten)
#layer_outputs.extend(
layer_outputs = [layer for layer in net.classifier if not isinstance(layer, Dropout)]
layer_names = [layer.__class__.__name__ for layer in net.classifier if not isinstance(layer, Dropout)]
#)
first_part = lambda x: flatten(net.avgpool(net.features(x)))

#
# Prepare dataset 
# 
print("Prepare dataset ... ", end="", flush=True)
transform = transforms.Compose([
    transforms.Resize((227,227)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_ds = CIFAR10("../example/data/", train=False, download=True,
                  transform=transform)
test_dl = DataLoader(test_ds, batch_size=1, num_workers=8, shuffle=False)


#
# Eval network 
#
print("Evaluating network ...")
net.eval()
num_correct = 0
num_samples = 0
output_values_set=[]

with torch.no_grad():
    for data, targets in tqdm(test_dl):
        output_values_list = [] 
        data = data.to(device=device)
        targets = targets.to(device=device)
        # Forward Pass
        #scores = net(data)
        output = first_part(data)
        for i, layer in enumerate(layer_outputs):
            output = layer(output)
            if layer_names[i] == "ReLU":
                output_map = output<=0
                #            output_values_list.append(output_map.cpu().numpy().flatten())
                output_values_list.append(flatten(output_map))
        scores = output
        # geting predictions
        _, predictions = scores.max(1)
        num_correct += (predictions == targets).sum()
        num_samples += predictions.size(0)
        output_values = torch.hstack(output_values_list)
        if not any([torch.all(x == output_values) for x in output_values_set]):
            output_values_set.append(output_values)
        else:
            print("JE TAM!")
        
print(num_correct/num_samples)
print(len(output_values_set))

