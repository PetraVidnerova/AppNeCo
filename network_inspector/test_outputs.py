import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.nn import Dropout, Sequential
import torchvision.models as models 
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, CIFAR10
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

DATA_ROOT = "/opt/data/imagenet_raw/"
NETWORK_NAME = "alexnet_from_scratch_cifar10"
NETWORK_FILE = f"{NETWORK_NAME}.pt"

print("Loading network ... ")
net = torch.load(NETWORK_FILE)
print(net)

#net.classifier = net.classifier[:-1]
#print(net)

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
# DO NOT CHANGE BATCH SIZE, must be 1
test_dl = DataLoader(test_ds, batch_size=1, num_workers=8, shuffle=False)

        
df_list = [] 

print("Evaluating network ...")
net.eval()
num_correct = 0
num_samples = 0
with torch.no_grad():
    for data, targets in tqdm(test_dl):
        data = data.to(device=device)
        targets = targets.to(device=device)
        # Forward Pass
        scores = net(data)

        target = targets.cpu().numpy()[0]
        #print(target)
        outputs = scores.cpu().numpy()[0]
        #print(outputs)
        for i, out in enumerate(outputs):
            #print(i, ":", out)
            df_list.append({"class": target, "output": i, "value": out})
        # geting predictions
        _, predictions = scores.max(1)
        num_correct += (predictions == targets).sum()
        num_samples += predictions.size(0)

print(
    f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
)

df = pd.DataFrame(df_list)
print(df)

df.to_parquet(f"{NETWORK_NAME}_outputs.parquet")
