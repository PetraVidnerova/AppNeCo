from operator import mul
from functools import reduce

from tqdm import tqdm 
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim

class ReLU(nn.ReLU):

    counter_all = 0
    counter_negative = 0
    
    def forward(self, input):
        ReLU.counter_all += 1

        size = reduce(mul,list(input.shape))
        ReLU.counter_all += size
        ReLU.counter_negative += torch.sum(torch.lt(input, 0)).item()
        return super().forward(input)

net = torchvision.models.AlexNet()

print(net)

print(net.features)

for i, module in enumerate(net.features):
    if isinstance(module, nn.ReLU):
        net.features[i] = ReLU(inplace=True)
        
for i, module in enumerate(net.classifier):
    if isinstance(module, nn.ReLU):
        net.classifier[i] = ReLU(inplace=True)

        
print(net)

tranform_train = transforms.Compose([
    transforms.Resize((227,227)),
    transforms.RandomHorizontalFlip(p=0.7),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

torch.manual_seed(43)
train_ds = CIFAR10("/opt/data/bench_net/", train=True, download=True, transform=tranform_train) #40,000 original images + transforms
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.to(device=device)

learning_rate = 1e-4 
load_model = True
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr= learning_rate) 

for epoch in range(1):
    loss_ep = 0

    with tqdm(total=len(train_dl)) as t:
        for batch_idx, (data, targets) in enumerate(train_dl):
            data = data.to(device=device)
            targets = targets.to(device=device)
            ## Forward Pass
            optimizer.zero_grad()
            scores = net(data)
            loss = criterion(scores,targets)
            loss.backward()
            optimizer.step()
            loss_ep += loss.item()
            t.update()
    print(f"Loss in epoch {epoch} :::: {loss_ep/len(train_dl)}")

    # with torch.no_grad():
    #     num_correct = 0
    #     num_samples = 0
    #     for batch_idx, (data,targets) in enumerate(val_dl):
    #         data = data.to(device=device)
    #         targets = targets.to(device=device)
    #         ## Forward Pass
    #         scores = net(data)
    #         _, predictions = scores.max(1)
    #         num_correct += (predictions == targets).sum()
    #         num_samples += predictions.size(0)
    #     print(
    #         f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
    #     )

print("all calls", ReLU.counter_all)
print("negatives", ReLU.counter_negative)

print(f"{(ReLU.counter_negative/ReLU.counter_all)*100:.2f}%")
