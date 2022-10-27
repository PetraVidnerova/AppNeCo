import sys 
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
from torch.optim.lr_scheduler import LambdaLR

PRETRAINED=True if sys.argv[1] == "pretrained" else False
BATCH_SIZE=int(sys.argv[2])


net = torchvision.models.vgg11(pretrained=PRETRAINED)
net.classifier[4] = nn.Linear(4096,1024)
net.classifier[6] = nn.Linear(1024,10)
#net.classifier.append(nn.Softmax(1)) softmax is part of entropy loss
print(net)


transform_train = transforms.Compose([
    transforms.Resize((227,227)),
    transforms.RandomHorizontalFlip(p=0.7),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

torch.manual_seed(42)
train_ds = CIFAR10("../example/data/", train=True, download=True, transform=transform_train) #40,000 original images + transforms

#BATCH_SIZE=512
train_ds, val_ds = torch.utils.data.random_split(train_ds, [45000, 5000])
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = net.to(device=device)

learning_rate = 1e-4
load_model = True
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr= learning_rate) 


net.train()

for epoch in range(1000):
    loss_ep = 0

    print(f"Epoch {epoch} ... ")
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

    with torch.no_grad():
        num_correct = 0
        num_samples = 0
        print("Computing validation accuracy ...")
        with tqdm(total=len(val_dl)) as t:
            for batch_idx, (data,targets) in enumerate(val_dl):
                data = data.to(device=device)
                targets = targets.to(device=device)
                ## Forward Pass
                scores = net(data)
                _, predictions = scores.max(1)
                num_correct += (predictions == targets).sum()
                num_samples += predictions.size(0)
                t.update()
        print(
            f"VAL accuracy: {float(num_correct) / float(num_samples) * 100:.2f}"
        )
    #scheduler.step()
#    print("--->", scheduler.get_last_lr())

torch.save(net, f"vgg11_{'pretrained' if PRETRAINED else 'from_scratch'}_{BATCH_SIZE}.pt")

test_ds = CIFAR10("../example/data/", train=False, download=True,
                  transform=transform_train)
test_dl = DataLoader(test_ds, batch_size=64, num_workers=8, shuffle=False)

net.eval()
num_correct = 0
num_samples = 0
with torch.no_grad():
    for data, targets in tqdm(test_dl):
        data = data.to(device=device)
        targets = targets.to(device=device)
        # Forward Pass
        scores = net(data)
        # geting predictions
        _, predictions = scores.max(1)
        num_correct += (predictions == targets).sum()
        num_samples += predictions.size(0)

    print(
        f"TEST accuracy: {float(num_correct) / float(num_samples) * 100:.2f}"
    )
