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

id_ = int(sys.argv[1])

net = torchvision.models.AlexNet()
print(net)

net.classifier[4] = nn.Linear(4096,1024)
net.classifier[6] = nn.Linear(1024,10)
print(net)


tranform_train = transforms.Compose([
    transforms.Resize((227,227)),
    transforms.RandomHorizontalFlip(p=0.7),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

torch.manual_seed(id_)
train_ds = CIFAR10("./data/", train=True, download=True, transform=tranform_train) #40,000 original images + transforms

train_ds, val_ds = torch.utils.data.random_split(train_ds, [45000, 5000])
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)

test_ds = CIFAR10("./data/", train=False, download=True, transform=tranform_train) 
test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = net.to(device=device)

learning_rate = 1e-4 
load_model = True
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr= learning_rate) 

for epoch in range(20):
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

torch.save(net.state_dict(), f"cifar10_net{id_}.pt")

# test set
net.eval()
correct = 0
total = 0
with torch.no_grad():
    with tqdm(total=len(test_dl)) as t:
        for data in test_dl:
            images, labels = data
            images, labels = images.to(device=device), labels.to(device=device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            t.update()

print(f'Accuracy of the network on test images: {100 * correct / total:.2f} %')
