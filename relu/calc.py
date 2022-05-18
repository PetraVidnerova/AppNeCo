from tqdm import tqdm 
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from torchsummary import summary 
    
net = torchvision.models.AlexNet()
net.classifier[4] = nn.Linear(4096,1024)
net.classifier[6] = nn.Linear(1024,10)

net.to("cuda:0")

summary(net, (3,227,227))
