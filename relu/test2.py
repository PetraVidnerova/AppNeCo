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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ReLU(nn.ReLU):

    instances = [] 
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counters_inited = False
        ReLU.instances.append(self)
        
    def init_counters(self, shape):
        shape = shape[1:]
        self.only_negative_inputs = torch.full(shape, True, dtype=torch.bool, device=device)
        self.only_positive_inputs = torch.full(shape, True, dtype=torch.bool, device=device)
        self.counters_inited = True
        
    def reset_counters(self):
        self.only_negative_inputs = toch.full_like(self.only_negative_inputs, True, dtype=torch.bool, device=device)
        self.only_positive_inputs = torch.full_like(self.only_positive_inputs, True, dtype=torch.bool, device=device)

    def update_counters(self, input):
        assert input.shape[1:] == self.only_negative_inputs.shape
        batch_size = input.shape[0] 
        negative = torch.le(input, torch.full(input.shape, 0.0, device=device)).sum(dim=0)
        positive = torch.ge(input, torch.full(input.shape, 0.0, device=device)).sum(dim=0)

        negative = torch.eq(negative, batch_size)
        positive = torch.eq(positive, batch_size)
        
        self.only_negative_inputs = torch.logical_and(self.only_negative_inputs, negative)
        self.only_positive_inputs = torch.logical_and(self.only_positive_inputs, positive)

    def only_positive_count(self):
        return self.only_positive_inputs.sum()

    def only_negative_count(self):
        return self.only_negative_inputs.sum()

    def only_zero_inputs(self):
        return torch.logical_and(self.only_negative_inputs, self.only_positive_inputs).sum()

    def number_of_inputs(self):
        return self.only_negative_inputs.numel()
        
    def forward(self, input):
        if not self.counters_inited:
            self.init_counters(input.shape) 
        self.update_counters(input) 
        return super().forward(input)


def hack_network(net):
    for i, module in enumerate(net.features):
        if isinstance(module, nn.ReLU):
            net.features[i] = ReLU(inplace=True)
    for i, module in enumerate(net.classifier):
        if isinstance(module, nn.ReLU):
            net.classifier[i] = ReLU(inplace=True)
    return net 
            
    
net = torchvision.models.AlexNet()
net.classifier[4] = nn.Linear(4096,1024)
net.classifier[6] = nn.Linear(1024,10)

net.load_state_dict(torch.load("cifar10_7a.pt"))
print(net)

net = hack_network(net)
print(net)

print(ReLU.instances)


tranform_train = transforms.Compose([
    transforms.Resize((227,227)),
    transforms.RandomHorizontalFlip(p=0.7),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_ds = CIFAR10("./data/", train=False, download=True, transform=tranform_train) 
test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)

net = net.to(device=device)
net.eval()

with torch.no_grad():
    num_correct = 0
    num_samples = 0

    with tqdm(total=len(test_dl)) as t:
        for batch_idx, (data,targets) in enumerate(test_dl):
            data = data.to(device=device)
            targets = targets.to(device=device)
            scores = net(data)
            _, predictions = scores.max(1)
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)
            t.update()

    print(
        f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
    )

total_inputs = sum([x.number_of_inputs() for x in ReLU.instances])
only_negative_inputs = sum([x.only_negative_count() for x in ReLU.instances])
only_positive_inputs = sum([x.only_positive_count() for x in ReLU.instances])
    
print("All ReLU inputs:", total_inputs)
print("Only negative inputs:", only_negative_inputs, (only_negative_inputs/total_inputs)*100)
print("Only positive inputs:", only_positive_inputs, (only_positive_inputs/total_inputs)*100)

