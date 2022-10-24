import sys
import itertools
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader

from utils import test 

VARIANT=sys.argv[1] # [from_scratch | pretrained]

NETWORK_FILE = f"alexnet_{VARIANT}_cifar10.pt"

print("Loading network ... ")
net = torch.load(NETWORK_FILE)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = net.to(device=device)

print(net)

print("Loading  data ...")
transform = transforms.Compose([
    transforms.Resize((227,227)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


print("Prepare dataset ... ", end="", flush=True)
test_ds = CIFAR10("../example/data/", train=False, download=True,
                  transform=transform)
test_dl = DataLoader(test_ds, batch_size=256, num_workers=16, shuffle=False)
test(net, test_dl, device)
test(net, test_dl, device)

i = 0 
for param in net.parameters():
    print(param.data.shape)
    indices = [range(0, x) for x in param.data.shape]
    
    #    print(param.data)
    for x in itertools.product(*indices):
        backup = param.data[x].clone()
        #print("-->", backup.data)
        param.data[x] = 0

        test(net, test_dl, device)
        
        param.data[x] = backup.data
        #        print("-->", backup, param.data[x])
        #        test(net, test_dl, device)
