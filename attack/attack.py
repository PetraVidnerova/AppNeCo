import time
import sys
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import AlexNet
from torchvision.datasets import  CIFAR10
from torch.utils.data.dataloader import DataLoader

import torchattacks as ta

from utils import test  

VARIANT= sys.argv[1]

NETWORK_FILE = f"../network_inspector/alexnet_{VARIANT}_cifar10.pt"

print("Loading network ... ")
net = torch.load(NETWORK_FILE)


# attracks require imagese in [0,1]
# buth the alexnet is trained to normalized images
# so we add normalization layer (and do not include normalization in transform!)

class Normalize(nn.Module):
    def __init__(self, mean, std) :
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))
        
    def forward(self, input):
        # broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std
   
net = nn.Sequential(
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    net
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = net.to(device=device)

print("Loading  data ...")
transform = transforms.Compose([
    transforms.Resize((227,227)),
    transforms.ToTensor(),
#    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


print("Prepare dataset ... ", end="", flush=True)
test_ds = CIFAR10("../example/data/", train=False, download=True,
                  transform=transform)
test_dl = DataLoader(test_ds, batch_size=256, num_workers=16, shuffle=False)
test(net, test_dl, device)

train_ds = CIFAR10("../example/data/", train=True, download=True,
                   transform=transform)
train_dl = DataLoader(train_ds, batch_size=256, num_workers=16, shuffle=False)
test(net, train_dl, device)

# atk = torchattacks.PGD(net, eps=8/255, alpha=2/255, steps=4)
# #adv_images = atk(images, labels)
# print(atk)

net.eval()

attacks = [
    ta.FGSM(net, eps=8/255),
    ta.BIM(net, eps=8/255, alpha=2/255, steps=100),
    ta.RFGSM(net, eps=8/255, alpha=2/255, steps=100),
    ta.CW(net, c=1, lr=0.01, steps=100, kappa=0),
    ta.PGD(net, eps=8/255, alpha=2/225, steps=100, random_start=True),
    ta.PGDL2(net, eps=1, alpha=0.2, steps=100),
    ta.EOTPGD(net, eps=8/255, alpha=2/255, steps=100, eot_iter=2),
    ta.FFGSM(net, eps=8/255, alpha=10/255),
    ta.TPGD(net, eps=8/255, alpha=2/255, steps=100),
    ta.MIFGSM(net, eps=8/255, alpha=2/255, steps=100, decay=0.1),
    ta.VANILA(net),
    ta.GN(net, std=0.1),
    # ta.APGD(net, eps=8/255, steps=100, eot_iter=1, n_restarts=1, loss='ce'),
    # ta.APGD(net, eps=8/255, steps=100, eot_iter=1, n_restarts=1, loss='dlr'),
    # ta.APGDT(net, eps=8/255, steps=100, eot_iter=1, n_restarts=1),
    ta.FAB(net, eps=8/255, steps=100, n_classes=10, n_restarts=1, targeted=False),
    ta.FAB(net, eps=8/255, steps=100, n_classes=10, n_restarts=1, targeted=True),
    ta.Square(net, eps=8/255, n_queries=5000, n_restarts=1, loss='ce'),
    #    ta.AutoAttack(net, eps=8/255, n_classes=10, version='standard'),
    ta.OnePixel(net, pixels=5, inf_batch=50),
    ta.DeepFool(net, steps=100),
    ta.DIFGSM(net, eps=8/255, alpha=2/255, steps=100, diversity_prob=0.5, resize_rate=0.9)
]


data_loader = DataLoader(test_ds, batch_size=64, num_workers=16, shuffle=False)

for attack in attacks :
    
    print(attack.__class__.__name__)
    
    correct = 0
    total = 0
    start = time.time()
    
    for images, labels in tqdm(data_loader):
        adv_images = attack(images, labels)
        labels = labels.to(device)
        outputs = net(adv_images)

        _, pre = torch.max(outputs.data, 1)

        total += len(images)
        correct += (pre == labels).sum()

    #     imshow(torchvision.utils.make_grid(adv_images.cpu().data, normalize=True), [imagnet_data.classes[i] for i in pre])

    print('Total elapsed time (sec): %.2f' % (time.time() - start))
    print('Robust accuracy: %.2f %%' % (100 * float(correct) / total))

