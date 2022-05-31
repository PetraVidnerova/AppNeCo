from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader

print("Loading net ...")
if not torch.cuda.is_available():
    net = torch.load("alexnet_cifar10.pt", map_location=torch.device('cpu'))
else:
    net = torch.load("alexnet_cifar10.pt")
print(net)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.to(device=device)


print("Loading CIRAR10 test data ...")
tranform_train = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_ds = CIFAR10("./data/", train=False, download=True,
                  transform=tranform_train)
test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)


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
        # geting predictions
        _, predictions = scores.max(1)
        num_correct += (predictions == targets).sum()
        num_samples += predictions.size(0)

print(
    f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
)
