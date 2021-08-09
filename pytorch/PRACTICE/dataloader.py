import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

test_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

writer = SummaryWriter("logs")
step = 0
for data in test_loader:
    imgs, targets = data
    writer.add_images("load", imgs, step)
    step = step + 1
writer.close()    