import torch
import torchvision

# -------- CONFIGURE --------
# Choose dataset MNIST, CIFAR10 or FashionMNIST
dataset = 'MNIST'
# -------- CONFIGURE --------


if dataset == 'MNIST':
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    testset = torchvision.datasets.MNIST(root='./datasets', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

elif dataset == 'CIFAR10':
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

elif dataset == 'FashionMNIST':
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    testset = torchvision.datasets.FashionMNIST(root='./datasets', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

    classes = ('T-Shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot')

else:
    raise NotImplementedError
