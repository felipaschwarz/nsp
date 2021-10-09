import torch.nn as nn
import torch.nn.functional as F

class Network_mnist_1(nn.Module):
    def __init__(self):
        super(Network_mnist_1, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(320, 10)

    def forward(self, x):
        out = F.relu(self.cnn1(x))
        out = self.maxpool1(out)
        out = F.relu(self.cnn2(out))
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out
