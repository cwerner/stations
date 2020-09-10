import torch
import torch.nn.functional as F
from torch import nn


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 28 * 28 + 1000, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.fc3 = nn.Linear(10, 1000)

    def forward(self, x, labels):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 28, 28)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # concat
        x = x.view(batch_size, 64 * 28 * 28)
        y_ = F.relu(self.fc3(labels))
        x = torch.cat([x, y_], 1)

        x = self.fc2(F.relu(self.fc1(x)))

        return torch.sigmoid(x)


class Generator(nn.Module):
    def __init__(self, z_dim):
        self.z_dim = z_dim
        super(Generator, self).__init__()
        self.fc2 = nn.Linear(10, 1000)
        self.fc = nn.Linear(self.z_dim + 1000, 64 * 28 * 28)
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.deconv2 = nn.ConvTranspose2d(32, 1, 5, 1, 2)

    def forward(self, x, labels):
        batch_size = x.size(0)
        y_ = F.relu(self.fc2(labels))

        x = torch.cat([x, y_], 1)

        x = self.fc(x)
        x = x.view(batch_size, 64, 28, 28)
        x = F.relu(self.bn1(x))

        x = F.relu(self.bn2(self.deconv1(x)))
        x = self.deconv2(x)

        return torch.sigmoid(x)
