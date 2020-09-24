import torch
from torch import nn

# TODO: make MNIST dataset specifics configurable


class View(nn.Module):
    """View implementation for use in nn.Sequential"""

    def __init__(self, shape):
        super().__init__()
        self.shape = (shape,)

    def forward(self, x):
        return x.view(*self.shape)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Discriminator(nn.Module):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__(**kwargs)

        self.ylabel = nn.Sequential(nn.Linear(10, 1000), nn.ReLU(True))

        self.main = nn.Sequential(
            # reshape
            View((-1, 1, 28, 28)),
            nn.Conv2d(1, 32, 5, 1, 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            View((-1, 64 * 28 * 28)),
        )

        self.head = nn.Sequential(
            nn.Linear(64 * 28 * 28 + 1000, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, label):

        x = self.main(x)
        y = self.ylabel(label)

        x = torch.cat((x, y), dim=1)

        out = self.head(x)

        return out


class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()

        self.ylabel = nn.Sequential(nn.Linear(10, 1000), nn.ReLU(True))

        self.main = nn.Sequential(
            nn.Linear(z_dim + 1000, 64 * 28 * 28),
            View((-1, 64, 28, 28)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 5, 1, 2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 5, 1, 2, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, label):
        # bs = x.size(0)

        y = self.ylabel(label)

        x = torch.cat((x, y), dim=1)
        out = self.main(x)
        return out
