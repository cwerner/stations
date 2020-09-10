from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from models.simplecgan import Discriminator, Generator


@hydra.main(config_path="conf", config_name="config.yaml")
def my_app(cfg: DictConfig) -> None:

    cfg.cuda = torch.cuda.is_available()

    print(OmegaConf.to_yaml(cfg))

    INPUT_SIZE = 784
    SAMPLE_SIZE = 80
    NUM_LABELS = 10

    # data
    data_dir = Path(hydra.utils.get_original_cwd()) / Path(cfg.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = Path(hydra.utils.get_original_cwd()) / Path(cfg.sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(hydra.utils.get_original_cwd()) / Path(cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transforms.ToTensor()
    )

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=cfg.batch_size)

    # models
    discriminator = Discriminator()
    generator = Generator(cfg.nz)

    criterion = nn.BCELoss()

    input = torch.FloatTensor(cfg.batch_size, INPUT_SIZE)
    noise = torch.FloatTensor(cfg.batch_size, (cfg.nz))
    label = torch.FloatTensor(cfg.batch_size)
    labels_onehot = torch.FloatTensor(cfg.batch_size, 10)

    fixed_noise = torch.FloatTensor(SAMPLE_SIZE, cfg.nz).normal_(0, 1)

    # TODO: check the outcome and simplify
    fixed_labels = torch.zeros(SAMPLE_SIZE, NUM_LABELS)
    for i in range(NUM_LABELS):
        for j in range(SAMPLE_SIZE // NUM_LABELS):
            fixed_labels[i * (SAMPLE_SIZE // NUM_LABELS) + j, i] = 1.0

    # use GPU

    if cfg.cuda:
        generator.cuda()
        discriminator.cuda()
        input, label = input.cuda(), label.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
        labels_onehot = labels_onehot.cuda()
        fixed_labels = fixed_labels.cuda()

    optim_discriminator = optim.SGD(discriminator.parameters(), lr=cfg.optimizer.lr)
    optim_generator = optim.SGD(generator.parameters(), lr=cfg.optimizer.lr)

    fixed_noise = Variable(fixed_noise)
    fixed_labels = Variable(fixed_labels)

    real_label, fake_label = 1, 0

    for epoch in range(cfg.epochs):
        discriminator.train()
        generator.train()

        loss_discriminator, loss_generator = 0.0, 0.0

        for batch_idx, (train_x, train_y) in enumerate(train_loader):
            batch_size = train_x.size(0)
            train_x = train_x.view(-1, INPUT_SIZE)

            if cfg.cuda:
                train_x = train_x.cuda()
                train_y = train_y.cuda()

            # TODO: check these lines
            input.resize_as_(train_x).copy_(train_x)
            label.resize_(batch_size).fill_(real_label)

            labels_onehot.resize_(batch_size, NUM_LABELS).zero_()

            # TODO: check out scatter
            labels_onehot.scatter_(1, train_y.view(batch_size, 1), 1)

            inputv = Variable(input)
            labelv = Variable(label)

            output = discriminator(inputv, Variable(labels_onehot))
            optim_discriminator.zero_grad()

            errD_real = criterion(output, labelv)
            errD_real.backward()

            realD_mean = output.data.cpu().mean()

            labels_onehot.zero_()
            rand_y = torch.from_numpy(
                np.random.randint(0, NUM_LABELS, size=(batch_size, 1))
            )
            if cfg.cuda:
                rand_y = rand_y.cuda()

            labels_onehot.scatter_(1, rand_y.view(batch_size, 1), 1)
            noise.resize_(batch_size, cfg.nz).normal_(0, 1)
            label.resize_(batch_size).fill_(fake_label)

            noisev = Variable(noise)
            labelv = Variable(label)
            onehotv = Variable(labels_onehot)

            g_out = generator(noisev, onehotv)
            output = discriminator(g_out, onehotv)

            errD_fake = criterion(output, labelv)
            fakeD_mean = output.data.cpu().mean()
            errD = errD_real + errD_fake
            errD_fake.backward()

            optim_discriminator.step()

            # train the G
            noise.normal_(0, 1)
            labels_onehot.zero_()
            rand_y = torch.from_numpy(
                np.random.randint(0, NUM_LABELS, size=(batch_size, 1))
            )
            if cfg.cuda:
                rand_y = rand_y.cuda()

            labels_onehot.scatter_(1, rand_y.view(batch_size, 1), 1)
            label.resize_(batch_size).fill_(real_label)
            onehotv = Variable(labels_onehot)

            noisev = Variable(noise)
            labelv = Variable(label)
            g_out = generator(noisev, onehotv)
            output = discriminator(g_out, onehotv)
            errG = criterion(output, labelv)

            optim_generator.zero_grad()
            errG.backward()
            optim_generator.step()

            loss_discriminator += errD.data.item()
            loss_generator += errG.data.item()

            if batch_idx % 5 == 0:
                print(
                    f"\t {epoch} ({batch_idx} / {len(train_loader)}) mean D(fake)"
                    f"= {fakeD_mean}, mean D(real) = {realD_mean}"
                )

                g_out = (
                    generator(fixed_noise, fixed_labels)
                    .data.view(SAMPLE_SIZE, 1, 28, 28)
                    .cpu()
                )

                save_image(g_out, f"{sample_dir}/{epoch:02}_{batch_idx:03}.png")

        print(
            f"Epoch {epoch} - D loss = {loss_discriminator:.4f}, "
            f"G loss = {loss_generator:.4f}"
        )
        if epoch % 2 == 0:
            torch.save(
                {"state_dict": discriminator.state_dict()},
                f"{model_dir}/model_d_epoch_{epoch}.pth",
            )
            torch.save(
                {"state_dict": generator.state_dict()},
                f"{model_dir}/model_g_epoch_{epoch}.pth",
            )


if __name__ == "__main__":
    my_app()
