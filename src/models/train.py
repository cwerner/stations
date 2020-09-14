from pathlib import Path

import hydra
import numpy as np
import torch
import wandb

# A logger for this file
from loguru import logger as log
from omegaconf import DictConfig, OmegaConf
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from models.simplecgan import Discriminator, Generator

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5,), std=(0.5,))
    ]
)


@hydra.main(config_path="conf", config_name="config.yaml")
def my_app(cfg: DictConfig) -> None:

    wandb.init(project="cgan-mnist-demo", config=cfg)
    # run_id = wandb.run.id

    # Decide which device we want to run on
    device = torch.device("cuda:0" if cfg.cuda else "cpu")
    log.info(f"Cuda status: {'enabled' if cfg.cuda else 'disabled'} [{device}]")
    log.info(OmegaConf.to_yaml(cfg))

    INPUT_SIZE = 784  # 28x28
    SAMPLE_SIZE = 80  # 8x10 samples as check image
    NUM_LABELS = 10  # 10 classes

    # data
    base_path = Path(hydra.utils.get_original_cwd())
    data_dir = base_path / Path(cfg.data_dir)
    sample_dir = base_path / Path(cfg.sample_dir)
    model_dir = base_path / Path(cfg.model_dir)

    for d in [data_dir, sample_dir, model_dir]:
        d.mkdir(parents=True, exist_ok=True)

    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=cfg.batch_size)

    # models
    model_D = Discriminator().to(device)
    model_G = Generator(cfg.nz).to(device)

    wandb.watch(model_D)
    wandb.watch(model_G)

    criterion = nn.BCELoss()

    input = torch.FloatTensor(cfg.batch_size, INPUT_SIZE)
    noise = torch.FloatTensor(cfg.batch_size, (cfg.nz))
    label = torch.FloatTensor(cfg.batch_size)
    labels_onehot = torch.FloatTensor(cfg.batch_size, 10)

    fixed_noise = torch.randn(SAMPLE_SIZE, cfg.nz).to(device)

    # TODO: check the outcome and simplify
    fixed_labels = torch.zeros(SAMPLE_SIZE, NUM_LABELS)
    for i in range(NUM_LABELS):
        for j in range(SAMPLE_SIZE // NUM_LABELS):
            fixed_labels[i * (SAMPLE_SIZE // NUM_LABELS) + j, i] = 1.0

    # use GPU

    if cfg.cuda:
        input = input.cuda()
        label = label.cuda()
        noise = noise.cuda()
        labels_onehot = labels_onehot.cuda()
        fixed_labels = fixed_labels.cuda()

    optim_discriminator = optim.SGD(model_D.parameters(), lr=cfg.optimizer.lr)
    optim_generator = optim.SGD(model_G.parameters(), lr=cfg.optimizer.lr)

    # fixed_noise = Variable(fixed_noise)
    fixed_labels = Variable(fixed_labels)

    real_label, fake_label = 1, 0

    for epoch in range(cfg.epochs):
        model_D.train()
        model_G.train()

        loss_discriminator, loss_generator = 0.0, 0.0

        for batch_idx, (train_x, train_y) in enumerate(train_loader):
            batch_size = train_x.size(0)

            # real image
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
            labelv = Variable(label).unsqueeze(dim=1)

            # descriminator on real image
            out_d = model_D(inputv, Variable(labels_onehot))
            optim_discriminator.zero_grad()

            errD_real = criterion(out_d, labelv)
            errD_real.backward()

            realD_mean = out_d.data.cpu().mean()

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
            labelv = Variable(label).unsqueeze(dim=1)
            onehotv = Variable(labels_onehot)

            # generator on fake image
            fake_image = model_G(noisev, onehotv)

            # descriminator on real image
            out_d = model_D(fake_image, onehotv)

            errD_fake = criterion(out_d, labelv)
            fakeD_mean = out_d.data.cpu().mean()
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
            labelv = Variable(label).unsqueeze(dim=1)
            g_out = model_G(noisev, onehotv)
            d_out = model_D(g_out, onehotv)
            errG = criterion(d_out, labelv)

            optim_generator.zero_grad()
            errG.backward()
            optim_generator.step()

            loss_discriminator += errD.data.item()
            loss_generator += errG.data.item()

            if batch_idx % 10 == 0:
                log.info(
                    f"{epoch:02d} ({batch_idx:03d}/{len(train_loader)}) mean D(fake)"
                    f"= {fakeD_mean:.5f}, mean D(real) = {realD_mean:.5f}"
                )

                g_out = (
                    model_G(fixed_noise, fixed_labels)
                    .data.view(SAMPLE_SIZE, 1, 28, 28)
                    .cpu()
                )

                save_image(g_out, f"{sample_dir}/{epoch:02}_{batch_idx:03}.png")

            wandb.log(
                {
                    "g_loss_train": errG.data.item(),
                    "d_loss_train": errD.data.item(),
                    "d_fake_mean": fakeD_mean,
                    "d_real_mean": realD_mean,
                    "examples": wandb.Image(
                        model_G(fixed_noise, fixed_labels)
                        .data.view(SAMPLE_SIZE, 1, 28, 28)
                        .cpu()
                    ),
                }
            )

        print(
            f"Epoch {epoch} - D loss = {loss_discriminator:.4f}, "
            f"G loss = {loss_generator:.4f}"
        )
        if epoch % 2 == 0:
            torch.save(
                {"state_dict": model_D.state_dict()},
                f"{model_dir}/model_d_epoch_{epoch}.pth",
            )
            torch.save(
                {"state_dict": model_G.state_dict()},
                f"{model_dir}/model_g_epoch_{epoch}.pth",
            )


if __name__ == "__main__":
    my_app()
