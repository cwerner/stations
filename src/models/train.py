import sys
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

from models.simplecgan import Discriminator, Generator, weights_init


class NoCheckpointError(BaseException):
    pass


# TODO: move to auxillary file
def latest(self: Path, pattern: str = "*"):
    files = self.glob(pattern)
    try:
        max(files, key=lambda x: x.stat().st_ctime)
    except ValueError:
        raise NoCheckpointError
    return max(files, key=lambda x: x.stat().st_ctime)


Path.latest = latest

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5,), std=(0.5,))
    ]
)


# small loguru modification to allow multi-line logging
def formatter(record):
    lines = record["message"].splitlines()
    prefix = (
        "{time:YY-MM-DD HH:mm:ss.S} | {level.name:<8} | "
        + "{file}.{function}:{line} - ".format(**record)
    )
    indented = (
        lines[0] + "\n" + "\n".join(" " * len(prefix) + line for line in lines[1:])
    )
    record["message"] = indented.strip()
    return (
        "<g>{time:YY-MM-DD HH:mm:ss.S}</> | <lvl>{level.name:<8}</> | "
        + "<e>{file}.{function}:{line}</> - <lvl>{message}\n</>{exception}"
    )


log.remove()
log.add(sys.stderr, format=formatter)


@hydra.main(config_path="conf", config_name="config.yaml")
def my_app(cfg: DictConfig) -> None:

    wandb.init(project="cgan-mnist-demo", config=cfg)
    run_id = wandb.run.id
    log.info(f"Run ID: {run_id}")

    # Decide which device we want to run on
    device = torch.device("cuda:0" if cfg.cuda else "cpu")
    log.info(f"Cuda status {'enabled' if cfg.cuda else 'disabled'} [{device}]")
    log.info(OmegaConf.to_yaml(cfg))

    INPUT_SIZE = 784  # 28x28
    SAMPLE_SIZE = 80  # 8x10 samples as check image
    NUM_LABELS = 10  # 10 classes

    # data
    base_path = Path(hydra.utils.get_original_cwd())
    sample_dir = base_path / Path(cfg.sample_dir)
    model_dir = base_path / Path(cfg.model_dir)
    data_dir = Path().home() / ".torch" / "datasets" / "mnist"

    for d in [data_dir, sample_dir, model_dir]:
        d.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_data = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    # test_data = datasets.MNIST(data_dir, train=False, transform=transform)

    train_loader = DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    # test_loader = DataLoader(
    #    test_data, batch_size=cfg.batch_size, num_workers=4, pin_memory=True
    # )

    # setup models
    D = Discriminator().to(device).apply(weights_init)
    G = Generator(cfg.nz).to(device).apply(weights_init)

    wandb.watch(D)
    wandb.watch(G)

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

    input = input.to(device)
    label = label.to(device)
    noise = noise.to(device)
    labels_onehot = labels_onehot.to(device)
    fixed_labels = fixed_labels.to(device)

    if cfg.optimizer.type == "sgd":
        optim_D = optim.SGD(D.parameters(), lr=cfg.optimizer.lr)
        optim_G = optim.SGD(G.parameters(), lr=cfg.optimizer.lr)
    elif cfg.optimizer.type == "adam":
        optim_D = optim.Adam(
            D.parameters(), lr=cfg.optimizer.lr, betas=(cfg.optimizer.beta, 0.999)
        )
        optim_G = optim.Adam(
            G.parameters(), lr=cfg.optimizer.lr, betas=(cfg.optimizer.beta, 0.999)
        )
    else:
        log.error(f"Optimizer setup '{cfg.optimizer.type}' not valid")
        sys.exit(-1)

    start_epoch = 0

    if cfg.resume:
        try:
            checkpoint = torch.load(model_dir.latest("checkpoint_epoch_*.tar"))
            start_epoch = checkpoint["epoch"] + 1
            D.load_state_dict(checkpoint["D_state_dict"])
            G.load_state_dict(checkpoint["G_state_dict"])
            optim_D.load_state_dict(checkpoint["optim_D_state_dict"])
            optim_G.load_state_dict(checkpoint["optim_G_state_dict"])
            if start_epoch >= cfg.epochs:
                cfg.epochs = cfg.epochs + cfg.epochs
                log.warning("Resume epoch > config number of epochs.")
            log.info(f"Resuming run. Starting at epoch {start_epoch}/ {cfg.epochs}")
        except NoCheckpointError:
            log.warning("No checkpoint present, starting from scratch")

    if cfg.clean:
        log.info("Cleaning checkpoint and sample directories")
        [child.unlink() for child in model_dir.iterdir()]
        [child.unlink() for child in sample_dir.iterdir()]

    # fixed_noise = Variable(fixed_noise)
    fixed_labels = Variable(fixed_labels)

    real_label, fake_label = 1, 0

    for epoch in range(start_epoch, cfg.epochs):
        D.train()
        G.train()

        loss_discriminator, loss_generator = 0.0, 0.0

        for batch_idx, (train_x, train_y) in enumerate(train_loader):
            batch_size = train_x.size(0)

            # real image
            train_x = train_x.view(-1, INPUT_SIZE)
            train_x = train_x.to(device)
            train_y = train_y.to(device)

            # TODO: check these lines
            input.resize_as_(train_x).copy_(train_x)
            label.resize_(batch_size).fill_(real_label)

            labels_onehot.resize_(batch_size, NUM_LABELS).zero_()

            # TODO: check out scatter
            labels_onehot.scatter_(1, train_y.view(batch_size, 1), 1)

            inputv = Variable(input)
            labelv = Variable(label).unsqueeze(dim=1)

            # descriminator on real image
            out_d = D(inputv, Variable(labels_onehot))
            optim_D.zero_grad()

            errD_real = criterion(out_d, labelv)
            errD_real.backward()

            realD_mean = out_d.data.cpu().mean()

            labels_onehot.zero_()
            rand_y = torch.from_numpy(
                np.random.randint(0, NUM_LABELS, size=(batch_size, 1))
            )
            rand_y = rand_y.to(device)

            labels_onehot.scatter_(1, rand_y.view(batch_size, 1), 1)
            noise.resize_(batch_size, cfg.nz).normal_(0, 1)
            label.resize_(batch_size).fill_(fake_label)

            noisev = Variable(noise)
            labelv = Variable(label).unsqueeze(dim=1)
            onehotv = Variable(labels_onehot)

            # generator on fake image
            fake_image = G(noisev, onehotv)

            # descriminator on real image
            out_d = D(fake_image, onehotv)

            errD_fake = criterion(out_d, labelv)
            fakeD_mean = out_d.data.cpu().mean()
            errD = errD_real + errD_fake
            errD_fake.backward()

            optim_D.step()

            # train the G
            noise.normal_(0, 1)
            labels_onehot.zero_()
            rand_y = torch.from_numpy(
                np.random.randint(0, NUM_LABELS, size=(batch_size, 1))
            )
            rand_y = rand_y.to(device)

            labels_onehot.scatter_(1, rand_y.view(batch_size, 1), 1)
            label.resize_(batch_size).fill_(real_label)
            onehotv = Variable(labels_onehot)

            noisev = Variable(noise)
            labelv = Variable(label).unsqueeze(dim=1)
            g_out = G(noisev, onehotv)
            d_out = D(g_out, onehotv)
            errG = criterion(d_out, labelv)

            optim_G.zero_grad()
            errG.backward()
            optim_G.step()

            loss_discriminator += errD.data.item()
            loss_generator += errG.data.item()

            if batch_idx % 10 == 0:
                log.info(
                    f"{epoch:02d} ({batch_idx:03d}/{len(train_loader)}) mean D(fake)"
                    f"= {fakeD_mean:.5f}, mean D(real) = {realD_mean:.5f}"
                )

                g_out = (
                    G(fixed_noise, fixed_labels).data.view(SAMPLE_SIZE, 1, 28, 28).cpu()
                )

                save_image(g_out, f"{sample_dir}/{epoch:02}_{batch_idx:03}.png")

            wandb.log(
                {
                    "g_loss_train": errG.data.item(),
                    "d_loss_train": errD.data.item(),
                    "d_fake_mean": fakeD_mean,
                    "d_real_mean": realD_mean,
                    "examples": wandb.Image(
                        G(fixed_noise, fixed_labels)
                        .data.view(SAMPLE_SIZE, 1, 28, 28)
                        .cpu()
                    ),
                }
            )

        torch.save(
            {
                "D_state_dict": D.state_dict(),
                "G_state_dict": G.state_dict(),
                "optim_D_state_dict": optim_D.state_dict(),
                "optim_G_state_dict": optim_G.state_dict(),
                "epoch": epoch,
            },
            model_dir / f"checkpoint_epoch_{epoch:002d}.tar",
        )


if __name__ == "__main__":
    my_app()
