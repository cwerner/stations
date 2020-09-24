import sys

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import wandb
from models.cdcgan import Discriminator, Generator, weights_init
from utils.io.imaging import save_image
from utils.io.logging import log
from utils.io.pathlib_extensions import Path

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


@hydra.main(config_path="conf", config_name="config.yaml")
def my_app(cfg: DictConfig) -> None:

    wandb.init(project="cgan-mnist-demo", config=cfg, tags=["test"])
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

    input = torch.FloatTensor(cfg.batch_size, INPUT_SIZE).to(device)
    noise = torch.FloatTensor(cfg.batch_size, (cfg.nz)).to(device)
    label = torch.FloatTensor(cfg.batch_size).to(device)
    label_id = torch.FloatTensor(cfg.batch_size, 10).to(device)

    fixed_noise = torch.randn(SAMPLE_SIZE, cfg.nz).to(device)

    # TODO: check the outcome and simplify
    fixed_labels = torch.zeros(SAMPLE_SIZE, NUM_LABELS)
    for i in range(NUM_LABELS):
        for j in range(SAMPLE_SIZE // NUM_LABELS):
            fixed_labels[i * (SAMPLE_SIZE // NUM_LABELS) + j, i] = 1.0
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
        except FileNotFoundError:
            log.warning("No checkpoint present, starting from scratch")

    if cfg.clean:
        log.info("Cleaning checkpoint and sample directories")
        [child.unlink() for child in model_dir.iterdir()]
        [child.unlink() for child in sample_dir.iterdir()]

    real_label, fake_label = 1, 0

    for epoch in range(start_epoch, cfg.epochs):
        D.train()
        G.train()

        loss_discriminator, loss_generator = 0.0, 0.0

        for batch_idx, (train_x, train_y) in enumerate(train_loader):
            batch_size = train_x.size(0)

            # real image
            # original shape: (bs) x (c) x (w) x (h)
            train_x = train_x.view(-1, INPUT_SIZE).to(device)  # (bs) 128 x (data) 784
            train_y = train_y.to(device)  # (bs)

            # copy data into input and mark with label=1
            input.resize_as_(train_x).copy_(train_x)

            # TODO: rename label to y_isreal, label_id to train_y_onehot
            label.resize_(batch_size).fill_(real_label)
            label_id.resize_(batch_size, NUM_LABELS).zero_()

            # one-hot encoded class id
            label_id.scatter_(1, train_y.view(batch_size, 1), 1)

            # descriminator on real image
            d_real = D(input, label_id)

            optim_D.zero_grad()

            d_real_err = criterion(d_real, label.unsqueeze(dim=1))
            d_real_err.backward()

            d_real_mean = d_real.data.cpu().mean()

            label_id.zero_()
            rand_y = torch.from_numpy(
                np.random.randint(0, NUM_LABELS, size=(batch_size, 1))
            )
            rand_y = rand_y.to(device)

            label_id.scatter_(1, rand_y.view(batch_size, 1), 1)
            noise.resize_(batch_size, cfg.nz).normal_(0, 1)
            label.resize_(batch_size).fill_(fake_label)

            # generator on fake image
            fake_image = G(noise, label_id)

            # descriminator on real image
            d_fake = D(fake_image, label_id)

            d_fake_err = criterion(d_fake, label.unsqueeze(dim=1))
            d_fake_mean = d_fake.data.cpu().mean()
            d_err = (d_real_err + d_fake_err) / 2

            d_fake_err.backward()

            optim_D.step()

            # train the G
            noise.normal_(0, 1)
            label_id.zero_()
            rand_y = torch.from_numpy(
                np.random.randint(0, NUM_LABELS, size=(batch_size, 1))
            )
            rand_y = rand_y.to(device)

            label_id.scatter_(1, rand_y.view(batch_size, 1), 1)
            label.resize_(batch_size).fill_(real_label)

            fake_image = G(noise, label_id)
            d_fake = D(fake_image, label_id)
            g_err = criterion(d_fake, label.unsqueeze(dim=1))

            optim_G.zero_grad()
            g_err.backward()
            optim_G.step()

            loss_discriminator += d_err.data.item()
            loss_generator += g_err.data.item()

            if batch_idx % 10 == 0:
                log.info(
                    f"{epoch:02d} ({batch_idx:03d}/{len(train_loader)}) mean D(fake)"
                    f"= {d_fake_mean:.5f}, mean D(real) = {d_real_mean:.5f}"
                )

                fake_image = (
                    G(fixed_noise, fixed_labels).data.view(SAMPLE_SIZE, 1, 28, 28).cpu()
                )
                img_label = (
                    f"Epoch:{epoch:02d} [{batch_idx:04d}/{len(train_loader):04d}]"
                )
                save_image(
                    fake_image,
                    f"{sample_dir}/{epoch:02}_{batch_idx:03}.png",
                    label=img_label,
                    label2="cDCGAN",
                )

            wandb.log(
                {
                    "g_loss_train": g_err.data.item(),
                    "d_loss_train": d_err.data.item(),
                    "d_fake_mean": d_fake_mean,
                    "d_real_mean": d_real_mean,
                    "examples": wandb.Image(
                        G(fixed_noise, fixed_labels)
                        .data.view(SAMPLE_SIZE, 1, 28, 28)
                        .cpu()
                    ),
                }
            )

        if cfg.checkpoint:
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
