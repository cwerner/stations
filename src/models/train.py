import sys

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import wandb
from models.model.cdcgan import Discriminator, Generator, weights_init
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

    wandb.init(project=f"cdcgan-{cfg.dataset.name}", config=cfg, tags=[])
    run_id = wandb.run.id
    log.info(f"cDCGAN Dataset:{cfg.dataset.name}")
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
    data_dir = Path().home() / ".torch" / "datasets"

    for d in [data_dir, sample_dir, model_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if cfg.dataset.name == "mnist":
        train_data = datasets.MNIST(
            data_dir, train=True, download=True, transform=transform
        )
    elif cfg.dataset.name == "fashionmnist":
        train_data = datasets.FashionMNIST(
            data_dir, train=True, download=True, transform=transform
        )
    elif cfg.dataset.name == "tereno":
        raise NotImplementedError
    else:
        log.error(f"Dataset {cfg.dataset.name} not valid")
        raise ValueError

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

    # fixed data (noise and labels)
    fixed_noise = torch.randn(SAMPLE_SIZE, cfg.nz).to(device)
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

    G_losses = []
    D_losses = []

    for epoch in range(start_epoch, cfg.epochs):
        D.train()
        G.train()
        for batch_idx, (train_x, train_y) in enumerate(train_loader):
            bs = train_x.size(0)

            # real image original shape: (bs) x (c) x (w) x (h)
            train_x = train_x.view(-1, INPUT_SIZE).to(device)  # (bs) 128 x (data) 784
            train_y = train_y.to(device)  # (bs)

            # copy data into input and mark with label=1
            label = torch.FloatTensor(cfg.batch_size).to(device)
            label.resize_(bs).fill_(real_label)

            # one-hot encoded class id
            label_id = torch.FloatTensor(cfg.batch_size, 10).to(device)
            label_id.resize_(bs, NUM_LABELS).zero_()
            label_id.scatter_(1, train_y.view(bs, 1), 1)

            # forward pass real batch (D), calc loss and
            # calc gradient for backward pass
            D.zero_grad()
            output = D(train_x, label_id).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            rand_y = torch.from_numpy(
                np.random.randint(0, NUM_LABELS, size=(bs, 1))
            ).to(device)
            label_id.zero_()
            label_id.scatter_(1, rand_y.view(bs, 1), 1)

            noise = torch.FloatTensor(bs, (cfg.nz)).normal_(0, 1).to(device)
            label = torch.FloatTensor(bs).to(device)
            label.resize_(bs).fill_(fake_label)

            # generator on fake image
            fake = G(noise, label_id)

            # descriminator on real image
            output = D(fake.detach(), label_id).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            # add gradients from all-real and all-fake batches
            errD = errD_real + errD_fake

            optim_D.step()

            # train the G network
            G.zero_grad()

            rand_y = torch.from_numpy(
                np.random.randint(0, NUM_LABELS, size=(bs, 1))
            ).to(device)
            label_id = torch.FloatTensor(bs, 10).to(device)
            label_id.zero_()
            label_id.scatter_(1, rand_y.view(bs, 1), 1)

            label = torch.FloatTensor(bs).to(device)
            label.fill_(real_label)

            # since we just updated D run it again on all fake
            # TODO: check if detach() here is correct
            output = D(fake, label_id).view(-1)

            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()

            optim_G.step()

            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if batch_idx % 10 == 0:
                log.info(
                    f"{epoch:02d} ({batch_idx:03d}/{len(train_loader)}) LossD = "
                    f"{errD.item():.5f}, LossG = {errG.item():.5f}, "
                    f"D(x): {D_x:.5f}, D(G(z)): {D_G_z1:.5f}/{D_G_z2:.5f}"
                )

                with torch.no_grad():
                    fake_image = (
                        G(fixed_noise, fixed_labels)
                        .detach()
                        .view(SAMPLE_SIZE, 1, 28, 28)
                        .cpu()
                    )
                img_label = (
                    f"Epoch:{epoch:02d} [{batch_idx:04d}/{len(train_loader):04d}]"
                )
                im = save_image(
                    fake_image,
                    f"{sample_dir}/{epoch:02}_{batch_idx:03}.png",
                    label=img_label,
                    label2="cDCGAN",
                )
                wandb.log({"sample": wandb.Image(im)}, commit=False)

            wandb.log(
                {
                    "g_loss": errG.item(),
                    "d_loss": errD.item(),
                    "D(x)": D_x,
                    "D_G_z1": D_G_z1,
                    "D_G_z2": D_G_z2,
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
