from pathlib import Path
from typing import Union

import hydra
import wandb
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image
from omegaconf import DictConfig, OmegaConf

from modeling.diffusion import DiffusionModel
from modeling.training import generate_samples, train_epoch
from modeling.unet import UnetModel


def main():
    cfg = OmegaConf.load("params.yaml")
    print(cfg)
    wandb.init(config=dict(cfg), project='effdl_example', name='baseline-hydra')

    ddpm = _get_model(cfg)
    wandb.watch(ddpm)

    train_transforms_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ]
    if cfg.training.augmentations:
        train_transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
    train_transforms = transforms.Compose(train_transforms_list)

    inverse_norm_transform = transforms.Compose(
        [transforms.Normalize((-0.4914/0.247, -0.4822/0.243, -0.4465/0.261), (1/0.247, 1/0.243, 1/0.261))]
    )

    dataset = CIFAR10(
        cfg.data.output_folder,
        train=True,
        download=False,
        transform=train_transforms,
    )
    _log_samples_from_dataset(dataset, inverse_norm_transform)
    _log_artifact_to_wandb('config', 'conf/', 'config')

    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, num_workers=4, shuffle=True)
    optim = _get_optimizer(ddpm, cfg)

    for i in range(cfg.training.num_epochs):
        loss = train_epoch(ddpm, dataloader, optim, cfg.training.device)

        grid_path = Path().cwd() / 'samples' / f'{i:02d}.png'
        generate_samples(ddpm, cfg.training.device, grid_path, transforms=inverse_norm_transform)
        _log_artifact_to_wandb(grid_path.name, grid_path)

        wandb.log({'loss': loss})


def _log_samples_from_dataset(dataset, inverse_norm, n_samples=8):
    samples = [inverse_norm(dataset[i][0]) for i in range(n_samples)]

    grid = make_grid(samples, nrow=n_samples // 2)
    samples_grid_path = Path().cwd() / 'samples' / 'sample_from_dataset.png'
    samples_grid_path.parent.mkdir(exist_ok=True)
    save_image(grid, samples_grid_path)

    _log_artifact_to_wandb('sample_from_dataset', samples_grid_path)


def _log_artifact_to_wandb(artifact_name: str, artifact_path: Union[Path, str], artifact_type: str = 'dataset'):
    artifact_path = Path(artifact_path)
    if artifact_path.is_dir():
        wandb.log_artifact(artifact_path, artifact_name, artifact_type)
    else:
        artifact = wandb.Artifact(artifact_name, type=artifact_type)
        artifact.add_file(artifact_path)
        wandb.log_artifact(artifact)


def _get_model(cfg: DictConfig):
    ddpm = DiffusionModel(
        eps_model=UnetModel(
            cfg.model.in_channels,
            cfg.model.out_channels,
            hidden_size=cfg.model.hidden_size
        ),
        betas=cfg.model.betas,
        num_timesteps=cfg.model.num_timesteps,
        device=cfg.training.device,
    )
    ddpm.to(cfg.training.device)

    return ddpm


def _get_optimizer(model: nn.Module, cfg: DictConfig):
    if cfg.optimizer.type == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)
    elif cfg.optimizer.type == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum)
    else:
        raise ValueError(f'Invalid optimizer type: {cfg.optimizer.type}')


if __name__ == "__main__":
    torch.manual_seed(0)

    main()
