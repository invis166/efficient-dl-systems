from pathlib import Path
import wandb
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image

from modeling.diffusion import DiffusionModel
from modeling.training import generate_samples, train_epoch
from modeling.unet import UnetModel
from config import training_options, diffusion_model_options


def main(device: str):
    ddpm = _get_model()
    wandb.watch(ddpm)

    train_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
    )
    inverse_norm = transforms.Compose(
        [transforms.Normalize((-0.4914/0.247, -0.4822/0.243, -0.4465/0.261), (1/0.247, 1/0.243, 1/0.261))]
    )

    dataset = CIFAR10(
        "cifar10",
        train=True,
        download=True,
        transform=train_transforms,
    )
    _log_samples_from_dataset(dataset, inverse_norm)

    dataloader = DataLoader(dataset, batch_size=training_options.batch_size, num_workers=4, shuffle=True)
    optim = torch.optim.Adam(ddpm.parameters(), lr=training_options.lr)

    for i in range(training_options.num_epochs):
        loss = train_epoch(ddpm, dataloader, optim, device)

        grid_path = Path().cwd() / f'{i:02d}.png'
        generate_samples(ddpm, device, grid_path, transforms=inverse_norm)
        _log_artifact_to_wandb(grid_path.name, grid_path)

        wandb.log({'loss': loss})


def _log_samples_from_dataset(dataset, inverse_norm, n_samples=8):
    samples = [inverse_norm(dataset[i][0]) for i in range(n_samples)]

    grid = make_grid(samples, nrow=n_samples // 2)
    samples_grid_path = Path().cwd() / 'sample_from_dataset.png'
    save_image(grid, samples_grid_path)

    _log_artifact_to_wandb('sample_from_dataset', samples_grid_path)


def _log_artifact_to_wandb(artifact_name: str, artifact_path: Path, artifact_type='dataset'):
    artifact = wandb.Artifact(artifact_name, type=artifact_type)
    artifact.add_file(artifact_path)
    wandb.log_artifact(artifact)


def _get_model():
    ddpm = DiffusionModel(
        eps_model=UnetModel(
            diffusion_model_options.in_channels,
            diffusion_model_options.out_channels,
            hidden_size=diffusion_model_options.hidden_size
        ),
        betas=diffusion_model_options.betas,
        num_timesteps=diffusion_model_options.num_timesteps,
        device=device,
    )
    ddpm.to(device)

    return ddpm


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    config = {
        'training_options': training_options,
        'diffusion_model_options': diffusion_model_options
    }

    wandb.init(config=config, project='effdl_example', name='baseline')

    main(device=device)
