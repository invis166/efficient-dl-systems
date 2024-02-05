from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import CIFAR10
from torchvision import transforms

from modeling.diffusion import DiffusionModel
from modeling.training import train_step, generate_samples, train_epoch
from modeling.unet import UnetModel


@pytest.fixture
def set_seed():
    torch.manual_seed(0)

@pytest.fixture
def train_dataset():
    transforms = Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    dataset = CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=transforms,
    )
    return dataset


@pytest.mark.parametrize(["device"], [["cpu"], ["cuda"]])
def test_train_on_one_batch(device, train_dataset, set_seed):
    # note: you should not need to increase the threshold or change the hyperparameters
    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=32),
        betas=(1e-4, 0.02),
        num_timesteps=1000,
        device=device,
    )
    ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=5e-4)
    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False)

    x, _ = next(iter(dataloader))
    loss = None
    for i in range(50):
        loss = train_step(ddpm, x, optim, device)
    assert loss < 0.5


@pytest.mark.parametrize(
        'device,num_epochs,hidden_size,expected_loss',
        [
            ('cpu', 1, 8, 1.03),
            ('cuda', 2, 16, 0.77)
        ]
)
def test_training(
    device,
    num_epochs,
    hidden_size,
    expected_loss,
    train_dataset,
    set_seed,
    tmp_path
):
    # note: implement and test a complete training procedure (including sampling)
    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=hidden_size),
        betas=(1e-4, 0.02),
        num_timesteps=1000,
        device=device,
    )
    ddpm.to(device)

    dataloader = DataLoader(train_dataset, batch_size=128, num_workers=4, shuffle=True)
    optim = torch.optim.Adam(ddpm.parameters(), lr=1e-5)

    loss = None
    for _ in range(num_epochs):
        loss = train_epoch(ddpm, dataloader, optim, device)
        samples_path = tmp_path/'{i:02d}.png'
        generate_samples(ddpm, device, samples_path)

        assert samples_path.exists()

    assert abs(loss.detach().item() - expected_loss) < 1e-3
