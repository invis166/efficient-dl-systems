from pathlib import Path
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from modeling.diffusion import DiffusionModel


def train_step(model: DiffusionModel, inputs: torch.Tensor, optimizer: Optimizer, device: str):
    optimizer.zero_grad()
    inputs = inputs.to(device)
    loss = model(inputs)
    loss.backward()
    optimizer.step()
    return loss


def train_epoch(model: DiffusionModel, dataloader: DataLoader, optimizer: Optimizer, device: str) -> float:
    model.train()
    pbar = tqdm(dataloader)
    loss_ema = None
    for x, _ in pbar:
        train_loss = train_step(model, x, optimizer, device)
        loss_ema = train_loss if loss_ema is None else 0.9 * loss_ema + 0.1 * train_loss
        pbar.set_description(f"loss: {loss_ema:.4f}")

    return loss_ema


def generate_samples(model: DiffusionModel, device: str, path: str, transforms=None, num_samples=8):
    Path(path).parent.mkdir(exist_ok=True)
    model.eval()
    with torch.no_grad():
        samples = model.sample(num_samples, (3, 32, 32), device=device)
        if transforms:
            samples = transforms(samples)
        grid = make_grid(samples, nrow=num_samples // 2)
        save_image(grid, path)
    
    return samples, grid
