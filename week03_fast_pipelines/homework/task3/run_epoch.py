import itertools
import typing as tp

import torch
import torch.nn as nn
import torch.optim as optim
import dataset
import pandas as pd

from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import Settings, Clothes, seed_everything
from vit import ViT

from torch.profiler import profile, record_function, ProfilerActivity, schedule


def get_vit_model() -> torch.nn.Module:
    model = ViT(
        depth=12,
        heads=4,
        image_size=224,
        patch_size=32,
        num_classes=20,
        channels=3,
    ).to(Settings.device)
    return model


def get_loaders() -> torch.utils.data.DataLoader:
    dataset.download_extract_dataset()
    train_transforms = dataset.get_train_transforms()
    val_transforms = dataset.get_val_transforms()

    frame = pd.read_csv(f"{Clothes.directory}/{Clothes.csv_name}")
    train_frame = frame.sample(frac=Settings.train_frac)
    val_frame = frame.drop(train_frame.index)

    train_data = dataset.ClothesDataset(
        f"{Clothes.directory}/{Clothes.train_val_img_dir}",
        train_frame,
        transform=train_transforms,
    )
    val_data = dataset.ClothesDataset(
        f"{Clothes.directory}/{Clothes.train_val_img_dir}",
        val_frame,
        transform=val_transforms,
    )

    print(f"Train Data: {len(train_data)}")
    print(f"Val Data: {len(val_data)}")

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=Settings.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=Settings.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader


def run_epoch(model, train_loader, val_loader, criterion, optimizer) -> tp.Tuple[float, float]:
    my_schedule = schedule(
        skip_first=1,
        wait=0,
        warmup=2,
        active=15,
        repeat=1
    )
    with profile(
        activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
        schedule=my_schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/vit'),
        profile_memory=True,
    ) as prof:
        epoch_loss, epoch_accuracy = torch.tensor(0.0, dtype=torch.float, device=Settings.device), \
            torch.tensor(0, dtype=torch.float, device=Settings.device)
        val_loss, val_accuracy = torch.tensor(0.0).to(Settings.device), torch.tensor(0).to(Settings.device)

        with record_function('model_train'):
            model.train()
            for data, label in tqdm(train_loader, desc="Train"):
                with record_function('batch_to_device'):
                    data = data.to(Settings.device)
                    label = label.to(Settings.device)
                with record_function('model_forward'):
                    output = model(data)
                loss = criterion(output, label)
                optimizer.zero_grad()
                with record_function('model_backward'):
                    loss.backward()
                optimizer.step()
                with record_function('metrics'):
                    acc = (output.argmax(dim=1) == label).float().mean()
                    epoch_accuracy += acc.detach() / len(train_loader)
                    epoch_loss += loss.detach() / len(train_loader)
                prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

    # removed (made 100% optimal) the eval code because the task is about the training pipeline ¯\_(ツ)_/¯

    return epoch_loss, epoch_accuracy, val_loss, val_accuracy



def main():
    seed_everything()
    model = get_vit_model()
    train_loader, val_loader = get_loaders()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Settings.lr)

    run_epoch(model, train_loader, val_loader, criterion, optimizer)


if __name__ == "__main__":
    main()
