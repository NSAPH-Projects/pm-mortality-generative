import json
from itertools import product
import time

import hydra
import numpy as np
import torch
import torchvision.transforms as transforms
import xarray as xr
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ComponentsWashuDataset(Dataset):
    def __init__(
        self,
        root_dir,
        transform=None,
        components=["pm25", "no3", "so4", "ss", "nh4", "dust", "bc", "om"],
        years=list(range(2000, 2023)),
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.components = components
        self.yyyymm = [
            f"{year}{month:02d}" for year, month in product(years, range(1, 13))
        ]

    def __len__(self):
        return len(self.yyyymm)

    def __getitem__(self, idx):
        yyyymm = self.yyyymm[idx]

        # read files for all components from
        layers = []
        for component in self.components:
            filename = f"{self.root_dir}/{component}/{yyyymm}.nc"
            da = xr.open_dataarray(filename)
            layers.append(da.values)

        tensor = torch.FloatTensor(np.stack(layers, axis=0))

        if self.transform:
            tensor = self.transform(tensor)

        return tensor

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def initialize_dataset(cfg: DictConfig):

    with open(f"{cfg.root_dir}/summary.json", "r") as f:
        summary = json.load(f)

    means = [summary["means"][component] for component in cfg.dataloader_components]
    stds = [summary["stds"][component] for component in cfg.dataloader_components]

    transform = transforms.Resize(cfg.grid_size)
    transform = transforms.Compose(
        [
            transforms.Resize(cfg.grid_size),
            transforms.Normalize(mean=means, std=stds),
        ]
    )

    dataset = ComponentsWashuDataset(
        root_dir=cfg.root_dir,
        transform=transform,
        components=cfg.dataloader_components,
    )

    return dataset
 
@hydra.main(config_path="../conf", config_name="config", version_base=None)
def get_mean_and_std(cfg: DictConfig):
    with open(f"{cfg.root_dir}/summary.json", "r") as f:
        summary = json.load(f)
    return summary["means"], summary["stds"]

def denormalize(tensor):
    device = tensor.device  # Use the same device as the input tensor

    mean, std = get_mean_and_std()
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=device).view(1, -1, 1, 1)  # (1, C, 1, 1)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=device).view(1, -1, 1, 1)    # (1, C, 1, 1)

    return tensor * std + mean

if __name__ == "__main__":
    initialize_dataset()
    #denormalize()