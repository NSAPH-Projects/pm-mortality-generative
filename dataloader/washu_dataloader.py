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
        mean_dict, std_dict = get_mean_and_std(root_dir)
        self.means = [mean_dict[component] for component in components]
        self.stds = [std_dict[component] for component in components]

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
    
    def denormalize(self, tensor):
        device = tensor.device  # Use the same device as the input tensor
        
        mean = torch.as_tensor(self.means, dtype=tensor.dtype, device=device).view(-1, 1, 1)  # (1, C, 1, 1)
        std = torch.as_tensor(self.stds, dtype=tensor.dtype, device=device).view(-1, 1, 1)    # (1, C, 1, 1)
        
        #if we want to denormalize a batch of images
        if(tensor.dim() == 4):
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)

        return tensor * std + mean

def initialize_dataset(root_dir, grid_size, components):

    mean_dict, std_dict = get_mean_and_std(root_dir)
    means = [mean_dict[component] for component in components]
    stds = [std_dict[component] for component in components]
    
    transform = transforms.Resize(grid_size)
    transform = transforms.Compose(
        [
            transforms.Resize(grid_size),
            transforms.Normalize(mean=means, std=stds),
        ]
    )

    dataset = ComponentsWashuDataset(
        root_dir=root_dir,
        transform=transform,
        components=components,
    )

    return dataset
 
def get_mean_and_std(root_dir):
    with open(f"{root_dir}/summary.json", "r") as f:
        summary = json.load(f)
    return summary["means"], summary["stds"]

if __name__ == "__main__":
    initialize_dataset()
    #denormalize()