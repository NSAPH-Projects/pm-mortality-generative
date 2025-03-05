import json
from itertools import product
import time
from pathlib import Path


import hydra
import numpy as np
import torch
import torchvision.transforms as transforms
import xarray as xr
from torchvision.transforms.functional import InterpolationMode
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
        
        #tensor = torch.nan_to_num(tensor, nan=0.0)

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
            transforms.Resize(grid_size, interpolation=InterpolationMode.NEAREST),
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

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """This script tests the dataloader and saves aggregate statistics in a summary file."""

    transform = transforms.Resize(cfg.grid_size)

    dataset = ComponentsWashuDataset(
        root_dir=cfg.root_dir,
        transform=transform,
        components=cfg.components,
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # compute the means and standard deviations without storing all the data
    # do not sum nans
    totals_sum = torch.zeros(len(cfg.components))
    totals_ss = torch.zeros(len(cfg.components))
    totals_n = torch.zeros(len(cfg.components))

    # also keep track of time
    start_time = time.time()

    input_size = None
    for batch in tqdm(loader):
        if input_size is None:
            input_size = batch.shape[2:]
        totals_n += (~torch.isnan(batch)).sum(dim=(0, 2, 3))
        x = torch.nan_to_num(batch, nan=0.0)
        totals_sum += x.sum(dim=(0, 2, 3))
        totals_ss += (x**2).sum(dim=(0, 2, 3))

    elapsed_time = time.time() - start_time

    # compute means and stds
    means = totals_sum / totals_n
    stds = torch.sqrt(totals_ss / totals_n - means**2)

    # conver to dict with components as keys
    means_dict = {component: float(mean) for component, mean in zip(cfg.components, means)}
    stds_dict = {component: float(std) for component, std in zip(cfg.components, stds)}

    # save to file
    summary = {"means": means_dict, "stds": stds_dict, "elapsed_time": elapsed_time, "input_grid_size": input_size}

    summary_file = f"{cfg.root_dir}/summary.json"
    Path(summary_file).touch(exist_ok=True)

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
    #denormalize()