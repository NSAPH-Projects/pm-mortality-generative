from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import torchvision.transforms as transforms
from os import getcwd
import torch
from tqdm import tqdm
import json

class ClimateDataset(Dataset):
    def __init__(self, root, components, years, as_numpy=False, transformations=None):
        self.root = root
        self.components = components
        self.years = years
        self.as_numpy = as_numpy
        self.transformations = transformations

        # make a list of all months for the given years
        self.months = []
        for year in years:
            for month in range(1, 13):
                self.months.append(f"{year}{month:02d}")
    
    def __len__(self):
        return len(self.months)

    def __getitem__(self, idx):
        month = self.months[idx]
        arr= []
        for component in self.components:
            file_path = prepare_file_path(self.root, month, component)
            ds = xr.open_dataset(file_path, engine="h5netcdf")
            item_values = ds[component].values
            tensor_values = torch.from_numpy(item_values)
            arr.append(tensor_values)
        
        tensor = torch.stack(arr, dim=0)
        mask = ~torch.isnan(tensor)
        # it is imporant to convert nan values to 0 before applying the normalization
        # otherwise zero padding would have greater value than some data points
        # also the downscaling is smoother this way. bc otherwise nanvalues propogate
        tensor = torch.where(mask, tensor, torch.zeros_like(tensor))
        
        if self.transformations:
            tensor = self.transformations(tensor)
        
        resize = get_resize(self.transformations)
        if resize:
            mask = resize(mask)
            mask = (mask > 0.5)

        return tensor, mask
    
def get_resize(transformation):
    """Returns the transforms.Resize transformation if present, otherwise None."""
    if isinstance(transformation, transforms.Compose):
        for t in transformation.transforms:
            if isinstance(t, transforms.Resize):
                return t
    return None

def prepare_file_path(root, month, component):
        file_path = f"{root}/{component}/GWRwSPEC"
        # for years starting at 2017, add the extension .HEI
        if(int(month) > 201700): file_path =  file_path + ".HEI"

        file_path = file_path +  f"_{component}_NA_{month}_{month}"
        if(component=="PM25"): file_path = file_path + "-RH35"
        
        file_path = file_path + ".nc"
        return file_path

def find_stats_gpu(root, components):
    root = "./data/climate-monthly/netcdf"
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    # to check if the normalization is correct
    #normy = transforms.Normalize(mean=[4.889685, 0.3213127], std=[4.6892176, 0.323582213])
    
    dataset = ClimateDataset(root=root, components=components, years=list(range(2000, 2018)), transformations=None)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=False)

    stats = {component: {
                'min': np.inf,
                'max': -np.inf,
                'mean': 0.0,
                'm2': 0.0,
                'count': 0
             } for component in components}

    for batch, mask in tqdm(dataloader, desc='Batches', leave=False):
        batch = batch.to(device)
        mask = mask.to(device)

        for idx, component in enumerate(components):
            component_data = batch[:, idx, ...]  # Ensure double precision
            component_mask = mask[:, idx, ...]

            # Apply mask: consider only elements where mask is True (originally non-NaN)
            valid_data = component_data[component_mask]

            if valid_data.numel() == 0:
                continue  # No valid data to process

            # Min and Max
            b_min = valid_data.min().item()
            b_max = valid_data.max().item()
            
            if b_min < stats[component]['min']:
                stats[component]['min'] = b_min

            if b_max > stats[component]['max']:
                stats[component]['max'] = b_max

            # Mean and Variance (Welford's method) - https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
            # allows for calculation of variance in a single pass and using gpu
            new_count = valid_data.numel()
            new_mean = valid_data.mean().item()
            new_m2 = ((valid_data - new_mean) ** 2).sum().item()

            delta = new_mean - stats[component]['mean']
            total_count = stats[component]['count'] + new_count

            # Update mean
            stats[component]['mean'] += delta * new_count / total_count
            stats[component]['m2'] += new_m2 + delta**2 * stats[component]['count'] * new_count / total_count

            # Update count
            stats[component]['count'] = total_count

    # Calculate final statistics
    for component, values in stats.items():
        mean = values['mean']
        # Calculate sample variance
        variance = values['m2'] / (values['count'] - 1) if values['count'] > 1 else 0.0 
        std = np.sqrt(variance)

        print(f"Component: {component}, Min: {values['min']}, Max: {values['max']}, Mean: {mean}, Std: {std}")

#Component: PM25, Min: 0.0, Max: 485.6000061035156, Mean: 4.8896847201718225, Std: 4.68921759434198                                                                 
#Component: BC, Min: 0.0, Max: 17.700000762939453, Mean: 0.321312690636626, Std: 0.32358221282327493

def denormalize(tensor, mean=[4.889685, 0.3213127], std=[4.6892176, 0.323582213]):
    """
    Denormalizes a tensor by reversing the normalization process.

    Args:
        tensor (torch.Tensor): Normalized tensor of shape (B, C, H, W).
        mean (list or torch.Tensor): Mean values for each channel.
        std (list or torch.Tensor): Standard deviation values for each channel.

    Returns:
        torch.Tensor: Denormalized tensor.
    """
    device = tensor.device  # Use the same device as the input tensor

    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=device).view(1, -1, 1, 1)  # (1, C, 1, 1)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=device).view(1, -1, 1, 1)    # (1, C, 1, 1)

    return tensor * std + mean

def initialize_data_loader(components, batch_size, shuffle, img_size):
        # Load your custom dataset
        root = "./data/climate-monthly/netcdf"

        transformations = transforms.Compose([
            transforms.Normalize(mean=[4.889685, 0.3213127], std=[4.6892176, 0.323582213]),
            transforms.Resize(img_size)
        ])
        dataset = ClimateDataset(root, components, years = list(range(2000, 2018)), transformations=transformations)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader

def print_first_elements (root, components, years) :
    transformations = transforms.Resize((128, 256))
    dataset = ClimateDataset(root, components, years, transformations=transformations)
    loader = DataLoader(dataset, batch_size=3, shuffle=True)
    for i, data in enumerate(loader):
        print(data.shape)
        if i == 5:
            break


if __name__ == "__main__":
    root = "./data/climate-monthly/netcdf"
    components = ["PM25", "BC"]
    years = [2013, 2014, 2015, 2016]

    find_stats_gpu(root, components)