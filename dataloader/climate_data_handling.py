from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import torchvision.transforms as transforms
from os import getcwd
import torch

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

    def prepare_file_path(self, month, component):
        file_path = f"{self.root}/{component}/GWRwSPEC"
        # for years starting at 2017, add the extension .HEI
        if(int(month) > 201700): file_path =  file_path + ".HEI"

        file_path = file_path +  f"_{component}_NA_{month}_{month}"
        if(component=="PM25"): file_path = file_path + "-RH35"
        
        file_path = file_path + ".nc"
        return file_path

    def __getitem__(self, idx):
        month = self.months[idx]
        arr= []
        for component in self.components:
            file_path = self.prepare_file_path(month, component)
            ds = xr.open_dataset(file_path, engine="h5netcdf")
            item_values = ds[component].values
            tensor_values = torch.from_numpy(item_values)
            arr.append(tensor_values)
        
        tensor = torch.stack(arr, dim=0)
        if self.transformations:
            tensor = self.transformations(tensor)

        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)

        return tensor

def initialize_data_loader(components, years, batch_size, shuffle, img_size):
        # Load your custom dataset
        root = "./data/climate-monthly/netcdf"
        transformations = transforms.Resize(img_size)
        dataset = ClimateDataset(root, components, years, transformations=transformations)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader

if __name__ == "__main__":
    root = "./data/climate-monthly/netcdf"
    components = ["PM25", "BC"]
    years = [2013, 2014, 2015, 2016]
    transformations = transforms.Resize((128, 256))
    dataset = ClimateDataset(root, components, years, transformations=transformations)
    loader = DataLoader(dataset, batch_size=3, shuffle=True)

    for i, data in enumerate(loader):
        print(data.shape)
        if i == 5:
            break



