from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np

class ClimateDataset(Dataset):
    def __init__(self, root, components, years):
        self.root = root
        self.components = components
        self.years = years

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
            file_path = f"{root}/{component}/GWRwSPEC_{component}_NA_{month}_{month}.nc" 
            ds = xr.open_dataset(file_path)
            arr.append(ds.values)
        
        return np.stack(arr, axis=0)



if __name__ == "__main__":
    root = "../data/climate-monthly/netcdf"
    components = ["PM25","NH4"]
    years = [2010, 2011, 2012]
    dataset = ClimateDataset(root, components, years)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, data in enumerate(loader):
        print(data.shape)
        if i == 5:
            break



