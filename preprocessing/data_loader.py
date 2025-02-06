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
            file_path = f"{root}/{component}/GWRwSPEC_{component}_NA_{month}_{month}"
            if(component=="PM25"): file_path = file_path + "-RH35"
            file_path = file_path + ".nc"
            print(file_path)
            ds = xr.open_dataset(file_path, engine="h5netcdf")
            item_values = ds[component].values
            arr.append(item_values)
        
        return np.stack(arr, axis=0)

if __name__ == "__main__":
    root = "./data/climate-monthly/netcdf"
    components = ["PM25", "BC"]
    years = [2016, 2017, 2018]
    dataset = ClimateDataset(root, components, years)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for i, data in enumerate(loader):
        print(data.shape)
        if i == 5:
            break



