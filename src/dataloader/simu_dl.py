from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Union

class HistoryDataset(Dataset):
    """
    A PyTorch Dataset that provides history of covariates (X) and actions (A)
    based on a specified horizon from precomputed .npz data.
    Each item returns:
      - horizon = 0: ((h_tm1_A, h_tm1_X), a_tm, y_t)
      - horizon > 0: ((h_tm1_A, h_tm1_X), a_tm1, (h_t_A, h_t_X))
    """
    def __init__(
        self,
        folder: Union[str, Path],
        fname: str,
        horizon: int = 0,
    ) -> None:
        # Load .npz file
        path = Path(folder) / fname
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        data = np.load(path)

        # Convert to torch tensors
        self.H = torch.from_numpy(data["H"]).float()     # shape: (pop, T, d+1, 2)
        self.Y = torch.from_numpy(data["Y"][..., 0]).float()  # shape: (pop, T)

        pop, T, d1, two = self.H.shape
        assert two == 2, "Expected last dim of H to be (X,A)"

        self.pop_size = pop
        self.num_time_steps = T
        self.d = d1 - 1
        if not (0 <= horizon <= self.d):
            raise ValueError(f"horizon must be in [0, {self.d}], got {horizon}")
        self.horizon = horizon

        self.mean, self.std = self.get_mean_std()
        self.standardize()

    def __len__(self) -> int:
        return self.pop_size * self.num_time_steps
    
    def get_mean_std(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the mean and standard deviation of the covariates (X), treatments (A), and outcomes (Y).
        """
        X = self.H[:, :, self.d, 0].flatten()
        A = self.H[:, :, self.d, 1].flatten()
        Y = self.Y
        return (X.mean(), A.mean(), Y.mean()), (X.std(), A.std(), Y.std())
    
    def standardize(self):
        """
        Standardizes the covariates (X) and treatments (A) in-place.
        """
        self.H[:, :, :, 0] = (self.H[:, :, :, 0] - self.mean[0]) / self.std[0]
        self.H[:, :, :, 1] = (self.H[:, :, :, 1] - self.mean[1]) / self.std[1]
        self.Y = (self.Y - self.mean[2]) / self.std[2]

    def unstandardize(self, y: torch.Tensor) -> torch.Tensor:
        """
        Unstandardizes the outcome tensor y.
        """
        return y * self.std[2] + self.mean[2]

    def __getitem__(self, idx: int):
        # map flat idx to (individual i, time t)
        i, t = divmod(idx, self.num_time_steps)
        H_it = self.H[i, t]  # shape: (d+1, 2)
        y_t = self.Y[i, t]

        hz = self.horizon
        d = self.d

        if hz == 0:
            return (H_it[:d, 1], H_it[:d+1, 0]), H_it[d, 1], y_t

        indx = d - hz

        # previous history
        h_tm1_A = H_it[: indx, 1]
        h_tm1_X = H_it[: indx + 1, 0]

        a_tm1 = H_it[indx, 1]  # action at t-1

        # current history
        h_t_A = H_it[: indx + 1, 1]
        h_t_X = H_it[: indx + 2, 0]
        return (h_tm1_A, h_tm1_X), a_tm1, (h_t_A, h_t_X)