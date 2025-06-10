from pathlib import Path
import numpy as np
from typing import Tuple, Union

class HistoryDataset:
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
        self.H = data["H"]    # shape: (pop, T, d+1, 2)
        self.Y = data["Y"][..., 0]  # shape: (pop, T)

        pop, T, d1, two = self.H.shape
        assert two == 2, "Expected last dim of H to be (X,A)"

        self.pop_size = pop
        self.num_time_steps = T
        self.d = d1 - 1
        if not (0 <= horizon <= self.d):
            raise ValueError(f"horizon must be in [0, {self.d}], got {horizon}")
        self.horizon = horizon
    
    def get_training_data(self, num_samples: int) -> np.ndarray:
        """
        Returns a random sample of training data.
        """

        flat_H = self.H.reshape(-1, self.H.shape[2], self.H.shape[3]) 
        indices = np.random.choice(len(self), num_samples, replace=False)
        H_train = flat_H[indices, :, :]  # shape: (num_samples, d+1, 2)

        X = np.concatenate((H_train[:, :, 0], H_train[:, :, 1]), axis=1)        
        y = self.Y.flatten()[indices]

        return X, y

    def __len__(self) -> int:
        return self.pop_size * self.num_time_steps