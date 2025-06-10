import matplotlib.pyplot as plt
import numpy as np
from treeffuser import Treeffuser, Samples
import pickle

from dataloader.vanilla_simue_ds import HistoryDataset

def test_on_covariates(model, seed):
    x = np.array([[-0.55351447, -0.4848565, 1.16419383,  0.49827393, 0. , 1. , 1. , 0. ],
                [ 0.28977297, -1.3809542,   3.01439762, -1.90573333, 0. , 1. , 1. , 0. ],
                [-0.19696685,  0.85740084, -0.886278, 0.76179763, 0. , 1. , 0. , 1.],
                [-2.42675319,  2.06844515, -1.3104208, 1.04826, 0. , 1. , 0. , 1.],
                ])
    
    y_samples = model.sample(x, n_samples=50, seed=seed, verbose=True)

    for i in range(4):
        y_mean = y_samples[:, i].mean(axis=0)  # conditional mean
        y_std = y_samples[:, i].std(axis=0)    # conditional std
        print(f"Sampled mean: {y_mean}, Sampled std: {y_std}")

def save_model(model):
    with open('models/treeffuser_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved to 'models/treeffuser_model.pkl'")

def main():
   
    # Generate data
    seed = 0
    dependency_length = 3
    intervention = [0, 0, 0, 0]

    prev_model = Treeffuser(seed=seed)
    ds = HistoryDataset(folder='data/sim-data', fname='horizon_3.npz', horizon=0)
    X, y = ds.get_training_data(num_samples=100000)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print("Training the model...")
    prev_model.fit(X, y)
    print("Training completed")


    for i in range(1, dependency_length+1):
        ds = HistoryDataset(folder='data/sim-data', fname=f'horizon_3.npz', horizon=i)

   


    # Generate and plot samples

if __name__ == "__main__":
    main()
