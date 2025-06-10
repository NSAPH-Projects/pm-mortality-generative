'''
Adapted from https://github.com/ShenghaoWu/Counterfactual-Generative-Models

'''

import itertools
import sys
from typing import Dict, Tuple
import argparse
import scipy as sci
import scipy.io as sio
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import logging
import time
import matplotlib as mpl
from matplotlib import colors
import os
from scipy.special import softmax
import pickle
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KernelDensity
import numpy as np
import math
from dataloader.simu_dl import HistoryDataset
from torch.utils.data import DataLoader

#Schedules
def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim, double_layers=False):
        super(EmbedFC, self).__init__()

        self.input_dim = input_dim
        
        if double_layers:
            layers = [
                nn.Linear(input_dim, emb_dim * 16),
                nn.GELU(),
                nn.Linear(emb_dim * 16, emb_dim * 8 ),
                nn.GELU(),
                nn.Linear(emb_dim * 8, emb_dim * 4 ),
                nn.GELU(),
                nn.Linear(emb_dim * 4, emb_dim * 2 ),
                nn.GELU(),
                nn.Linear(emb_dim * 2, emb_dim ),
            ]

        else:
            layers = [
            nn.Linear(input_dim, emb_dim * 4),
            nn.GELU(),
            nn.Linear(emb_dim * 4, emb_dim * 2 ),
            nn.GELU(),
            nn.Linear(emb_dim * 2, emb_dim ),
            ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        # embedding values need to be small
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        
        #print(f"embeddings shape: {embeddings.shape}, time shape: {time.shape}")
        embeddings = embeddings * time[:, None]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ContextAE(nn.Module):
    def __init__(self, cov_dimension, hidden_dim = 64 ):
        super(ContextAE, self).__init__()

        self.hidden_dim = hidden_dim
        self.cov_dimension = cov_dimension

        self.time_pe = SinusoidalPositionEmbeddings(hidden_dim)  # fixed embeddings
        self.encode_out = EmbedFC(1, hidden_dim // 4)
        self.encode_treat = EmbedFC(1, hidden_dim // 4)
        self.encode_cov = EmbedFC(cov_dimension, hidden_dim // 2 , double_layers=True) 
        
        self.mid_layers =  nn.ModuleList([nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                     nn.Linear(hidden_dim, hidden_dim )
                                      ])
        
        self.encode_time = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                     nn.Linear(hidden_dim, hidden_dim)
                                      ])
        
        self.decode = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                      nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
                                      nn.Linear(hidden_dim * 2, hidden_dim * 4),nn.GELU(),
                                      nn.Linear(hidden_dim * 4 , hidden_dim * 8), nn.GELU(),
                                      nn.Linear(hidden_dim * 8, hidden_dim * 16), nn.GELU(),
                                      nn.Linear(hidden_dim * 16 , 1 )
                                      ])

    def forward(self, x, cov, treat, cov_mask, treat_mask, t):

        # x is (noisy) data, c is covariate, t is timestep, 
        # context_mask says which samples to block the context on
        #print(f"Input shapes - x: {x.shape}")
        x = self.encode_out(x)

        #print(f"Input shapes - x: {x.shape}")

        treat = self.encode_treat(treat)

        cov = self.encode_cov(cov)

        # if the mask is 0, make it 
        cov_mask = 1 - cov_mask # need to flip 0 <-> 1
        treat_mask = 1 - treat_mask # need to flip 0 <-> 1
        
        cov = cov * cov_mask  # apply mask to covariates
        treat = treat * treat_mask # apply mask to treatment

        #print time shape 
        #print(f"time shape: {t.shape}")
        t = self.time_pe(t)  # positional encoding for time
        #for layer in self.encode_time:
        #    t = layer(t)
        
        #print(f"time embedding shape: {t.shape}")

        # print shapes for debugging
        #print(f"NNMODEL x shape: {x.shape}, treat shape: {treat.shape}, cov shape: {cov.shape}, t shape: {t.shape}")

        x = torch.cat((x, treat, cov), 1)

        for layer in self.mid_layers:
            x = layer(x)

        x = x.add(t)  # add time embedding

        for layer in self.decode:
            x = layer(x)
        return x

class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, cov, treat):
        """
        this method is used in training, so samples t and noise randomly
        """
        batch_size = x.shape[0]
        _ts = torch.randint(1, self.n_T+1, (batch_size,)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None] * x
            + self.sqrtmab[_ts, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        p_tensor = torch.full((batch_size, 1), self.drop_prob, device=self.device)
        cov_mask = torch.bernoulli(p_tensor)  # shape = [batch_size, 1], values are 0/1
        treat_mask = torch.bernoulli(p_tensor)  # shape = [batch_size, 1], values are 0/1

        # return MSE between added noise, and our predicted noise
        pred = self.nn_model(x_t, cov, treat, cov_mask, treat_mask,  _ts / self.n_T)

        return self.loss_mse(noise, pred)

    def sample(self, cov: torch.Tensor, treat: torch.Tensor, device, guide_w: float = 0.0):
        """
        Sample once per condition, assuming a 1-D output (feature_dim=1).

        Args:
            cov (torch.Tensor): batch of context/covariate vectors, shape (batch_size, context_dim).
            treat (torch.Tensor): batch of treatment indicators or features, shape (batch_size, treat_dim).
            device (str or torch.device).
            guide_w (float): classifier-free guidance weight.

        Returns:
            torch.Tensor: sampled outputs of shape (batch_size, 1).
        """
        batch_size = cov.shape[0]

        # Start from x_T ~ N(0, 1), shape (batch_size, 1)
        x_i = torch.randn(batch_size, 1, device=device)

        # Move cov and treat to device
        cov = cov.to(device)
        treat = treat.to(device)

        # Create masks for cov and treat (zeros = use true value, ones = mask/unconditional)
        cov_mask = torch.zeros((batch_size, 1), device=self.device)
        treat_mask = torch.zeros((batch_size, 1), device=self.device)

        # Duplicate cov, treat, and their masks along the batch dimension
        cov = cov.repeat(2, 1)
        treat = treat.repeat(2, 1)
        cov_mask = cov_mask.repeat(2, 1)
        treat_mask = treat_mask.repeat(2, 1)

        # In the second half, zero out (i.e. mask) both cov and treat
        cov_mask[batch_size:] = 1.0
        treat_mask[batch_size:] = 1.0

        #print(f"cov shape: {cov.shape}, treat shape: {treat.shape}, cov_mask shape: {cov_mask.shape}, treat_mask shape: {treat_mask.shape}")

        for t in range(self.n_T, 0, -1):
            # Build timestep tensor: shape (batch_size, 1), then duplicate
            t_lin = torch.full((batch_size, ), float(t) / self.n_T, device=device)
            t_is = t_lin.repeat(2)

            # Because we doubled cov and treat, replicate x_i as well
            x_i = x_i.repeat(2, 1)

            # Sample noise z_t for next step (zero at t=1)
            if t > 1:
                z = torch.randn(batch_size, 1, device=device)
            else:
                z = torch.zeros(batch_size, 1, device=device)

            #print shapes of xi cov treat cov_mask treat_mask t_is
            #print(f"t: {t}, x_i shape: {x_i.shape}, cov shape: {cov.shape}, treat shape: {treat.shape}, cov_mask shape: {cov_mask.shape}, treat_mask shape: {treat_mask.shape}, t_is shape: {t_is.shape}")
            
            # Predict εθ for both “with-cov/treat” and “without-cov/treat” (masked)
            eps = self.nn_model(x_i, cov, treat, cov_mask, treat_mask, t_is)
            #print(f"t: {t}, eps shape: {eps.shape}")
            eps1 = eps[:batch_size]       # predictions when cov_mask & treat_mask = 0
            eps2 = eps[batch_size:]       # predictions when cov_mask & treat_mask = 1

            #print(f"t: {t}, eps1 shape: {eps1.shape}, eps2 shape: {eps2.shape}")
            # Combine via classifier-free guidance
            eps_combined = (1 + guide_w) * eps1 - guide_w * eps2

            #print(f"t: {t}, eps_combined shape: {eps_combined.shape}")

            # Keep only the first half of x_i to update
            x_prev = x_i[:batch_size]

            # Compute x_{t-1} = 1/√α_t * (x_t – ((1−α_t)/√(1−ᾱ_t)) * eps_combined) + √β_t * z
            coef1 = self.oneover_sqrta[t]
            coef2 = self.mab_over_sqrtmab[t]
            coef3 = self.sqrt_beta_t[t]

            #print(f"t: {t}, coef1: {coef1.shape}, coef2: {coef2.shape}, coef3: {coef3.shape}, eps_combined shape: {eps_combined.shape}, z shape: {z.shape} x_prev shape: {x_prev.shape}")
            x_i = coef1 * (x_prev - coef2 * eps_combined) + coef3 * z

        # Final tensor has shape (batch_size, 1)
        return x_i


#train the diffusion model
def train_model(args, n_T, device):
    
    print(f'Training {args.model_name}')
    n_epoch = args.n_epoch
    n_feat = args.nfeat
    batch_size = 256
    lrate = args.lr
    save_model = True
    save_dir = 'models/'
    os.makedirs(save_dir,  exist_ok = True) 

    ds = HistoryDataset(folder='data/sim-data', fname='horizon_3.npz', horizon=0)
    training_batched = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=5)

    # grab a single batch
    batch = next(iter(training_batched))

    (h_A, h_X), a, y = batch
    cov_dim = h_A.shape[1] + h_X.shape[1] 
    nn = ContextAE(cov_dimension = cov_dim, hidden_dim = n_feat)
    ddpm = DDPM(nn_model= nn, betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()
        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)
        pbar = tqdm(training_batched)
        loss_ema = None
        for i, ((h_A, h_X), a, y) in enumerate(pbar):
            optim.zero_grad()
            h_A = h_A.to(device)
            h_X = h_X.to(device)
            a = a.unsqueeze(1).to(device)  # shape (batch_size, 1)
            y = y.unsqueeze(1).to(device)  # shape (batch_size, 1)
            #print(f"Input shapes - h_A: {h_A.shape}, h_X: {h_X.shape}, a: {a.shape}, y: {y.shape}")
            # concatenate the covariates shape (batch_size, cov_dim)
            cov = torch.cat((h_A, h_X), dim=1)  # shape (batch_size, cov_dim)

            loss = ddpm(y, cov, a)  # y is the noisy data, cov is the covariates, a is the treatment
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

            
        print(f"Epoch {ep} finished, loss: {loss_ema:.4f}")


        if save_model:
            torch.save(ddpm.state_dict(), os.path.join(save_dir, f"{args.model_name}.pth"))
            print("saved model at" + os.path.join(save_dir, f"{args.model_name}.pth"))

    return ddpm

def get_model(n_T, n_feat, device):
    ds = HistoryDataset(folder='data/sim-data', fname='horizon_3.npz', horizon=0)
    training_batched = DataLoader(ds, batch_size=2, shuffle=True, num_workers=5)

    # grab a single batch
    batch = next(iter(training_batched))

    (h_A, h_X), a, y = batch
    cov_dim = h_A.shape[1] + h_X.shape[1] 
    nn = ContextAE(cov_dimension = cov_dim, hidden_dim = n_feat)
    ddpm = DDPM(nn_model= nn, betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    return ddpm

def plot_distribution(samples):#, cov, intervention):
    samples = samples.cpu().detach().numpy()
    fig, ax = plt.subplots(1, 1, figsize=(12, 2.5))
    # Ensure axes is iterable if only one subplot is created
    ax.hist(
        samples,
        alpha=0.8,
        bins=50,
        label='true dist',
        density=True
    )
    ax.set_title(
        f'mean = {samples.mean():.2f}'
    )
    ax.set_xlabel('Y')
    ax.set_ylabel('Density')
    ax.legend()
    plt.subplots_adjust(wspace=0.5)
    plt.show()

def normalize(data, ind):
    ds = HistoryDataset(folder='data/sim-data', fname='horizon_3.npz', horizon=0)
    mean, std = ds.mean, ds.std
    # Assuming data is a tensor of shape (batch_size, 1)
    data = (data - mean[ind]) / std[ind]  # normalize the outcome
    #print the mean for all indices

    #print(f"Mean for X index {0}: {mean[0]}, Std for index {0}: {std[0]} ")
    #print(f"Mean for A index {1}: {mean[1]}, Std for index {1}: {std[1]} ")
    #print(f"Mean for Y index {2}: {mean[2]}, Std for index {0}: {std[2]} ")

    return data

def denormalize(samples, ind):
    ds = HistoryDataset(folder='data/sim-data', fname='horizon_3.npz', horizon=0)
    mean, std = ds.mean, ds.std
    # Assuming samples is a tensor of shape (batch_size, 1)
    samples = samples * std[ind] + mean[ind]  # denormalize the outcome
    return samples

def main(args):

    n_T = 400
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    sampling_batch_size = 64

    if args.train:
        model = train_model(args, n_T, device)
    else:
        model = get_model(n_T, args.nfeat, device)
        model.load_state_dict(torch.load(f'models/{args.model_name}.pth'))
        model = model.to(device)

    #read the covariates and treatment from the dataset
    data = np.load('data/sim-data/covs.npz')
    data = data["C"]
    print(data)
    # convert the data to tensor
    data = torch.tensor(data, dtype=torch.float32).to(device)
    cov_number = 1
    
    a = data[2,cov_number,3]
    h_A = data[2,cov_number,:3]
    h_X = data[1,cov_number,:4]

    a = normalize(a, 1)   # normalize the treatment
    h_A = normalize(h_A, 1)
    h_X = normalize(h_X, 0)

    cov = torch.cat((h_A, h_X), axis=0).unsqueeze(0).repeat(sampling_batch_size, 1)
    treat = a.unsqueeze(0).repeat(sampling_batch_size, 1)

    samples = []
    for i in range(2):
        samples += model.sample(cov=cov, treat=treat, device=device, guide_w=5).detach().cpu()

    samples = torch.stack(samples, dim=0)
    denormalized_samples = denormalize(samples, 2)

    plot_distribution(denormalized_samples)

    #take an element from the dataset and check if the predicted value is close to the true value
    ds = HistoryDataset(folder='data/sim-data', fname='horizon_3.npz', horizon=0)
    training_batched = DataLoader(ds, batch_size=1, shuffle=True, num_workers=5)
    batch = next(iter(training_batched))
    (h_A, h_X), a, y = batch
    h_A = h_A.repeat(sampling_batch_size, 1).to(device)  # repeat to match sampling batch size
    h_X = h_X.repeat(sampling_batch_size, 1).to(device)  # repeat to match sampling batch size
    a = a.unsqueeze(1).repeat(sampling_batch_size, 1).to(device)  # repeat to match sampling batch size

    cov = torch.cat((h_A, h_X), dim=1)  # shape (batch_size, cov_dim)
    #print(cov)
    #print(a)
    prediction = model.sample(cov=cov, treat=a, device=device, guide_w=10)
    print(f"True value: {y.item()}, Predicted value: {prediction.mean().item()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Running script arguments"
    )
    parser.add_argument("-n","--model_name",type=str, help="model name", default="diffusion")
    parser.add_argument("-l","--lr", help="learning rate", type=float, default = 1e-5)
    parser.add_argument("-f","--nfeat", help="feature dim", type=int, default = 256)
    parser.add_argument("-e","--n_epoch", help="number of training epochs", type=int, default = 1)
    parser.add_argument("-t","--train", help="train the model", type=bool, default = False)
    parser.add_argument("-ip","--use_iptw", help="if iptw values should be used", type=bool, default = False)

    args = parser.parse_args()
    main(args)