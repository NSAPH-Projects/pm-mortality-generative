
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
from tqdm import tqdm
from torch.nn import Parameter
import scipy.io as sio
import scipy.stats as ss
import itertools
from torch import nn
import torch.nn.functional as F
from sklearn.neighbors import KernelDensity
import time
import matplotlib as mpl
from matplotlib import colors
import os
import hydra
from omegaconf import DictConfig

plt.rc('text', usetex=False)

font = {
    'family' : 'serif',
    'weight' : 'normal',
    'size'   : 16}
plt.rc('font', **font)
mpl.rcParams['axes.linewidth'] = 2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def plot_trajectories(simu_data, num_individuals=3, num_time_points=30):

    fig,ax = plt.subplots(1,4, figsize=(12,2.5))
    labels = ['Y','X','A','f']
    # First 30 time points of the first 3 individuals
    # From the historical data, just take the last time point
    for i in range(4):
        if i==0:
            ax[i].plot((simu_data[i][: num_individuals, : num_time_points]).T)
        else:
            ax[i].plot((simu_data[i][ : num_individuals, : num_time_points]).T)
        ax[i].set_ylabel(labels[i])
        ax[i].set_xlabel('Time')
        #remove the top and right borders
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
    
    plt.subplots_adjust(wspace=0.5)
    plt.show()

def plot_marginal_distributions(simu_data):
    '''
    Plot simulated data
    '''
    fig,ax = plt.subplots(1,4, figsize=(12,2.5))
    labels = ['Y','X','A','f']

    # Plot the histogram of the data
    fig,ax = plt.subplots(1,4, figsize=(12,2.5))
    for i in range(4):
        ax[i].hist(simu_data[i].flatten(),bins=100)
        ax[i].set_xlabel(labels[i])
        ax[i].set_ylabel('count')
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
    plt.subplots_adjust(wspace=0.5)
    plt.show()

# Disregard the first d time points and return the history added data
def add_history(simu_data, d):
    Y,L,A,f = simu_data
    S,T = Y.shape

    L_,f_,A_ = np.zeros([S,T-d+1,d]),np.zeros([S,T-d+1,d]),np.zeros([S,T-d+1,d])
    for kk in range(T-d+1):
        L_[:,kk] = L[:,kk:kk+d]
        f_[:,kk] = f[:,kk:kk+d]
        A_[:,kk] = A[:,kk:kk+d]
    
    return (Y[:,d-1:],L_,A_,f_)

def plot_intervention_conditional_distributions(outcomes_under_intervention, intervention, d):
    num_cov = outcomes_under_intervention.shape[0]
    fig, axes = plt.subplots(1, num_cov, figsize=(12, 2.5))
    # Ensure axes is iterable if only one subplot is created
    if num_cov == 1:
        axes = [axes]
    for cnt, ax in enumerate(axes):
        ax.hist(
            outcomes_under_intervention[cnt],
            alpha=0.8,
            bins=50,
            label='true dist',
            density=True
        )
        ax.set_title(
            f'cov number {cnt}, A={intervention}\n'
            f'mean = {outcomes_under_intervention[cnt].mean():.2f}'
        )
        ax.set_xlabel('Y')
        ax.set_ylabel('Density')
        ax.legend()
    plt.subplots_adjust(wspace=0.5)
    plt.show()

# I don't know why this function is useful
def plot_distributions_of_unique_treatments(simu_data_with_history, d):
    Y,L,A,f = simu_data_with_history

    #flatten the data to one dimension row-major to create the histogram
    Y = Y.reshape(-1, order='C')
    L = L.reshape(-1, d, order='C')
    A = A.reshape(-1, d, order='C')
    f = f.reshape(-1, d, order='C')

    # Plot the distribution of Y for all different treatment combinations
    A_unique = np.unique(A,axis=0)
    Y_trues = []     
    for a in A_unique:
        inds  = np.where(np.all(A==a,axis=1)) 
        Y_trues.append(Y[inds])

    for cnt in range(len(A_unique)):
        plt.hist(Y_trues[cnt],alpha = 0.8,bins=50, label='true dist',density=True)

        plt.title('A=' + str(A_unique[cnt]) + '\n proportion = %.2f'%(len(Y_trues[cnt])/len(Y)) + '\n mean = %.2f'%(Y_trues[cnt].mean()))

        plt.xlabel('Y')
        plt.ylabel('count')

        plt.show()

def training_data_simulator(
    d: int,
    alphas: np.ndarray,
    betas: np.ndarray,
    gammas: np.ndarray,
    S: int,
    T: int,
    buffer: int
) -> np.ndarray:
    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    Y = np.zeros((S, T - d))
    L = np.zeros((S, T))
    A = np.zeros((S, T))
    f = np.zeros((S, T))
    
    # t = 0
    L[:, 0] = np.random.uniform(size=S)
    f[:, 0] = sigmoid(betas[0] + betas[1] * L[:, 0])
    A[:, 0] = np.random.binomial(1, f[:, 0])
    
    # warm-up
    for t in range(1, d):
        A_hist = A[:, :t]     # (S, t)
        L_hist = L[:, :t]     # (S, t)
        
        L[:, t] = (
            gammas[0]
            + np.dot(A_hist, gammas[1:1+t])
            + np.dot(L_hist, gammas[1+d:1+d+t])
        )
        
        L_full = L[:, :t+1]   # (S, t+1)
        f[:, t] = sigmoid(
            betas[0]
            + np.dot(A_hist, betas[1:1+t])
            + np.dot(L_full, betas[1+d:2+d+t])
        )
        A[:, t] = np.random.binomial(1, f[:, t])
    
    # main sim
    for t in range(d, T):
        A_win = A[:, t-d:t]      # (S, d)
        L_win = L[:, t-d:t]      # (S, d)
        
        L[:, t] = (
            gammas[0]
            + np.dot(A_win, gammas[1:1+d])
            + np.dot(L_win, gammas[1+d:])
        )
        
        L_win_full = L[:, t-d:t+1]  # (S, d+1)
        f[:, t] = sigmoid(
            betas[0]
            + np.dot(A_win, betas[1:1+d])
            + np.dot(L_win_full, betas[1+d:])
        )
        A[:, t] = np.random.binomial(1, f[:, t])
        
        Y[:, t-d] = (
            alphas[0]
            + np.dot(A_win, alphas[1:1+d])
            + np.dot(L_win, alphas[1+d:])
        )
    
    # slice off buffer
    start = buffer
    end = T - d
    Y_slice = Y[:, start:end]
    L_slice = L[:, d+start:d+end]
    A_slice = A[:, d+start:d+end]
    f_slice = f[:, d+start:d+end]
    
    return np.stack([Y_slice, L_slice, A_slice, f_slice], axis=0)

#TODO implement this function
def conditional_data(
    d: int,
    initial_sim_data: np.ndarray,
    idx: np.ndarray,
    intervention: np.ndarray,
    alphas: np.ndarray,
    gammas: np.ndarray,
    S: int,
    T: int,
    n_test: int = 5
):
    subset = initial_sim_data[:, idx, :]  # (4, n_test, d)

    L = np.zeros((n_test, S, 2*d+1))
    A = np.zeros((n_test, S, 2*d))
    f = np.zeros((n_test, S, 2*d))
    Y = np.zeros((n_test, S, 2*d+1))

    # broadcast history
    for arr, vals in zip((Y, L, A, f), subset):
        arr[:, :, :d] = vals[:, None, :]

    # intervention
    A[:, :, d:2*d] = intervention[None, None, :]

    for t in range(d, 2*d+1):
        A_win = A[:, :, t-d:t]   # (n_test, S, d)
        L_win = L[:, :, t-d:t]   # (n_test, S, d)

        # np.dot with 3D X and 1D w => dots over last axis
        L[:, :, t] = (
            gammas[0]
            + np.dot(A_win, gammas[1:1+d])
            + np.dot(L_win, gammas[1+d:])
        )

        Y[:, :, t] = (
            alphas[0]
            + np.dot(A_win, alphas[1:1+d])
            + np.dot(L_win, alphas[1+d:])
        )

    return subset[:, :, :d], Y[:, :, -1]


def conditional_intervened_data(
    d: int,
    initial_sim_data: np.ndarray,
    idx: np.ndarray,
    intervention: np.ndarray,
    alphas: np.ndarray,
    gammas: np.ndarray,
    S: int,
):
    subset = initial_sim_data[:, idx, :]  # (4, n_test, num_time_steps)
    n_test = idx.shape[0]

    L = np.zeros((n_test, S, 2*d+1))
    A = np.zeros((n_test, S, 2*d))
    f = np.zeros((n_test, S, 2*d))
    Y = np.zeros((n_test, S, 2*d+1))

    # broadcast history
    for arr, vals in zip((Y, L, A, f), subset):
        arr[:, :, :d] = vals[:, None, :d]

    # intervention
    A[:, :, d:2*d] = intervention[None, None, :]

    for t in range(d, 2*d+1):
        A_win = A[:, :, t-d:t]   # (n_test, S, d)
        L_win = L[:, :, t-d:t]   # (n_test, S, d)

        # np.dot with 3D X and 1D w => dots over last axis
        L[:, :, t] = (
            gammas[0]
            + np.dot(A_win, gammas[1:1+d])
            + np.dot(L_win, gammas[1+d:])
        )

        Y[:, :, t] = (
            alphas[0]
            + np.dot(A_win, alphas[1:1+d])
            + np.dot(L_win, alphas[1+d:])
        )

    return subset[:, :, :d], Y[:, :, -1]

def get_coefficients(d):

    if d==1:
        ycoeff = [-3,2,-1] # Y
        acoeff = [-0.5,0.5,-0.5,0.5] # A
        xcoeff = [0,1,-1] # X
    elif d==3:
        ycoeff=[-1] + [3,6,12] + [0.5,1,2]# Y
        acoeff= [-0.5] + [0.5,-0.5,0.5] + [-0.5,0.5,-0.5,0.5]# A
        xcoeff= [-1] + [0.5,1,1.5] + [-0.5,-1,-1.5]# X
    else:
        ycoeff=[-1] + [0.5,1,3,6,12] + [0.05,0.1, 0.5,1,2]# Y
        acoeff= [-0.5] + [0.5,-0.5,0.5,-0.5,0.5] + [-0.5,0.5,-0.5,0.5,-0.5,0.5]# A
        xcoeff= [-1] + [0.05,0.1, 0.5,1,1.5] + [-0.05,-0.1, -0.5,-1,-1.5]  # X
    
    return ycoeff, acoeff, xcoeff



@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Create the data
    print('Creating data...')
    num_time_steps = cfg.data.train_size + cfg.data.buffer + 2 * cfg.data.dep_len
    alphas, betas, gammas = get_coefficients(cfg.data.dep_len)
    data = training_data_simulator(cfg.data.dep_len, alphas, betas, gammas, cfg.data.pop_size , num_time_steps, cfg.data.buffer)
    print('Simulated data shape:', data.shape)
    #plot_marginal_distributions(data)

    interventions = np.array([[0, 0, 0], [1,1,1]])
    assert interventions.shape[1] == cfg.data.dep_len, "Intervention length must be equal to dep_len"
    
    cov_idx = np.random.choice(cfg.data.pop_size, cfg.data.num_test_cov, replace=False)

    covs, outcomes = conditional_intervened_data(cfg.data.dep_len, data, cov_idx, interventions[0], alphas, gammas, cfg.data.pop_size)
    plot_intervention_conditional_distributions(outcomes, interventions[0], cfg.data.dep_len)

if __name__ == "__main__":
    seed = 42
    if seed >=0:
        np.random.seed(seed)
    main()