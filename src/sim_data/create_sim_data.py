
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

def add_history(simu_data, d):
    """
    Returns
    -------
    H : np.ndarray, shape (pop_size, num_time_steps-d, d+1, 2)
    Y : np.ndarray, shape (pop_size, num_time_steps-d, 1)
    """
    Y_raw = simu_data[0]   # shape (pop_size, T)
    X_raw = simu_data[1]   # shape (pop_size, T)
    A_raw = simu_data[2]   # shape (pop_size, T)

    pop_size, T = Y_raw.shape
    H = np.zeros((pop_size, T - d, d + 1, 2))
    Y = np.zeros((pop_size, T - d, 1))

    for t in range(d, T):
        window = slice(t - d, t + 1)
        H[:, t - d, :, 0] = X_raw[:, window]
        H[:, t - d, :, 1] = A_raw[:, window]
        Y[:, t - d, 0]  = Y_raw[:, t]

    return H, Y

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
# DANGER : MIGHT NOT WORK
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

def simulator(
    d: int,
    alphas: np.ndarray,
    betas: np.ndarray,
    gammas: np.ndarray,
    S: int,
    T: int,
    buffer: int,
    var_X: float = 0.1,
    mean_diff_X: float = 1,
    p_X = 0.5,
    model_X: str = "id"
) -> np.ndarray:
    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    Y = np.zeros((S, T - d))
    L = np.zeros((S, T))
    A = np.zeros((S, T))
    A_f = np.zeros((S, T))
    
    # t = 0
    L_f= np.random.uniform(size=S)
    L[:, 0] = np.random.normal(loc=L_f, scale=var_X, size=S)
    A_f[:, 0] = sigmoid(betas[0] + betas[1] * L[:, 0])
    A[:, 0] = np.random.binomial(1, A_f[:, 0])
    
    # warm-up
    for t in range(1, d):
        A_hist = A[:, :t]     # (S, t)
        L_hist = L[:, :t]     # (S, t)
        
        L_f = (
            gammas[0]
            + np.dot(A_hist, gammas[1:1+t])
            + np.dot(L_hist, gammas[1+d:1+d+t])
        )
        if model_X == "id":
            L[:, t] = L_f
        elif model_X == "normal":
            L[:, t] = np.random.normal(loc=L_f, scale=var_X, size=S)
        elif model_X == "mixture":
            upper = np.random.rand(S) < p_X
            locs  = L_f + np.where(upper, +mean_diff_X, -mean_diff_X)
            L[:, t] = np.random.normal(loc=locs, scale=var_X, size=S)
        
        L_full = L[:, :t+1]   # (S, t+1)
        A_f[:, t] = sigmoid(
            betas[0]
            + np.dot(A_hist, betas[1:1+t])
            + np.dot(L_full, betas[1+d:2+d+t])
        )
        A[:, t] = np.random.binomial(1, A_f[:, t])
    
    # main sim
    for t in range(d, T):
        A_win = A[:, t-d:t]      # (S, d)
        L_win = L[:, t-d:t]      # (S, d)
        
        L_f = (
            gammas[0]
            + np.dot(A_win, gammas[1:1+d])
            + np.dot(L_win, gammas[1+d:])
        )
        if model_X == "id":
            L[:, t] = L_f
        elif model_X == "normal":
            L[:, t] = np.random.normal(loc=L_f, scale=var_X, size=S)
        elif model_X == "mixture":
            upper = np.random.rand(S) < p_X
            locs  = L_f + np.where(upper, +mean_diff_X, -mean_diff_X)
            L[:, t] = np.random.normal(loc=locs, scale=var_X, size=S)
        
        L_win_full = L[:, t-d:t+1]  # (S, d+1)
        A_f[:, t] = sigmoid(
            betas[0]
            + np.dot(A_win, betas[1:1+d])
            + np.dot(L_win_full, betas[1+d:])
        )
        A[:, t] = np.random.binomial(1, A_f[:, t])

        A_win_full = A[:, t-d:t+1]  # (S, d+1)
        
        Y[:, t-d] = (
            alphas[0]
            + np.dot(A_win_full, alphas[1:2+d])
            + np.dot(L_win_full, alphas[2+d:])
        )
    
    # slice off buffer
    start = buffer
    end = T - d
    Y_slice = Y[:, start:end]
    L_slice = L[:, d+start:d+end]
    A_slice = A[:, d+start:d+end]
    f_slice = A_f[:, d+start:d+end]
    
    return np.stack([Y_slice, L_slice, A_slice, f_slice], axis=0)

# For d=1 this function outputs the same as the conditional_intervened_data function
def data_conditional_on_cov_and_treatment(
    d: int,
    initial_sim_data: np.ndarray,
    idx: np.ndarray,
    treatment: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
    gammas: np.ndarray,
    S: int,
    var_X: float = 0.1,
    mean_diff_X: float = 1,
    p_X = 0.5,
    model_X: str = "id"
):
    subset = initial_sim_data[:3, idx, :]  # (3, n_test, num_time_steps)
    n_test = idx.shape[0]

    L = np.zeros((n_test, S, 2*d+1))
    A = np.zeros((n_test, S, 2*d+1))
    A_f = np.zeros((n_test, S, 2*d+1))
    Y = np.zeros((n_test, S, 2*d+1))

    # broadcast history
    for arr, vals in zip((Y, L, A, A_f), subset):
        arr[:, :, :d+1] = vals[:, None, :d+1]

    for t in range(d, 2*d+1):
        A_win = A[:, :, t-d:t]   # (n_test, S, d)
        L_win = L[:, :, t-d:t]   # (n_test, S, d)

        # np.dot with 3D X and 1D w => dots over last axis
        L_f = (
            gammas[0]
            + np.dot(A_win, gammas[1:1+d])
            + np.dot(L_win, gammas[1+d:])
        )

        if model_X == "id":
            L[:, :, t] = L_f
        elif model_X == "normal":
            L[:, :, t] = np.random.normal(loc=L_f, scale=var_X)
        elif model_X == "mixture":
            upper = np.random.rand(S) < p_X
            locs  = L_f + np.where(upper, +mean_diff_X, -mean_diff_X)
            L[:, :, t] = np.random.normal(loc=locs, scale=var_X)

        L_win_full = L[:,:, t-d:t+1]  # (S, d+1)
        A_f[:,:,t] = sigmoid(
            betas[0]
            + np.dot(A_win, betas[1:1+d])
            + np.dot(L_win_full, betas[1+d:])
        )
        A[:,:,t] = np.random.binomial(1, A_f[:,:, t])

        A_win_full = A[:, :, t-d:t+1]  # (n_test, S, d+1)
        Y[:, :, t] = (
            alphas[0]
            + np.dot(A_win_full, alphas[1:2+d])
            + np.dot(L_win_full, alphas[2+d:])
        )
        
        
    # --- build treatment‐match mask ---
    mask = (A[:, :, d:2*d+1] == treatment.reshape(1,1,d+1)).all(axis=2)  # (n_test, S)
    counts = mask.sum(axis=1)
    m      = int(counts.min())   # number of trajectories we can keep per test‐case

    # --- filter baseline (first d of L) and final Y ---
    L_base   = L[:, :, :d+1]       # (n_test, S, d)
    Y_final  = Y[:, :, -1]       # (n_test, S)

    L_out = np.zeros((n_test, m, d+1))
    Y_out = np.zeros((n_test, m))

    for i in range(n_test):
        js = np.where(mask[i])[0][:m]
        L_out[i] = L_base[i, js, :]    # keep first m matching baselines
        Y_out[i] = Y_final[i, js]      # keep first m matching outcomes

    return L_out, Y_out

def conditional_intervened_data(
    d: int,
    initial_sim_data: np.ndarray,
    idx: np.ndarray,
    intervention: np.ndarray,
    alphas: np.ndarray,
    gammas: np.ndarray,
    S: int,
    var_X: float = 0.1,
    mean_diff_X: float = 1,
    p_X = 0.5,
    model_X: str = "id"
):
    subset = initial_sim_data[:3, idx, :]  # (3, n_test, num_time_steps)
    n_test = idx.shape[0]

    L = np.zeros((n_test, S, 2*d+1))
    A = np.zeros((n_test, S, 2*d+1))
    Y = np.zeros((n_test, S, 2*d+1))

    # broadcast history
    for arr, vals in zip((Y, L, A), subset):
        arr[:, :, :d+1] = vals[:, None, :d+1]

    # intervention
    A[:, :, d:2*d+1] = intervention[None, None, :]

    for t in range(d, 2*d+1):
        A_win = A[:, :, t-d:t]   # (n_test, S, d)
        L_win = L[:, :, t-d:t]   # (n_test, S, d)

        # np.dot with 3D X and 1D w => dots over last axis
        L_f = (
            gammas[0]
            + np.dot(A_win, gammas[1:1+d])
            + np.dot(L_win, gammas[1+d:])
        )

        if model_X == "id":
            L[:, :, t] = L_f
        elif model_X == "normal":
            L[:, :, t] = np.random.normal(loc=L_f, scale=var_X)
        elif model_X == "mixture":
            upper = np.random.rand(S) < p_X
            locs  = L_f + np.where(upper, +mean_diff_X, -mean_diff_X)
            L[:, :, t] = np.random.normal(loc=locs, scale=var_X)

        A_win_full = A[:, :, t-d:t+1]  # (n_test, S, d+1)
        L_win_full = L[:, :, t-d:t+1]  # (n_test, S, d+1)

        #print(A_win_full.shape, alphas[1:2+d].__len__())
        Y[:, :, t] = (
            alphas[0]
            + np.dot(A_win_full, alphas[1:2+d])
            + np.dot(L_win_full, alphas[2+d:])
        )

    return subset[:, :, :d+1], Y[:, :, -1]

def horizon_zero_distr(
    d: int,
    initial_sim_data: np.ndarray,
    idx: np.ndarray,
    alphas: np.ndarray,
    S: int
):
    subset = initial_sim_data[:3, idx, :]  # (3, n_test, num_time_steps)
    n_test = idx.shape[0]

    L = np.zeros((n_test, S, d+1))
    A = np.zeros((n_test, S, d+1))
    Y = np.zeros((n_test, S, d+1))

    # broadcast history
    for arr, vals in zip((Y, L, A), subset):
        arr[:, :, :d+1] = vals[:, None, :d+1]

    A_win_full = A[:, :, 0:d+1]  # (n_test, S, d+1)
    L_win_full = L[:, :, 0:d+1]  # (n_test, S, d+1)

    Y[:, :, d] = (
        alphas[0]
        + np.dot(A_win_full, alphas[1:2+d])
        + np.dot(L_win_full, alphas[2+d:])
    )

    return subset[:, :, :d+1], Y[:, :, -1]

def get_coefficients(d):

    if d==1:
        ycoeff = [-3,2,0,-1,0]
        acoeff = [-0.5,0.5,-0.5,0.5]
        xcoeff = [0,1,-1]
    elif d==3:
        ycoeff=[-1] + [3,6,12,1] + [0.5,1,2,1]
        acoeff= [-0.5] + [0.5,-0.5,0.5] + [-0.5,0.5,-0.5,0.5]
        xcoeff= [-1] + [0.5,1,1.5] + [-0.5,-1,-1.5]
    else:
        ycoeff=[-1] + [0.5,1,3,6,12,0] + [0.05,0.1, 0.5,1,2,0]
        acoeff= [-0.5] + [0.5,-0.5,0.5,-0.5,0.5] + [-0.5,0.5,-0.5,0.5,-0.5,0.5]
        xcoeff= [-1] + [0.05,0.1, 0.5,1,1.5] + [-0.05,-0.1, -0.5,-1,-1.5]
    
    return ycoeff, acoeff, xcoeff

def save_history_npz(H: np.ndarray, Y: np.ndarray, folder: str='data/sim-data', fname: str='horizon_3.npz'):
    os.makedirs(folder, exist_ok=True)
    np.savez_compressed(os.path.join(folder, fname), H=H, Y=Y)


def create_training_data(cfg: DictConfig):
    num_time_steps = cfg.data.train_size + cfg.data.buffer + 2 * cfg.data.dep_len
    alphas, betas, gammas = get_coefficients(cfg.data.dep_len)
    data = simulator(cfg.data.dep_len, alphas, betas, gammas, cfg.data.pop_size ,
                                    num_time_steps, cfg.data.buffer, var_X = cfg.data.covariate_var,
                                    mean_diff_X = cfg.data.covariate_mean_difference, p_X= cfg.data.covariate_mixture_weight,
                                    model_X = cfg.data.covariate_model)
    print('Simulated data shape:', data.shape)
    H, Y = add_history(data, cfg.data.dep_len)
    print(f'H shape : {H.shape}, Y shape : {Y.shape}')
    save_history_npz(H, Y)

def visualize_data(cfg: DictConfig):
    # Create the data
    print('Creating data...')
    num_time_steps = cfg.data.train_size + cfg.data.buffer + 2 * cfg.data.dep_len
    alphas, betas, gammas = get_coefficients(cfg.data.dep_len)
    data = simulator(cfg.data.dep_len, alphas, betas, gammas, cfg.data.pop_size ,
                                    num_time_steps, cfg.data.buffer, var_X = cfg.data.covariate_var,
                                    mean_diff_X = cfg.data.covariate_mean_difference, p_X= cfg.data.covariate_mixture_weight,
                                    model_X = cfg.data.covariate_model)
    print('Simulated data shape:', data.shape)
    plot_marginal_distributions(data)

    interventions = np.array([[0, 0, 0, 0], [1,1,1,1]])
    if cfg.data.dep_len == 1 : interventions = np.array([[0], [1]])
    if cfg.data.dep_len == 5 : interventions = np.array([[0,0,0,0,0]])
    assert interventions.shape[1] == cfg.data.dep_len + 1, "Intervention length must be equal to dep_len"
    
    cov_idx = np.random.choice(cfg.data.pop_size, cfg.data.num_test_cov, replace=False)

    covs, outcomes = horizon_zero_distr(cfg.data.dep_len, data, cov_idx, alphas, 100 * cfg.data.pop_size)
    print('outcomes shape:', outcomes.shape)
    print(covs)
    
    plot_intervention_conditional_distributions(outcomes, np.zeros(cfg.data.dep_len), cfg.data.dep_len)
    #save covariates
    print(f'Covariates shape: {covs.shape}')
    os.makedirs('data/sim-data', exist_ok=True)
    np.savez_compressed(os.path.join('data/sim-data', 'covs.npz'), C=covs)


    covs, outcomes = data_conditional_on_cov_and_treatment(cfg.data.dep_len, data, cov_idx, interventions[0], alphas, betas, gammas,
                                                100 * cfg.data.pop_size, var_X = cfg.data.covariate_var,
                                                mean_diff_X = cfg.data.covariate_mean_difference, p_X= cfg.data.covariate_mixture_weight,
                                                model_X = cfg.data.covariate_model)  
    print('outcomes shape:', outcomes.shape)
    plot_intervention_conditional_distributions(outcomes, interventions[0], cfg.data.dep_len)

    covs, outcomes = conditional_intervened_data(cfg.data.dep_len, data, cov_idx, interventions[0], alphas, gammas,
                                                cfg.data.pop_size, var_X = cfg.data.covariate_var,
                                                mean_diff_X = cfg.data.covariate_mean_difference, p_X= cfg.data.covariate_mixture_weight,
                                                model_X = cfg.data.covariate_model)
    plot_intervention_conditional_distributions(outcomes, interventions[0], cfg.data.dep_len)
    print('outcomes shape:', outcomes.shape)


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    #create_training_data(cfg)
    visualize_data(cfg)

if __name__ == "__main__":
    seed = 21
    if seed >=0:
        np.random.seed(seed)
    main()