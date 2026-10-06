import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

def data_loader(data_name, batch_size):
    mean = None
    std = None

    if data_name == 'normal':
        x = np.load(f'./data/normal_x.npy')
        u = np.load(f'./data/normal_u.npy')

        # normalization
        mean = x.mean(0)
        std = x.std(0)
        x = (x - mean) / std

    elif data_name == 'exp':
        x = np.load(f'./data/exp_x.npy')
        u = np.load(f'./data/exp_u.npy')

        # normalization
        mean = x.mean(0)
        std = x.std(0)
        x = (x - mean) / std
    
    elif data_name == 'mix':
        x = np.load(f'./data/mix_x.npy')
        u = np.load(f'./data/mix_u.npy')

        # normalization
        mean = x.mean(0)
        std = x.std(0)
        x = (x - mean) / std
    
    elif data_name == 'copula':
        df = pd.read_csv('./data/copula.csv', index_col=0)
        x = df.drop(['u.1', 'u.2', 'u.3', 'u.4', 'u.5'], axis=1).values
        u = df[['u.1', 'u.2', 'u.3', 'u.4', 'u.5']].values
    
    elif data_name == 'cortex':
        x = np.load(f'./data/cortex_x.npy')
        u = np.load(f'./data/cortex_u.npy', allow_pickle=True)
    
    elif data_name == 'VerbAgg':
        df = pd.read_csv('./data/VerbAgg.csv', index_col=0)
        x = df.values
        u = None

    else:
        raise Exception("Please double-check the dataset name.")
     
    train_loader = DataLoader(torch.from_numpy(x).unsqueeze(1).to(torch.float32), batch_size = batch_size, shuffle = True)
    train_loader.mean = mean
    train_loader.std = std
    
    return x, u, train_loader 