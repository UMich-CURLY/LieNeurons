import sys  # nopep8
sys.path.append('.')  # nopep8

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.linalg import expm
from tqdm import tqdm


from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from core.lie_neurons_layers import *


class sp4InvDataSet(Dataset):
    def __init__(self, data_path, device='cuda'):
        data = np.load(data_path)
        _, num_points = data['x1'].shape
        _,_,num_conjugate = data['x1_conjugate'].shape
        self.x1 = rearrange(torch.from_numpy(data['x1']).type(
            'torch.FloatTensor').to(device), 'k n -> n 1 k 1')
        self.x2 = rearrange(torch.from_numpy(data['x2']).type(
            'torch.FloatTensor').to(device), 'k n -> n 1 k 1')
        self.x = torch.cat((self.x1, self.x2), dim=1)

        self.x1_conjugate = rearrange(torch.from_numpy(data['x1_conjugate']).type(
            'torch.FloatTensor').to(device),'k n c -> c n 1 k 1')
        self.x2_conjugate = rearrange(torch.from_numpy(data['x2_conjugate']).type(
            'torch.FloatTensor').to(device),'k n c -> c n 1 k 1')
        self.x_conjugate = torch.cat(
            (self.x1_conjugate, self.x2_conjugate), dim=2)
        self.y = torch.from_numpy(data['y']).type(
            'torch.FloatTensor').to(device).reshape(num_points, 1)

        self.num_data = self.x1.shape[0]

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'x1': self.x1[idx, :, :, :], 'x2': self.x2[idx, :, :, :], 'x': self.x[idx, :, :, :],
                  'x1_conjugate': self.x1_conjugate[:,idx, :, :, :], 'x2_conjugate': self.x2_conjugate[:,idx, :, :, :],
                  'x_conjugate': self.x_conjugate[:,idx, :, :, :], 'y': self.y[idx, :]}
        return sample


if __name__ == "__main__":

    DataLoader = sp4InvDataSet("data/sp4_inv_data/sp4_inv_10000_s_05_train_data.npz")

    print(DataLoader.x1.shape)
    print(DataLoader.x2.shape)
    print(DataLoader.x1_conjugate.shape)
    print(DataLoader.x2_conjugate.shape)
    print(DataLoader.x.shape)
    print(DataLoader.x_conjugate.shape)
    print(DataLoader.y.shape)
    for i, samples in tqdm(enumerate(DataLoader, start=0)):
        input_data = samples['x']
        y = samples['y']
        print(input_data.shape)
        print("x1: \n",input_data[0,:,0])
        print("x2: \n",input_data[1,:,0])
        print("y: ", y)
        print("------------------")
        # print(input_data.shape)
