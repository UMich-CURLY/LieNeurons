
import sys  # nopep8
sys.path.append('.')  # nopep8

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.linalg import expm
from tqdm import tqdm
import matplotlib.pyplot as plt


from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from core.lie_neurons_layers import *
from core.lie_group_util import *

class so3BchDataSet(Dataset):
    def __init__(self, data_path, device='cuda'):
        data = np.load(data_path)
        num_points, _ = data['x1'].shape
        
        self.x1 = rearrange(torch.from_numpy(data['x1']).type(
            'torch.FloatTensor').to(device), 'n k -> n 1 k 1')
        self.x2 = rearrange(torch.from_numpy(data['x2']).type(
            'torch.FloatTensor').to(device), 'n k -> n 1 k 1')
        self.x = torch.cat((self.x1, self.x2), dim=1)   # n 2 k 1

        self.y = torch.from_numpy(data['y']).type(
            'torch.FloatTensor').to(device) # [N,3]
        # print(self.y.shape)
        self.num_data = self.x1.shape[0]

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'x1': self.x1[idx, :, :, :], 'x2': self.x2[idx, :, :, :],
                  'x': self.x[idx, :, :, :], 'y': self.y[idx, :]}
        return sample

class so3BchTestDataSet(Dataset):
    '''
    Test data set contains augmented infomation
    '''
    def __init__(self, data_path, device='cuda'):
        data = np.load(data_path)
        num_points, _ = data['x1'].shape
        
        self.x1 = rearrange(torch.from_numpy(data['x1']).type(
            'torch.FloatTensor').to(device), 'n k -> n 1 k 1')
        self.x2 = rearrange(torch.from_numpy(data['x2']).type(
            'torch.FloatTensor').to(device), 'n k -> n 1 k 1')
        self.x = torch.cat((self.x1, self.x2), dim=1)   # n 2 k 1
        self.x1_conj = rearrange(torch.from_numpy(data['x1_conjugate']).type(
            'torch.FloatTensor').to(device), 'n c k -> c n 1 k 1')
        self.x2_conj = rearrange(torch.from_numpy(data['x2_conjugate']).type(
            'torch.FloatTensor').to(device), 'n c k -> c n 1 k 1')
        self.x_conj = torch.cat((self.x1_conj, self.x2_conj), dim=2)   # c n 2 k 1
        self.y = torch.from_numpy(data['y']).type(
            'torch.FloatTensor').to(device) # [N,3]
        self.y_conj = rearrange(torch.from_numpy(data['y_conj']).type(
            'torch.FloatTensor').to(device), 'n c k -> c n k') # [c, N,3]
        self.R = rearrange(torch.from_numpy(data['R_aug']).type(
            'torch.FloatTensor').to(device), 'n c k1 k2 -> c n k1 k2') # [c, N,3,3]
        # print(self.y.shape)
        self.num_data = self.x1.shape[0]

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'x1': self.x1[idx, :, :, :], 'x2': self.x2[idx, :, :, :],\
                  'x': self.x[idx, :, :, :], 'y': self.y[idx, :],\
                  'x1_conj': self.x1_conj[:,idx, :, :, :], 'x2_conj': self.x2_conj[:,idx, :, :, :],\
                  'x_conj': self.x_conj[:,idx,:,:,:],'y_conj': self.y_conj[:,idx, :], \
                  'R': self.R[:,idx, :, :]
                  }
        return sample


if __name__ == "__main__":

    DataLoader = so3BchDataSet(
        "data/so3_bch_data/so3_bch_10000_4_train_data.npz")

    hat = HatLayer(algebra_type='so3').to('cuda')
    # print(DataLoader.x1.shape)
    # print(DataLoader.x2.shape)
    # print(DataLoader.x.shape)
    # print(DataLoader.y.shape)

    sum = 0
    sum_inf = 0
    sum_nan = 0
    sum_y_nan = 0
    sum_y_inf = 0
    v3_list = torch.zeros((DataLoader.num_data, 3))
    for i, samples in tqdm(enumerate(DataLoader, start=0)):
        input_data = samples['x'].to('cuda')
        R1 = exp_so3(hat(input_data[0, :, :].squeeze(-1)))
        R2 = exp_so3(hat(input_data[1, :, :].squeeze(-1)))
        v3 = vee(log_SO3(torch.matmul(R1,R2)),algebra_type='so3')
        v3_list[i,:] = v3
        # print("v1", input_data[0, :, :].squeeze(-1))
        # print("v2", input_data[1, :, :].squeeze(-1))
        # print("v3", v3)
        # print("norm v3", torch.norm(v3))
        y = samples['y'].to('cuda')

        # print(torch.trace(hat(y)))
        # print(float(torch.norm(v3)))
        if(torch.isinf(input_data).any()):
            sum_inf += 1

        if(torch.isnan(input_data).any()):
            sum_nan += 1

        if(torch.isinf(y).any()):
            sum_y_inf += 1

        if(torch.isnan(y).any()):
            sum_y_nan += 1

        # print("norm", torch.norm(v3-y))

        if(torch.norm(v3-y) > 1e-3):
            # print("\nv3", v3)
            # print("y", y)
            # print("diff", v3-y)
            # print("norm", torch.norm(v3-y))
            # print("-------------")
            sum += 1
        
        # diff_norm = torch.norm(v3-y)
        
        # print(y)
        # print(input_data.shape)

    # write a code to plot histogram of v1, v2, v3
    plt.hist(v3_list[:,0].cpu().numpy(), bins=100)
    plt.show()

    print("error > 1e-3", sum)
    print("input inf num", sum_inf)
    print("input nan num", sum_nan)
    print("y inf num", sum_y_inf)
    print("y nan num", sum_y_nan)