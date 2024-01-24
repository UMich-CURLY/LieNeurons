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


def sl3_killing_form(x, y):
    return 6*torch.trace(x@y)


def lie_bracket(x, y):
    return x@y - y@x


def equivariant_function(x, y):
    # return sl3_killing_form(x,z)*lie_bracket(lie_bracket(x,y),z) + sl3_killing_form(y,z)*lie_bracket(z,y)
    # return sl3_killing_form(x,z)*lie_bracket(lie_bracket(x,y),z) + sl3_killing_form(y,z)*lie_bracket(z,y)

    return lie_bracket(lie_bracket(x, y), y) + lie_bracket(y, x)


class sl3EquivDataSet5Input(Dataset):
    def __init__(self, data_path, device='cuda'):
        data = np.load(data_path)
        _, num_points = data['x1'].shape
        _, _, num_conjugate = data['x1_conjugate'].shape
        self.x1 = rearrange(torch.from_numpy(data['x1']).type(
            'torch.FloatTensor').to(device), 'k n -> n 1 k 1')
        self.x2 = rearrange(torch.from_numpy(data['x2']).type(
            'torch.FloatTensor').to(device), 'k n -> n 1 k 1')
        self.x3 = rearrange(torch.from_numpy(data['x3']).type(
            'torch.FloatTensor').to(device), 'k n -> n 1 k 1')
        self.x4 = rearrange(torch.from_numpy(data['x4']).type(
            'torch.FloatTensor').to(device), 'k n -> n 1 k 1')
        self.x5 = rearrange(torch.from_numpy(data['x5']).type(
            'torch.FloatTensor').to(device), 'k n -> n 1 k 1')
        self.x = torch.cat(
            (self.x1, self.x2, self.x3, self.x4, self.x5), dim=1)

        self.x1_conjugate = rearrange(torch.from_numpy(data['x1_conjugate']).type(
            'torch.FloatTensor').to(device), 'k n c -> c n 1 k 1')
        self.x2_conjugate = rearrange(torch.from_numpy(data['x2_conjugate']).type(
            'torch.FloatTensor').to(device), 'k n c -> c n 1 k 1')
        self.x3_conjugate = rearrange(torch.from_numpy(data['x3_conjugate']).type(
            'torch.FloatTensor').to(device), 'k n c -> c n 1 k 1')
        self.x4_conjugate = rearrange(torch.from_numpy(data['x4_conjugate']).type(
            'torch.FloatTensor').to(device), 'k n c -> c n 1 k 1')
        self.x5_conjugate = rearrange(torch.from_numpy(data['x5_conjugate']).type(
            'torch.FloatTensor').to(device), 'k n c -> c n 1 k 1')
        self.x_conjugate = torch.cat(
            (self.x1_conjugate, self.x2_conjugate, self.x3_conjugate, self.x4_conjugate, self.x5_conjugate), dim=2)
        # [C,N,5,8,1]

        self.y = rearrange(torch.from_numpy(data['y']).type(
            'torch.FloatTensor').to(device), '1 n k -> n k')  # [N,8]

        self.H = torch.from_numpy(data['H']).type(
            'torch.FloatTensor').to(device)   # [C,N,3,3]

        self.num_data = self.x1.shape[0]

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'x1': self.x1[idx, :, :, :], 'x2': self.x2[idx, :, :, :], 'x3': self.x3[idx, :, :, :],
                  'x4': self.x4[idx, :, :, :], 'x5': self.x5[idx, :, :, :], 'x': self.x[idx, :, :, :],
                  'x1_conjugate': self.x1_conjugate[:, idx, :, :, :], 'x2_conjugate': self.x2_conjugate[:, idx, :, :, :],
                  'x3_conjugate': self.x3_conjugate[:, idx, :, :, :], 'x4_conjugate': self.x4_conjugate[:, idx, :, :, :],
                  'x5_conjugate': self.x2_conjugate[:, idx, :, :, :], 'x_conjugate': self.x_conjugate[:, idx, :, :, :],
                  'y': self.y[idx, :], 'H': self.H[:, idx, :, :]}
        return sample


class sl3EquivDataSetLieBracket(Dataset):
    def __init__(self, data_path, device='cuda'):
        data = np.load(data_path)
        _, num_points = data['x1'].shape
        _, _, num_conjugate = data['x1_conjugate'].shape
        self.x1 = rearrange(torch.from_numpy(data['x1']).type(
            'torch.FloatTensor').to(device), 'k n -> n 1 k 1')
        self.x2 = rearrange(torch.from_numpy(data['x2']).type(
            'torch.FloatTensor').to(device), 'k n -> n 1 k 1')
        self.x3 = rearrange(torch.from_numpy(data['x3']).type(
            'torch.FloatTensor').to(device), 'k n -> n 1 k 1')
        self.x = torch.cat((self.x1, self.x2, self.x3), dim=1)

        self.x1_conjugate = rearrange(torch.from_numpy(data['x1_conjugate']).type(
            'torch.FloatTensor').to(device), 'k n c -> c n 1 k 1')
        self.x2_conjugate = rearrange(torch.from_numpy(data['x2_conjugate']).type(
            'torch.FloatTensor').to(device), 'k n c -> c n 1 k 1')
        self.x3_conjugate = rearrange(torch.from_numpy(data['x3_conjugate']).type(
            'torch.FloatTensor').to(device), 'k n c -> c n 1 k 1')
        self.x_conjugate = torch.cat(
            (self.x1_conjugate, self.x2_conjugate, self.x3_conjugate), dim=2)
        # [C,N,5,8,1]

        self.y = rearrange(torch.from_numpy(data['y']).type(
            'torch.FloatTensor').to(device), '1 n k -> n k')  # [N,8]

        self.H = torch.from_numpy(data['H']).type(
            'torch.FloatTensor').to(device)   # [C,N,3,3]

        self.num_data = self.x1.shape[0]

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'x1': self.x1[idx, :, :, :], 'x2': self.x2[idx, :, :, :], 'x3': self.x3[idx, :, :, :],
                  'x': self.x[idx, :, :, :], 'x1_conjugate': self.x1_conjugate[:, idx, :, :, :],
                  'x2_conjugate': self.x2_conjugate[:, idx, :, :, :], 'x3_conjugate': self.x3_conjugate[:, idx, :, :, :],
                  'x_conjugate': self.x_conjugate[:, idx, :, :, :],
                  'y': self.y[idx, :], 'H': self.H[:, idx, :, :]}
        return sample


class sl3EquivDataSetLieBracket2Input(Dataset):
    def __init__(self, data_path, device='cuda'):
        data = np.load(data_path)
        _, num_points = data['x1'].shape
        _, _, num_conjugate = data['x1_conjugate'].shape
        self.x1 = rearrange(torch.from_numpy(data['x1']).type(
            'torch.FloatTensor').to(device), 'k n -> n 1 k 1')
        self.x2 = rearrange(torch.from_numpy(data['x2']).type(
            'torch.FloatTensor').to(device), 'k n -> n 1 k 1')
        self.x = torch.cat((self.x1, self.x2), dim=1)

        self.x1_conjugate = rearrange(torch.from_numpy(data['x1_conjugate']).type(
            'torch.FloatTensor').to(device), 'k n c -> c n 1 k 1')
        self.x2_conjugate = rearrange(torch.from_numpy(data['x2_conjugate']).type(
            'torch.FloatTensor').to(device), 'k n c -> c n 1 k 1')
        self.x_conjugate = torch.cat(
            (self.x1_conjugate, self.x2_conjugate), dim=2)
        # [C,N,5,8,1]

        self.y = rearrange(torch.from_numpy(data['y']).type(
            'torch.FloatTensor').to(device), '1 n k -> n k')  # [N,8]
        self.y_conj = rearrange(torch.from_numpy(data['y_conj']).type(
            'torch.FloatTensor').to(device), 'c n k -> c n k')  # [N,8]
        self.H = torch.from_numpy(data['H']).type(
            'torch.FloatTensor').to(device)   # [C,N,3,3]

        self.num_data = self.x1.shape[0]

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'x1': self.x1[idx, :, :, :], 'x2': self.x2[idx, :, :, :],
                  'x': self.x[idx, :, :, :], 'x1_conjugate': self.x1_conjugate[:, idx, :, :, :],
                  'x2_conjugate': self.x2_conjugate[:, idx, :, :, :],
                  'x_conjugate': self.x_conjugate[:, idx, :, :, :],
                  'y': self.y[idx, :], 'y_conj': self.y_conj[:, idx, :], 'H': self.H[:, idx, :, :]}
        return sample


if __name__ == "__main__":

    DataLoader = sl3EquivDataSetLieBracket2Input(
        "data/sl3_equiv_lie_bracket_data/sl3_equiv_10000_lie_bracket_2inputs_augmented_train_data.npz")

    hat = HatLayer(algebra_type='sl3').to('cuda')
    print(DataLoader.x1.shape)
    print(DataLoader.x2.shape)
    print(DataLoader.x1_conjugate.shape)
    print(DataLoader.x2_conjugate.shape)
    print(DataLoader.x.shape)
    print(DataLoader.x_conjugate.shape)
    print(DataLoader.H.shape)
    print(DataLoader.y.shape)
    sum = 0
    for i, samples in tqdm(enumerate(DataLoader, start=0)):
        input_data = samples['x']
        # print(input_data.shape)
        x1_hat = hat(input_data[0, :, :].squeeze(-1))
        x2_hat = hat(input_data[1, :, :].squeeze(-1))
        y_func = equivariant_function(x1_hat, x2_hat)
        y = hat(samples['y'])
        # print(torch.trace(hat(y)))
        # print("y_func", y_func)
        # print("y", y)
        # print("diff", y_func-y)
        # print("-------------")
        if(torch.norm(y_func-y) > 1e-3):
            print("\ny_func", y_func)
            print("y", y)
            print("diff", y_func-y)
            print("norm", torch.norm(y_func-y))
            print("-------------")
        sum += torch.norm(y)
        # print(y)
        # print(input_data.shape)

    print(sum/(i+1))
