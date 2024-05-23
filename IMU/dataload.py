import torch
import yaml
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

def yaml_loader(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as exc:
            print(exc)

class ToTensor:
    def __call__(self, sample):
        # Convert the sample to a tensor
        return torch.tensor(sample.values, dtype=torch.float)

class IMUDataset(Dataset):
    def __init__(self, csv_file, yaml_file = None, transform=ToTensor()):
        self.data = pd.read_csv(csv_file, header=1, names=["time", "w_x", "w_y", "w_z", "a_x", "a_y", "a_z"])
        
        # Convert the time column from nanoseconds to seconds
        # self.data['time'] = self.data['time'] - self.data['time'][0]
        self.data['time'] = self.data['time'] / 1e9
        
        self.transform = transform
        if yaml_file:
            yaml_file = yaml_loader(yaml_file)
            fre = yaml_file['rate_hz']
            self.dt = 1.0/fre

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data.iloc[idx]
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def get_dt(self):
        return self.dt

class GroudtruethDataset(Dataset):
    def __init__(self, csv_file, yaml_file = None, transform=ToTensor()):
        self.data = pd.read_csv(csv_file, header=1, names=["time", "px", "py", "pz", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "bgx", "bgy", "bgz", "bax", "bay", "baz"])
        
        # Convert the time column from nanoseconds to seconds
        # self.data['time'] = self.data['time'] - self.data['time'][0]
        self.data['time'] = self.data['time'] / 1e9
        
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data.iloc[idx]
        if self.transform:
            sample = self.transform(sample)
        
        return sample


def align_time(IMUdata: IMUDataset, GTdata: GroudtruethDataset):
    time_baseline = max(IMUdata.data['time'][0], GTdata.data['time'][0])
    
    if time_baseline == IMUdata.data['time'][0]:
        diff = np.abs(GTdata.data['time'] - time_baseline)
        idx = np.argmin(diff)
        GTdata.data = GTdata.data.iloc[idx:].reset_index(drop=True)
    else:
        diff = np.abs(IMUdata.data['time'] - time_baseline)
        idx = np.argmin(diff)
        IMUdata.data = IMUdata.data.iloc[idx:].reset_index(drop=True)
    
    IMUdata.data['time'] = IMUdata.data['time'] - time_baseline
    GTdata.data['time'] = GTdata.data['time'] - time_baseline
    return IMUdata, GTdata

# class IMUDataset(Dataset):
#     def __init__(self, csv_file, yaml_file) -> None:
        
#         self.ori_data = pd.read_csv(csv_file, header=1)
#         self.ori_data.columns = ["time", "w_x", "w_y", "w_z", "a_x", "a_y", "a_z"]

#         self.data = self.ori_data.to_numpy()

#         yaml_file = yaml_loader(yaml_file)
#         fre = yaml_file['rate_hz']
#         self.dt = 1.0/fre

#         self.datatime_unit = "ns"

#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
        
#         sample = self.data[idx]
#         sample = torch.tensor(sample)
#         return sample
    
#     def convert_time_ns_to_s(self,set_init_time_to_zero=True):
#         if self.datatime_unit == "ns":
#             self.data[:,0] = self.data[:,0] / 1e9
#             self.datatime_unit = "s"
#         else:
#             print("Data time unit is already in seconds.")
#         if set_init_time_to_zero and self.data[0,0] != 0:
#             self.data[:,0] = self.data[:,0] - self.data[0,0]
#         else:
#                 print("Initial time is already set to zero.")

    
#     def get_dt(self):
#         return self.dt
    
#     def get_data(self):
#         return self.data
    

# class GroudtruethDataset(Dataset):
#     def __init__(self, csv_file, yaml_file = None) -> None:
#         super().__init__()
#         self.ori_data = pd.read_csv(csv_file, header=1)
#         self.ori_data.columns = ["time", "px", "py", "pz", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "bgx", "bgy", "bgz", "bax", "bay", "baz"]
#         self.data = self.ori_data.to_numpy()

#         self.datatime_unit = "ns"

#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
        
#         sample = self.data[idx]
#         sample = torch.tensor(sample)
#         return sample
    
#     def convert_time_ns_to_s(self,set_init_time_to_zero=True):
#         if self.datatime_unit == "ns":
#             self.data[:,0] = self.data[:,0] / 1e9
#             self.datatime_unit = "s"
#         else:
#             print("Data time unit is already in seconds.")
#         if set_init_time_to_zero and self.data[0,0] != 0:
#             self.data[:,0] = self.data[:,0] - self.data[0,0]
#         else:
#             print("Initial time is already set to zero.")

#     def get_data(self):
#         return self.data
