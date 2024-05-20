import torch
import yaml
from torch.utils.data import Dataset, DataLoader
import pandas as pd

def yaml_loader(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as exc:
            print(exc)


class IMUDataset(Dataset):
    def __init__(self, csv_file, yaml_file):
        
        self.ori_data = pd.read_csv(csv_file, header=1)
        self.ori_data.columns = ["time", "w_x", "w_y", "w_z", "a_x", "a_y", "a_z"]
        self.data = self.ori_data.to_numpy()

        yaml_file = yaml_loader(yaml_file)
        self.dt = yaml_file['rate_hz']

        self.datatime_unit = "ns"

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.data[idx]
        sample = torch.tensor(sample)
        return sample
    
    def convert_time_ns_to_s(self,set_init_time_to_zero=True):
        if self.datatime_unit == "ns":
            self.data[:,0] = self.data[:,0] / 1e9
            self.datatime_unit = "s"
        else:
            print("Data time unit is already in seconds.")
        if set_init_time_to_zero and self.data[0,0] != 0:
            self.data[:,0] = self.data[:,0] - self.data[0,0]
        else:
                print("Initial time is already set to zero.")

    
    def get_dt(self):
        return self.dt
    
    def get_data(self):
        return self.data