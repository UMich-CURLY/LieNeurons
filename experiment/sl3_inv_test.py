import sys  # nopep8
sys.path.append('.')  # nopep8

import argparse
import os
import time
import yaml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

import torch
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

from core.lie_group_util import *
from core.lie_neurons_layers import *
from experiment.sl3_inv_layers import *
from data_loader.sl3_inv_data_loader import *


def test(model, test_loader, criterion, config, device):
    model.eval()
    with torch.no_grad():
        loss_sum = 0.0
        for i, sample in tqdm(enumerate(test_loader, start=0)):
            x = sample['x'].to(device)
            y = sample['y'].to(device)

            output = model(x)

            loss = criterion(output, y)
            loss_sum += loss.item()

        loss_avg = loss_sum/len(test_loader)

    return loss_avg


def test_invariance(model, test_loader, criterion, config, device):
    model.eval()
    with torch.no_grad():
        loss_sum = 0.0
        diff_output_sum = 0.0
        for i, sample in tqdm(enumerate(test_loader, start=0)):
            x = sample['x'].to(device)
            x_conj = sample['x_conjugate'].to(device)
            y = sample['y'].to(device)

            output_x = model(x)
            # print(output_x)
            for j in range(x_conj.shape[1]):
                x_conj_j = x_conj[:, j, :, :, :]
                output_conj = model(x_conj_j)
                diff_output = output_x - output_conj
                loss = criterion(output_conj, y)
                loss_sum += loss.item()
                diff_output_sum += torch.sum(diff_output)
                # print(output_conj)

        loss_avg = loss_sum/len(test_loader)/x_conj.shape[1]
        diff_output_avg = diff_output_sum/len(test_loader.dataset)/x_conj.shape[1]

    return loss_avg, diff_output_avg

def main():
    # torch.autograd.set_detect_anomaly(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using ', device)

    parser = argparse.ArgumentParser(description='Train the network')
    parser.add_argument('--test_config', type=str,
                        default=os.path.dirname(os.path.abspath(__file__))+'/../config/sl3_inv/testing_param.yaml')
    args = parser.parse_args()

    # load yaml file
    config = yaml.safe_load(open(args.test_config))

    test_set = sl3InvDataSet5Input(config['test_data_path'], device=device)
    test_loader = DataLoader(dataset=test_set, batch_size=config['batch_size'],
                             shuffle=config['shuffle'])
    # for i, samples in tqdm(enumerate(train_loader, start=0)):
    # model = SL3InvariantLayers(5).to(device)
    # model = SL3InvariantLayersTest(2).to(device)
    model = MLP(40).to(device)
    checkpoint = torch.load(config['model_path'])
    model.load_state_dict(checkpoint['model_state_dict'])

    criterion = nn.MSELoss().to(device)
    # test_loss = test(model, test_loader, criterion, config, device)
    test_loss_inv, diff_output_avg = test_invariance(model, test_loader, criterion, config, device)
    print("test loss: ", test_loss_inv)
    print("avg diff output: ", diff_output_avg)

if __name__ == "__main__":
    main()
