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
from core.lie_alg_util import *
from core.lie_neurons_layers import *
from experiment.sl3_equiv_layers import *
from data_loader.sl3_equiv_data_loader import *


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


def test_equivariance(model, test_loader, criterion, config, device):
    model.eval()
    hat_layer = HatLayerSl3().to(device)
    with torch.no_grad():
        loss_sum = 0.0
        diff_output_sum = 0.0
        for i, sample in tqdm(enumerate(test_loader, start=0)):
            x = sample['x'].to(device)  # [B, 5, 8, 1]
            x_conj = sample['x_conjugate'].to(device)   # [B, C, 5, 8, 1]
            y = sample['y'].to(device)  # [B, 8]
            H = sample['H'].to(device)  # [B, C, 3, 3]

            output_x = model(x) # [B, 8]
            output_x_hat = hat_layer(output_x)  # [B, 3, 3]
            # print(output_x)
            # print(x_conj.shape)
            # print(H.shape)

            for j in range(x_conj.shape[1]):
                x_conj_j = x_conj[:, j, :, :, :]    # [B, 5, 8, 1]
                H_j = H[:, j, :, :] # [B, 3, 3]
                conj_output = model(x_conj_j)   # [B, 8]
                # print('output hat', output_x_hat.shape)
                # print('H_j', H_j.shape)
                output_then_conj_hat = torch.matmul(H_j, torch.matmul(output_x_hat, torch.inverse(H_j)))
                # print('output then conj hat', output_then_conj_hat.shape)
                output_then_conj = vee_sl3(output_then_conj_hat)

                diff_output = output_then_conj - conj_output
                # print(output_then_conj)
                # print(conj_output)
                # print("----------------------")

                y_hat = hat_layer(y)
                conj_y_hat = torch.matmul(H_j, torch.matmul(y_hat, torch.inverse(H_j)))
                conj_y = vee_sl3(conj_y_hat)
                loss = criterion(conj_output, conj_y)

                loss_sum += loss.item()
                diff_output_sum += torch.sum(torch.abs(diff_output))

                # print('diff_output: ', diff_output)

        loss_avg = loss_sum/len(test_loader)/x_conj.shape[1]
        diff_output_avg = diff_output_sum/len(test_loader.dataset)/x_conj.shape[1]/x_conj.shape[3]

    return loss_avg, diff_output_avg

def main():
    # torch.autograd.set_detect_anomaly(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using ', device)

    parser = argparse.ArgumentParser(description='Train the network')
    parser.add_argument('--test_config', type=str,
                        default=os.path.dirname(os.path.abspath(__file__))+'/../config/sl3_equiv/testing_param.yaml')
    args = parser.parse_args()

    # load yaml file
    config = yaml.safe_load(open(args.test_config))

    test_set = sl3EquivDataSet5Input(config['test_data_path'], device=device)
    test_loader = DataLoader(dataset=test_set, batch_size=config['batch_size'],
                             shuffle=config['shuffle'])
    
    if config['model_type'] == "LN":
        model = SL3EquivariantLayers(5).to(device)
    elif config['model_type'] == "MLP":
        model = MLP(40).to(device)

    # model = SL3InvariantLayersTest(2).to(device)
    checkpoint = torch.load(config['model_path'])
    model.load_state_dict(checkpoint['model_state_dict'])

    criterion = nn.MSELoss().to(device)
    test_loss_equiv, diff_output_avg = test_equivariance(model, test_loader, criterion, config, device)
    print("test loss: ", test_loss_equiv)
    print("avg diff output: ", diff_output_avg)

if __name__ == "__main__":
    main()
