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
    hat_layer = HatLayer(algebra_type='sl3').to(device)
    with torch.no_grad():
        loss_sum = 0.0
        loss_non_conj_sum = 0.0
        diff_output_sum = 0.0
        for i, sample in tqdm(enumerate(test_loader, start=0)):
            x = sample['x'].to(device)  # [B, 5, 8, 1]
            x_conj = sample['x_conjugate'].to(device)   # [B, C, 5, 8, 1]
            y = sample['y'].to(device)  # [B, 8]
            H = sample['H'].to(device)  # [B, C, 3, 3]
            y_conj = sample['y_conj']

            output_x = model(x) # [B, 8]
            output_x_hat = hat_layer(output_x)  # [B, 3, 3]
            loss_non_conj_sum += criterion(output_x, y).item()

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
                
                conj_output_hat = hat_layer(conj_output)
                conj_output_hat_conj_back = torch.matmul(torch.inverse(H_j), torch.matmul(conj_output_hat, H_j))
                conj_output_conj_back = vee_sl3(conj_output_hat_conj_back)


                # y_hat = hat_layer(y)
                # conj_y_hat = torch.matmul(H_j, torch.matmul(y_hat, torch.inverse(H_j)))
                # conj_y = vee_sl3(conj_y_hat)
                loss = criterion(conj_output_conj_back, y)
                # print("conj_out", conj_output[0,:])
                # print("y_conj", y_conj[0, j, :])
                # print("diff_out_out_conj", diff_output[0,:])
                # print("out",output_x[0,:])
                # print("y", y[0,:])
                # print("diff_out",output_x[0,:] - y[0,:])
                # print(loss.item())
                # print("----------------------")
                loss_sum += loss.item()
                diff_output_sum += torch.sum(torch.abs(diff_output))

                # print('diff_output: ', diff_output)

        loss_avg = loss_sum/len(test_loader)/x_conj.shape[1]
        diff_output_avg = diff_output_sum/len(test_loader.dataset)/x_conj.shape[1]/x_conj.shape[3]
        loss_non_conj_avg = loss_non_conj_sum/len(test_loader)

    return loss_avg, loss_non_conj_avg, diff_output_avg

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

    test_set = sl3EquivDataSetLieBracket2Input(config['test_data_path'], device=device)
    test_loader = DataLoader(dataset=test_set, batch_size=config['batch_size'],
                             shuffle=config['shuffle'])
    
    if config['model_type'] == "LN_relu_bracket":
        model = SL3EquivariantReluBracketLayers(2).to(device)
    elif config['model_type'] == "LN_relu":
        model = SL3EquivariantReluLayers(2).to(device)
    elif config['model_type'] == "LN_bracket":
        model = SL3EquivariantBracketLayers(2).to(device)
    elif config['model_type'] == "MLP":
        model = MLP(16).to(device)
    elif config['model_type'] == "LN_bracket_no_residual":
        model = SL3EquivariantBracketNoResidualConnectLayers(2).to(device)

    print("Using model: ", config['model_type'])
    print("total number of parameters: ", sum(p.numel() for p in model.parameters()))

    # model = SL3InvariantLayersTest(2).to(device)
    checkpoint = torch.load(config['model_path'])
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)

    criterion = nn.MSELoss().to(device)
    test_loss_equiv, loss_non_conj_avg, diff_output_avg = test_equivariance(model, test_loader, criterion, config, device)
    print("test_loss type:",type(test_loss_equiv))
    # print("avg diff output type: ", diff_output_avg.dtype)
    print("test loss: ", test_loss_equiv)
    print("avg diff output: ", diff_output_avg)
    print("loss non conj avg: ", loss_non_conj_avg)

if __name__ == "__main__":
    main()
