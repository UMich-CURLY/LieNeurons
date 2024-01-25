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

from core.lie_neurons_layers import *
from experiment.so3_bch_layers import *
from data_loader.so3_bch_data_loader import *

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


def test_bch(model, test_loader, criterion, config, device):
    model.eval()
    hat_layer = HatLayer(algebra_type='so3').to(device)
    with torch.no_grad():
        loss_sum = 0.0
        loss_non_conj_sum = 0.0
        diff_output_sum = 0.0
        for i, sample in tqdm(enumerate(test_loader, start=0)):
            x = sample['x'].to(device)  # [B, 2, 3, 1]
            x_conj = sample['x_conj'].to(device)   # [B, C, 2, 3, 1]
            y = sample['y'].to(device)  # [B, 3]
            R = sample['R'].to(device)  # [B, C, 3, 3]
            y_conj = sample['y_conj']   # [B, C, 3]

            # print('x', x.shape)
            # print('x_conj', x_conj.shape)
            # print('y', y.shape)
            # print('R', R.shape)
            # print('y_conj', y_conj.shape)

            output_x = model(x) # [B, 3]
            output_x_hat = hat_layer(output_x)  # [B, 3, 3]
            loss_non_conj_sum += criterion(output_x, y).item()
            print("----------------------------------")
            # compute bch ground truth again
            R1 = exp_so3(hat_layer(x[:, 0, :, 0]))
            R2 = exp_so3(hat_layer(x[:, 1, :, 0]))
            error = log_SO3(R1 @ R2 @ exp_so3(-hat_layer(output_x)))
            # y_real = vee_so3(log_SO3(R3))
            print("norm: ", torch.norm(output_x - y))
            print("error: ", torch.norm(error[0,:]))
            print('output', output_x[0,:])
            # print("y_real", y_real[0,:])

            print("bch approx",vee(BCH_approx(hat_layer(x[:, 0, :, 0]),hat_layer(x[:, 1, :, 0])),algebra_type='so3'))
            print("output restricted", vee(log_SO3(exp_so3(hat_layer(output_x))),'so3'))
            print("bch restricted", vee(log_SO3(exp_so3(BCH_approx(hat_layer(x[:, 0, :, 0]),hat_layer(x[:, 1, :, 0])))),'so3'))
            print('y', y[0,:])

            for j in range(x_conj.shape[1]):
                x_conj_j = x_conj[:, j, :, :, :]    # [B, 5, 8, 1]
                R_j = R[:, j, :, :] # [B, 3, 3]
                conj_output = model(x_conj_j)   # [B, 8]
                # print('output hat', output_x_hat.shape)
                # print('R_j', R_j.shape)
                output_then_conj_hat = torch.matmul(R_j, torch.matmul(output_x_hat, R_j.transpose(1, 2)))
                # print('output then conj hat', output_then_conj_hat.shape)
                output_then_conj = vee_so3(output_then_conj_hat)

                diff_output = output_then_conj - conj_output
                # print(output_then_conj)
                
                conj_output_hat = hat_layer(conj_output)
                conj_output_hat_conj_back = torch.matmul(R_j.transpose(1,2), torch.matmul(conj_output_hat, R_j))
                conj_output_conj_back = vee_so3(conj_output_hat_conj_back)


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

    parser = argparse.ArgumentParser(description='Test the network')
    parser.add_argument('--test_config', type=str,
                        default=os.path.dirname(os.path.abspath(__file__))+'/../config/so3_bch/testing_param.yaml')
    args = parser.parse_args()

    # load yaml file
    config = yaml.safe_load(open(args.test_config))

    test_set = so3BchTestDataSet(
        config['test_data_path'], device=device)
    test_loader = DataLoader(dataset=test_set, batch_size=config['batch_size'],
                             shuffle=config['shuffle'])

    if config['model_type'] == "LN_relu_bracket":
        model = SO3EquivariantReluBracketLayers(2).to(device)
    elif config['model_type'] == "LN_relu":
        model = SO3EquivariantReluLayers(2).to(device)
    elif config['model_type'] == "LN_bracket":
        model = SO3EquivariantBracketLayers(2).to(device)
    elif config['model_type'] == "MLP":
        model = MLP(6).to(device)
    # elif config['model_type'] == "LN_bracket_no_residual":
    #     model = SO3EquivariantBracketNoResidualConnectLayers(2).to(device)


    print("Using model: ", config['model_type'])
    print("total number of parameters: ", sum(p.numel() for p in model.parameters()))

    # model = SL3InvariantLayersTest(2).to(device)
    checkpoint = torch.load(config['model_path'])
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)

    criterion = nn.MSELoss().to(device)
    test_loss_equiv, loss_non_conj_avg, diff_output_avg = test_bch(model, test_loader, criterion, config, device)
    print("test_loss type:",type(test_loss_equiv))
    # print("avg diff output type: ", diff_output_avg.dtype)
    print("test with augmentation loss: ", test_loss_equiv)
    print("avg diff output: ", diff_output_avg)
    print("loss non conj avg: ", loss_non_conj_avg)

if __name__ == "__main__":
    main()
