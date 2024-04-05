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


def test_bch(model, test_loader, config, device):
    model.eval()
    hat_layer = HatLayer(algebra_type='so3').to(device)
    with torch.no_grad():
        error_fro_sum = 0.0
        error_log_sum = 0.0
        error_fro_conj_sum = 0.0
        error_log_conj_sum = 0.0
        
        if config['calculate_eq_approx_error']:
            error_fro_eq_first_order_sum = 0.0
            error_log_eq_first_order_sum = 0.0
            error_fro_eq_second_order_sum = 0.0
            error_log_eq_second_order_sum = 0.0
            error_fro_eq_third_order_sum = 0.0
            error_log_eq_third_order_sum = 0.0

            error_fro_eq_first_order_conj_sum = 0.0
            error_log_eq_first_order_conj_sum = 0.0
            error_fro_eq_second_order_conj_sum = 0.0
            error_log_eq_second_order_conj_sum = 0.0
            error_fro_eq_third_order_conj_sum = 0.0
            error_log_eq_third_order_conj_sum = 0.0


        
        diff_output_sum = 0.0
        total_num = 0
        total_num_conj = 0
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
            # loss_non_conj_sum += criterion(output_x, y).item()
            # print("----------------------------------")
            # compute bch ground truth again
            R1 = exp_so3(hat_layer(x[:, 0, :, 0]))
            R2 = exp_so3(hat_layer(x[:, 1, :, 0]))
            residual_R = R1 @ R2 @ exp_so3(-hat_layer(output_x))
            # print(f"{output_x=}")
            # print(f"{R1=}")
            # print(f"{R2=}")
            # print(f"{hat_layer(output_x)=}")
            # print(f"{residual_R=}")
            # print(residual_R - torch.eye(3).to(device))
            residual_log = log_SO3(residual_R)
            error_log = torch.linalg.vector_norm(vee(residual_log, algebra_type='so3'), dim=1)
            error_fro = torch.linalg.matrix_norm(residual_R - torch.eye(3).to(device))
            # print(f"{error_fro=}")
            # error_fro = torch.norm(residual_R - torch.eye(3).to(device), p='fro')
            error_fro_sum += error_fro.sum().item()
            error_log_sum += error_log.sum().item()
            total_num += x.shape[0]

            if config['calculate_eq_approx_error']:
                # compute bch first order approx
                residual_R_eq_first_order = R1 @ R2 @ exp_so3(-BCH_first_order_approx(hat_layer(x[:, 0, :, 0]),hat_layer(x[:, 1, :, 0])))
                residual_log_eq_first_order = log_SO3(residual_R_eq_first_order)
                error_log_eq_first_order = torch.linalg.vector_norm(vee(residual_log_eq_first_order, algebra_type='so3'), dim=1)
                error_fro_eq_first_order = torch.linalg.matrix_norm(residual_R_eq_first_order - torch.eye(3).to(device))
                error_fro_eq_first_order_sum += error_fro_eq_first_order.sum().item()
                error_log_eq_first_order_sum += error_log_eq_first_order.sum().item()

                # compute bch second order approx
                residual_R_eq_second_order = R1 @ R2 @ exp_so3(-BCH_second_order_approx(hat_layer(x[:, 0, :, 0]),hat_layer(x[:, 1, :, 0])))
                residual_log_eq_second_order = log_SO3(residual_R_eq_second_order)
                error_log_eq_second_order = torch.linalg.vector_norm(vee(residual_log_eq_second_order, algebra_type='so3'), dim=1)
                error_fro_eq_second_order = torch.linalg.matrix_norm(residual_R_eq_second_order - torch.eye(3).to(device))
                error_fro_eq_second_order_sum += error_fro_eq_second_order.sum().item()
                error_log_eq_second_order_sum += error_log_eq_second_order.sum().item()

                # compute bch third order approx
                residual_R_eq_third_order = R1 @ R2 @ exp_so3(-BCH_third_order_approx(hat_layer(x[:, 0, :, 0]),hat_layer(x[:, 1, :, 0])))
                residual_log_eq_third_order = log_SO3(residual_R_eq_third_order)
                error_log_eq_third_order = torch.linalg.vector_norm(vee(residual_log_eq_third_order, algebra_type='so3'), dim=1)
                error_fro_eq_third_order = torch.linalg.matrix_norm(residual_R_eq_third_order - torch.eye(3).to(device))
                error_fro_eq_third_order_sum += error_fro_eq_third_order.sum().item()
                error_log_eq_third_order_sum += error_log_eq_third_order.sum().item()

            # print(f"{error_fro=}")
            # print(f"{error_log=}")

            # # y_real = vee_so3(log_SO3(R3))
            # print("norm: ", torch.norm(output_x - y))
            # print("error: ", torch.norm(error_log[0,:]))
            # print('output', output_x[0,:])
            # # print("y_real", y_real[0,:])

            # print("bch approx",vee(BCH_approx(hat_layer(x[:, 0, :, 0]),hat_layer(x[:, 1, :, 0])),algebra_type='so3'))
            # print("output restricted", vee(log_SO3(exp_so3(hat_layer(output_x))),'so3'))
            # print("bch restricted", vee(log_SO3(exp_so3(BCH_approx(hat_layer(x[:, 0, :, 0]),hat_layer(x[:, 1, :, 0])))),'so3'))
            # print('y', y[0,:])

            for j in range(x_conj.shape[1]):
                R_j = R[:, j, :, :] # [B, 1, 3, 3]
                x_conj_j = R_j.unsqueeze(1) @ x
                # x_conj_j = x_conj[:, j, :, :, :]    # [B, 5, 8, 1]
                conj_output = model(x_conj_j)   # [B, 8]
                # print('output hat', output_x_hat.shape)
                # print('R_j', R_j.shape)

                # check equivariance
                output_then_conj_hat = torch.matmul(R_j, torch.matmul(output_x_hat, R_j.transpose(1, 2)))
                # print('output then conj hat', output_then_conj_hat.shape)
                output_then_conj = vee_so3(output_then_conj_hat)
                diff_output = output_then_conj - conj_output
                diff_output_sum += torch.sum(torch.abs(diff_output))
                # print(output_then_conj)
                
                # compute bch error
                conj_output_hat = hat_layer(conj_output)
                R1 = exp_so3(hat_layer(x_conj_j[:, 0, :, 0]))
                R2 = exp_so3(hat_layer(x_conj_j[:, 1, :, 0]))
                # print(f"{x_conj_j=}")
                # print(f"{conj_output=}")
                # print(f"{R1=}")
                # print(f"{R2=}")
                # print(f"{conj_output_hat=}")
                residual_R = R1 @ R2 @ exp_so3(-conj_output_hat)
                # print(f"{residual_R=}")
                residual_log = log_SO3(residual_R)
                error_log_conj = torch.linalg.vector_norm(vee(residual_log, algebra_type='so3'), dim=1)
                error_fro_conj = torch.linalg.matrix_norm(residual_R - torch.eye(3).to(device))
                error_fro_conj_sum += error_fro_conj.sum().item()
                error_log_conj_sum += error_log_conj.sum().item()
                total_num_conj += x_conj_j.shape[0]
                
                if config['calculate_eq_approx_error']:
                    # compute bch first order approx
                    residual_R_eq_first_order = R1 @ R2 @ exp_so3(-BCH_first_order_approx(hat_layer(x_conj_j[:, 0, :, 0]),hat_layer(x_conj_j[:, 1, :, 0])))
                    residual_log_eq_first_order = log_SO3(residual_R_eq_first_order)
                    error_log_eq_first_order = torch.linalg.vector_norm(vee(residual_log_eq_first_order, algebra_type='so3'), dim=1)
                    error_fro_eq_first_order = torch.linalg.matrix_norm(residual_R_eq_first_order - torch.eye(3).to(device))
                    error_fro_eq_first_order_conj_sum += error_fro_eq_first_order.sum().item()
                    error_log_eq_first_order_conj_sum += error_log_eq_first_order.sum().item()

                    # compute bch second order approx
                    residual_R_eq_second_order = R1 @ R2 @ exp_so3(-BCH_second_order_approx(hat_layer(x_conj_j[:, 0, :, 0]),hat_layer(x_conj_j[:, 1, :, 0])))
                    residual_log_eq_second_order = log_SO3(residual_R_eq_second_order)
                    error_log_eq_second_order = torch.linalg.vector_norm(vee(residual_log_eq_second_order, algebra_type='so3'), dim=1)
                    error_fro_eq_second_order = torch.linalg.matrix_norm(residual_R_eq_second_order - torch.eye(3).to(device))
                    error_fro_eq_second_order_conj_sum += error_fro_eq_second_order.sum().item()
                    error_log_eq_second_order_conj_sum += error_log_eq_second_order.sum().item()

                    # compute bch third order approx
                    residual_R_eq_third_order = R1 @ R2 @ exp_so3(-BCH_third_order_approx(hat_layer(x_conj_j[:, 0, :, 0]),hat_layer(x_conj_j[:, 1, :, 0])))
                    residual_log_eq_third_order = log_SO3(residual_R_eq_third_order)
                    error_log_eq_third_order = torch.linalg.vector_norm(vee(residual_log_eq_third_order, algebra_type='so3'), dim=1)
                    error_fro_eq_third_order = torch.linalg.matrix_norm(residual_R_eq_third_order - torch.eye(3).to(device))
                    error_fro_eq_third_order_conj_sum += error_fro_eq_third_order.sum().item()
                    error_log_eq_third_order_conj_sum += error_log_eq_third_order.sum().item()

                # conj_output_hat_conj_back = torch.matmul(R_j.transpose(1,2), torch.matmul(conj_output_hat, R_j))
                # conj_output_conj_back = vee_so3(conj_output_hat_conj_back)

                # y_hat = hat_layer(y)
                # conj_y_hat = torch.matmul(H_j, torch.matmul(y_hat, torch.inverse(H_j)))
                # conj_y = vee_sl3(conj_y_hat)
                # loss = criterion(conj_output_conj_back, y)
                # print("conj_out", conj_output[0,:])
                # print("y_conj", y_conj[0, j, :])
                # print("diff_out_out_conj", diff_output[0,:])
                # print("out",output_x[0,:])
                # print("y", y[0,:])
                # print("diff_out",output_x[0,:] - y[0,:])
                # print(loss.item())
                # print("----------------------")
                # loss_sum += loss.item()

                # print('diff_output: ', diff_output)
        if config['calculate_eq_approx_error']:
            print("first order approx error fro avg : ",error_fro_eq_first_order_sum/total_num)
            print("first order approx error log avg : ",error_log_eq_first_order_sum/total_num)
            print("second order approx error fro avg : ",error_fro_eq_second_order_sum/total_num)
            print("second order approx error log avg : ",error_log_eq_second_order_sum/total_num)
            print("third order approx error fro avg : ",error_fro_eq_third_order_sum/total_num)
            print("third order approx error log avg : ",error_log_eq_third_order_sum/total_num)
            print("first order approx error fro conj avg : ",error_fro_eq_first_order_conj_sum/total_num_conj)
            print("first order approx error log conj avg : ",error_log_eq_first_order_conj_sum/total_num_conj)
            print("second order approx error fro conj avg : ",error_fro_eq_second_order_conj_sum/total_num_conj)
            print("second order approx error log conj avg : ",error_log_eq_second_order_conj_sum/total_num_conj)
            print("third order approx error fro conj avg : ",error_fro_eq_third_order_conj_sum/total_num_conj)
            print("third order approx error log conj avg : ",error_log_eq_third_order_conj_sum/total_num_conj)

        error_fro_avg = error_fro_sum/total_num
        error_log_avg = error_log_sum/total_num
        error_fro_conj_avg = error_fro_conj_sum/total_num_conj
        error_log_conj_avg = error_log_conj_sum/total_num_conj

        diff_output_avg = diff_output_sum/total_num_conj/x_conj.shape[3]
        # loss_non_conj_avg = loss_non_conj_sum/len(test_loader)

    return error_fro_avg, error_log_avg, error_fro_conj_avg, error_log_conj_avg, diff_output_avg

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
    elif config['model_type'] == "EMLP":
        import emlp.nn.pytorch as emlpnn
        from emlp.reps import T,V
        from emlp.groups import SO
        G = SO(3)
        reps = 2*V(G)
        reps_out = V(G)
        class EMLPModel(nn.Module):
            def __init__(self):
                super(EMLPModel, self).__init__()
                self.model = emlpnn.EMLP(reps, reps_out, group=G, num_layers=3,ch=128)

            def forward(self, x):
                B,_,_,_ = x.shape
                x = torch.reshape(x, (B, -1))
                return self.model(x)
            
        model = EMLPModel().to(device)
    elif config['model_type'] == "e3nn_norm":
        import e3nn
        from experiment.so3_bch_baseline_layers import E3nnMLPNorm
        model = E3nnMLPNorm(invariant=False).to(device)
    elif config['model_type'] == "e3nn_s2grid":
        import e3nn
        from experiment.so3_bch_baseline_layers import E3nnMLPS2Grid
        model = E3nnMLPS2Grid(invariant=False).to(device)
    elif config['model_type'] == "VN_relu":
        model = SO3EquivariantVNReluLayers(2).to(device)

    # elif config['model_type'] == "LN_bracket_no_residual":
    #     model = SO3EquivariantBracketNoResidualConnectLayers(2).to(device)


    print("Using model: ", config['model_type'])
    print("total number of parameters: ", sum(p.numel() for p in model.parameters()))
    
    # for name, param in model.named_parameters():
    #     print(name, param.shape)

    # print("model weight before load ", model.model.network[0].weight)
    # model = SL3InvariantLayersTest(2).to(device)
    checkpoint = torch.load(config['model_path'])
    print(checkpoint['test loss'])
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    
    # print("model weight after load ", model.model.network[0].weight)

    # criterion = nn.MSELoss().to(device)
    error_fro_avg, error_log_avg, error_fro_conj_avg, error_log_conj_avg, diff_output_avg = test_bch(model, test_loader, config, device)
    # test_loss_equiv, loss_non_conj_avg, diff_output_avg = test_bch(model, test_loader, config, device)
    print("error fro avg: ", error_fro_avg)
    print("error log avg: ", error_log_avg)
    print("error fro conj avg: ", error_fro_conj_avg)
    print("error log conj avg: ", error_log_conj_avg)

    # print("test_loss type:",type(test_loss_equiv))
    # # print("avg diff output type: ", diff_output_avg.dtype)
    # print("test with augmentation loss: ", test_loss_equiv)
    print("avg diff output: ", diff_output_avg)
    # print("loss non conj avg: ", loss_non_conj_avg)

if __name__ == "__main__":
    main()
