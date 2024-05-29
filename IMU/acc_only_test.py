import torch

from dataload import IMUDataset, GroudtruthDataset, align_time
from IMU_layers import *

import argparse

import sys

parser = argparse.ArgumentParser('OnlyAccTest')
parser.add_argument('--model_type', type=str, default='unspecified', help='specify the model type')
parser.add_argument('--data_type', type=str, default='unspecified', help='specify the data type')
args = parser.parse_args()

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

def get_testing_batch(IMUdata, GTdata, start_index = 0,time_len = 1000):

    IMU_test = torch.stack([IMUdata[start_index + i] for i in range(time_len)], dim=0).to(device)
    GT_test = torch.stack([GTdata[start_index + i] for i in range(time_len)], dim=0).to(device)

    time_test = IMU_test[:,0].to(device)
    time_test = time_test - time_test[0]
    acc_test = IMU_test[:,4:7].unsqueeze(1).to(device)

    quat_test = GT_test[:,4:8].unsqueeze(1).to(device)
    y_test = torch.cat((GT_test[...,8:11], GT_test[...,1:4]), dim=-1).unsqueeze(1).to(device)
    y0_test = y_test[0]
    return time_test, y0_test, y_test, acc_test, quat_test

if __name__ == "__main__":
    torch.manual_seed(256)
    ## Test the original Trajectory

    if args.model_type == 'unspecified':
        model_type = "Acc_model_LN_1"
    else:
        model_type = args.model_type

    if args.data_type == 'unspecified':
        data_type = "V2_01_easy"
    else:
        data_type = args.data_type

    print("model_type:", model_type)
    print("data_type:", data_type)

    fig_save_path = "figures/IMU_Dynamics/acc_only/" + data_type + "/" + model_type + "/"
    

    model = model_choose(model_type).to(device)
    func = ODE_vp_func().to(device)

    model.load_state_dict(torch.load("weights/IMU_Dynamics/acc_only/"+ model_type + "/" +'_best_val_loss_acc.pt')['model'])
    

    ## load data
    IMUdata = IMUDataset(csv_file="data/"+data_type+"/mav0/imu0/data.csv", yaml_file="data/"+data_type+"/mav0/imu0/sensor.yaml")
    """
    time, w_x, w_y, w_z, a_x, a_y, a_z
    """
    IMU_Dt = IMUdata.get_dt()
    IMU_Dt =torch.tensor(IMU_Dt).to(device)

    GTdata = GroudtruthDataset(csv_file="data/"+data_type+"/mav0/state_groundtruth_estimate0/data.csv")
    """
    time, px, py, pz, qw, qx, qy, qz, vx, vy, vz, bgx, bgy, bgz, bax, bay, baz
    Dt = 0.005 s
    """

    IMUdata, GTdata = align_time(IMUdata, GTdata)
    print("IMUdata length:", len(IMUdata),"   GTdata length:", len(GTdata))
    ##---------------------------------##

    from torchdiffeq import odeint
    from Plot_function import vec3d_31_plotter
    with torch.no_grad():
        start_index = 0
        time_test, y0_test, y_test, acc_test, quat_test = get_testing_batch(IMUdata, GTdata, start_index=start_index ,time_len = 5000)
        # print("start_index:", start_index)
        # print("time_test shape:", time_test.shape)
        # print("y0_test shape:", y0_test.shape)
        # print("y_test shape:", y_test.shape)
        # print("acc_test shape:", acc_test.shape)
        # print("quat_test shape:", quat_test.shape)
        ## no denoising
        func_with_u = lambda t,y: func(t,y,acc_test,quat_test,IMU_Dt)
        pred_y_from_ori = odeint(func_with_u, y0_test, time_test).to(device)
        print("pred_y shape:", pred_y_from_ori.shape)
        ## denoising
        func_with_u = lambda t,y: func(t,y,acc_test + model(acc_test),quat_test,IMU_Dt)
        pred_y_from_denoise = odeint(func_with_u, y0_test, time_test).to(device)
        print("pred_y shape:", pred_y_from_denoise.shape)

    ## Plot the result
    pred_y = pred_y_from_ori.squeeze(1)
    y_test = y_test.squeeze(1)
    pred_y_from_denoise = pred_y_from_denoise.squeeze(1)

    Vel_Plotter = vec3d_31_plotter(save_path=fig_save_path)
    Vel_Plotter.pltvec3d(time_test,y_test[...,0:3],title="Velocity",data_label_1="true_vx",data_label_2="true_vy",data_label_3="true_vz")
    Vel_Plotter.addvec3d(time_test,pred_y[...,0:3],data_label_1="no_denoise_vx",data_label_2="no_denoise_vy",data_label_3="no_denoise_vz")
    Vel_Plotter.addvec3d(time_test,pred_y_from_denoise[...,0:3],data_label_1="denoise_vx",data_label_2="denoise_vy",data_label_3="denoise_vz")
    Vel_Plotter.savefig("velocity_testing")

    Pos_Plotter = vec3d_31_plotter(save_path=fig_save_path)
    Pos_Plotter.pltvec3d(time_test,y_test[...,3:6],title="Position",data_label_1="true_px",data_label_2="true_py",data_label_3="true_pz")
    Pos_Plotter.addvec3d(time_test,pred_y[...,3:6],data_label_1="pred_px",data_label_2="pred_py",data_label_3="pred_pz")
    Pos_Plotter.addvec3d(time_test,pred_y_from_denoise[...,3:6],data_label_1="pred_denoise_px",data_label_2="pred_denoise_py",data_label_3="pred_denoise_pz")
    Pos_Plotter.savefig("position_testing")


    
    


    
