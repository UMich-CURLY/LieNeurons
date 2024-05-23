import torch
import numpy as np
import matplotlib.pyplot as plt
from IMU.dataload import IMUDataset, GroudtruethDataset, align_time

from Plot_function import plt3dvec
from torch.utils.data import DataLoader
from torchdiffeq import odeint
from quat_func import batch_slerp, quaternion_apply_vec

import os

from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_save_path = "weights/IMU_Dynamics/acc_only/"
log_writer_path = "logs/IMU_Dynamics/acc_only/"
train_batch_time_series = 20
train_batch_size = 10
niter = 2000
val_freq = 10
save_freq = 100


def interpolate_u(t, u, Dt):
    """
    Warning, when t is out of the range of u, the function will return the last element of u
    """
    ii = int(t // Dt)
    if ii >= len(u) - 1:
        ii = len(u) - 1
        u_t_interpolated = u[ii]
    else:
        u_t_interpolated = u[ii] + (t - ii * Dt) * (u[ii + 1] - u[ii]) / Dt
    return u_t_interpolated

def quat_interpolation(t, quat, Dt):
    """
    quat: [time_series, batch_size, 4] 
    """
    ii = int(t // Dt)
    if ii >= quat.shape[0] - 1:
        ii = quat.shape[0] - 1
        quat_t_interpolated = quat[ii]
    else:
        # keep the shape of quat as [batch_size, 4] even if batch_size = 1
        t_inter = torch.tensor([t - ii * Dt]).to(device)
        quat_t_interpolated = batch_slerp(quat[ii], quat[ii + 1], t_inter) 
    return quat_t_interpolated

class accel_func(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3)
        )

        for m in self.net.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0, std=0.1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, y):
        return self.net(y)



class ODE_vp_func(torch.nn.Module):
    """
    d(v,p) = (Ra+g, v)
    """
    def __init__(self):
        super().__init__()
        self.g = torch.tensor([0, 0, -9.81]).to(device)

    def forward(self, t, y, u, quat, Dt):
        """
        y = [v, p],                 shape: batch_size * 6
        u = [a],                    shape: time_serire * batch_size * 3
        quat = [qw, qx, qy, qz]     shape: time_serire * batch_size * 4
        Dt: torch scalar aussme u and quat have the same Dt
        """
        qt = quat_interpolation(t, quat, Dt)    # shape: batch_size * 4
        ut = interpolate_u(t, u, Dt)            # shape: batch_size * 3
        Ra = quaternion_apply_vec(qt, ut)       # shape: batch_size * 3
        return torch.cat((Ra + self.g, y[:,0:3]), dim=-1).to(device)
    
def get_batch(IMUdata, GTdata, batch_time_series = 100, batch_size = 10,):
    """
    IMUdata: [time, w_x, w_y, w_z, a_x, a_y, a_z]
    GTdata: [time, px, py, pz, qw, qx, qy, qz, vx, vy, vz, bgx, bgy, bgz, bax, bay, baz]
    """
    length = min(len(IMUdata), len(GTdata))
    s = torch.from_numpy(np.random.choice(np.arange(length - batch_time_series, dtype=np.int64), batch_size, replace=False))
    
    IMU_batch = torch.stack([IMUdata[s + i] for i in range(batch_time_series)], dim=0).to(device) # shape: [batch_time_series, batch_size, 7]
    GT_batch = torch.stack([GTdata[s + i] for i in range(batch_time_series)], dim=0).to(device) # shape: [batch_time_series, batch_size, 17]
    """
    can only use list to load the data, IMUdata is not a tensor, see IMUDataset, same for GTdata
    """

    batch_time = IMU_batch[:,0,0].to(device) # assume the time of IMU and GT are the same, and all Dt are the same
    batch_time = batch_time - batch_time[0] 

    batch_y0 = torch.cat((GT_batch[0,:,8:11], GT_batch[0,:,1:4]), dim=-1).to(device)
    batch_y = torch.cat((GT_batch[:,:,8:11], GT_batch[:,:,1:4]), dim=-1).to(device)
    batch_quat = GT_batch[:,:,4:8].to(device)
    batch_a = IMU_batch[:,:,4:7].to(device)
    
    return batch_time, batch_y0, batch_y, batch_a, batch_quat
    
def get_validation_batch(IMUdata, GTdata, batch_time_series = 100, batch_size = 10):
    length = min(len(IMUdata), len(GTdata))
    interval = round(length // batch_size)
    s = torch.arange(0, interval * batch_size, interval)

    IMU_batch = torch.stack([IMUdata[s + i] for i in range(batch_time_series)], dim=0).to(device) # shape: [batch_time_series, batch_size, 7]
    GT_batch = torch.stack([GTdata[s + i] for i in range(batch_time_series)], dim=0).to(device) # shape: [batch_time_series, batch_size, 17]

    val_batch_time = IMU_batch[:,0,0].to(device) # assume the time of IMU and GT are the same, and all Dt are the same
    val_batch_time = val_batch_time - val_batch_time[0]

    val_batch_y0 = torch.cat((GT_batch[0,:,8:11], GT_batch[0,:,1:4]), dim=-1).to(device)
    val_batch_y = torch.cat((GT_batch[:,:,8:11], GT_batch[:,:,1:4]), dim=-1).to(device)
    val_batch_quat = GT_batch[:,:,4:8].to(device)
    val_batch_a = IMU_batch[:,:,4:7].to(device)

    return val_batch_time, val_batch_y0, val_batch_y, val_batch_a, val_batch_quat


def init_writer():
    writer = SummaryWriter(log_writer_path)
    writer.add_text("num_iterations: ", str(niter))
    return writer

    

if __name__ == "__main__":
    ## load data
    IMUdata = IMUDataset(csv_file="data/V2_01_easy/mav0/imu0/data.csv", yaml_file="data/V2_01_easy/mav0/imu0/sensor.yaml")
    """
    time, w_x, w_y, w_z, a_x, a_y, a_z
    """
    IMU_Dt = IMUdata.get_dt()
    IMU_Dt =torch.tensor(IMU_Dt).to(device)

    GTdata = GroudtruethDataset(csv_file="data/V2_01_easy/mav0/state_groundtruth_estimate0/data.csv")
    """
    time, px, py, pz, qw, qx, qy, qz, vx, vy, vz, bgx, bgy, bgz, bax, bay, baz
    Dt = 0.005 s
    """

    IMUdata, GTdata = align_time(IMUdata, GTdata)
    print("IMUdata length:", len(IMUdata),"   GTdata length:", len(GTdata))
    ##---------------------------------##
    ## log and model save
    writer = init_writer()
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
        print("create model save path")
    if os.path.exists(model_save_path + '_best_val_loss_acc.pt'):
        best_loss = torch.load(model_save_path + '_best_val_loss_acc.pt')['loss']
        print("model exists, best loss: ", best_loss)
    else:
        best_loss = float('inf')
        print("model does not exist, initialize best loss")
    break_flag = 0
    ##---------------------------------##
    acc_network = accel_func().to(device)
    vpODEfunc = ODE_vp_func().to(device)

    optimizer = torch.optim.Adam(list(vpODEfunc.parameters()) + list(acc_network.parameters()), lr=0.01)
    criterion = torch.nn.MSELoss()

    for iiter in range(niter):
        optimizer.zero_grad()

        batch_time, batch_y0, batch_y, batch_a, batch_quat = get_batch(IMUdata, GTdata, train_batch_time_series, train_batch_size)
        # print("batch_time.shape:", batch_time.shape)
        # print("batch_y0.shape:", batch_y0.shape)
        # print("batch_y.shape:", batch_y.shape)
        # print("batch_a.shape:", batch_a.shape)
        # print("batch_quat.shape:", batch_quat.shape)

        func_with_u = lambda t, y: vpODEfunc(t, y, acc_network(batch_a), batch_quat, IMU_Dt)

        pred_y = odeint(func_with_u, batch_y0, batch_time, method='dopri5').to(device)
        loss = criterion(pred_y, batch_y)
        loss.backward()

        optimizer.step()
        writer.add_scalar('training loss', loss.item(), iiter)
        
        if iiter % val_freq == 0:
            with torch.no_grad():
                val_batch_time, val_batch_y0, val_batch_y, val_batch_a, val_batch_quat = get_validation_batch(IMUdata, GTdata, batch_time_series=50, batch_size=5)
                func_with_u_val = lambda t, y: vpODEfunc(t, y, acc_network(val_batch_a), val_batch_quat, IMU_Dt)
                val_pred_y = odeint(func_with_u_val, val_batch_y0, val_batch_time, method='dopri5').to(device)
                val_loss = criterion(val_pred_y, val_batch_y)
                writer.add_scalar('validation loss', val_loss.item(), iiter)
                print(f"Iteration: {iiter}, validation loss: {val_loss.item()}")
                print("------------------------------")

                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save({'loss': best_loss, 'model': acc_network.state_dict()}, model_save_path + '_best_val_loss_acc.pt')
                else:
                    break_flag += 1
                    if break_flag > 20:
                        print("Early stopping")
                        break
                    
        if iiter % save_freq == 0:
            torch.save({'loss': loss, 'model': acc_network.state_dict()}, model_save_path + f"_{iiter}.pt")
                

    writer.close()
    print("Training finished, best loss: ", best_loss)



                



            



    # legth_IMU = len(IMUdata[:,0])
    # lenth_GT = len(GTdata[:,0])
    # print("length_IMU:", legth_IMU)
    # print("length_GT:", lenth_GT)

    # print(len(IMUdata[:,1]))
    # print(len(IMUdata[:,2]))

    # print(len(GTdata[:,1]))
    # print(len(GTdata[:,2]))


    # print("IMU_start_time:", IMUdata.data['time'][0])
    # print("IMU_end_time:", IMUdata.data['time'][legth_IMU-1])
    # print("GT_start_time:", GTdata.data['time'][0])
    # print("GT_end_time:", GTdata.data['time'][lenth_GT-1])

    # temp1 = legth_IMU * IMU_Dt
    # temp2 = lenth_GT * 0.005
    # print("temp1:", temp1)
    # print("temp2:", temp2)

    # IMU_batch, GT_batch = get_batch(IMUdata, GTdata, 100)
    # print("IMU_batch.shape:", IMU_batch.shape)
    # print("GT_batch.shape:", GT_batch.shape)

    # plt3dvec(IMUdata[:,0],IMUdata[:,4:7], "Acceleration",'ax', 'ay', 'az')

    # acc_network = accel_func().to(device)
    # ODEfunc = ODE_vp_func().to(device)

    # quat = GTdata[:100,4:8].to(device)
    # acc = IMUdata[:100,4:7].to(device)

    # func_with_u = lambda t, y: ODEfunc(t, y, acc_network(acc), IMU_Dt)

    # y0 = torch.cat((GTdata[0,1:4], GTdata[0,8:11]), dim=0).to(device)
    # t = GTdata[:100,0].to(device)

    # optimizer = torch.optim.Adam(list(ODEfunc.parameters()) + list(acc_network.parameters()), lr=0.01)
    # criterion = torch.nn.MSELoss()

    # for i in range(100):
    #     optimizer.zero_grad()
    #     pred_y = odeint(func_with_u, y0, t, method='dopri5').to(device)
    #     loss = criterion(pred_y, torch.cat((GTdata[:100,1:4], GTdata[:100,8:11]), dim=1).to(device))
    #     loss.backward()
    #     optimizer.step()
    #     print(f"loss: {loss.item()}")





    


    