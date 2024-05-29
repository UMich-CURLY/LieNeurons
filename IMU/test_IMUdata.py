import torch
import os, sys
import matplotlib.pyplot as plt

from dataload import IMUDataset

from Plot_function import vec3d_31_plotter

sys.path.append(os.path.dirname(os.getcwd()))


# print(os.path.dirname(os.getcwd()))

data_type = 'V2_02_medium'
data_type = 'V2_01_easy'

IMUdata = IMUDataset(csv_file="data/"+data_type+"/mav0/imu0/data.csv", yaml_file="data/"+data_type+"/mav0/imu0/sensor.yaml")
IMUdata.set_init_time_to_zero()

myplotter = vec3d_31_plotter()
acc_batch = IMUdata[0:1000]

myplotter.pltvec3d(acc_batch[:,0],acc_batch[:,4:7])

from IMU_layers import Acc_model_LN_1, Acc_model_mlp_1, Acc_model_mlp_2, model_choose

model_type = 'Acc_model_LN_1'

model = Acc_model_LN_1().to('cuda')
model.load_state_dict(torch.load("weights/IMU_Dynamics/acc_only/"+ model_type + "/" +'_best_val_loss_acc.pt')['model'])
with torch.no_grad():
    acc_delta = model(acc_batch[:,4:7].unsqueeze_(1).to('cuda'))
    acc_delta = acc_delta.squeeze(1)
    print(acc_delta.shape)
    acc_denoised = acc_batch[:,4:7] + acc_delta.cpu()


myplotter.addvec3d(acc_batch[:,0],acc_denoised)
plt.show()


# plt.subplot(3,1,1)
# plt.plot(acc_batch[:,0],acc_batch[:,4])
# plt.plot(acc_batch[:,0],acc_denoised[:,0].cpu().numpy())
# plt.subplot(3,1,2)
# plt.plot(acc_batch[:,0],acc_batch[:,5])
# plt.plot(acc_batch[:,0],acc_denoised[:,1].cpu().numpy())
# plt.subplot(3,1,3)
# plt.plot(acc_batch[:,0],acc_batch[:,6])
# plt.plot(acc_batch[:,0],acc_denoised[:,2].cpu().numpy())
# plt.show()





