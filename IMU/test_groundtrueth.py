import torch
import matplotlib.pyplot as plt

from IMU.dataload import GroudtruthDataset, IMUDataset, align_time

gtdata = GroudtruthDataset(csv_file="data/MH_02_easy/mav0/state_groundtruth_estimate0/data.csv")
IMUdata = IMUDataset(csv_file="data/MH_02_easy/mav0/imu0/data.csv", yaml_file="data/MH_02_easy/mav0/imu0/sensor.yaml")

IMUdata, gtdata = align_time(IMUdata, gtdata)


device = 'cpu'

plot_len = 10000

IMU_torch = IMUdata[:plot_len].to(device)
gt_torch = gtdata[:plot_len].to(device)

fig, ax = plt.subplots(3,1)
ax[0].plot(gt_torch[:,0], gt_torch[:,8], label='vx')
ax[1].plot(gt_torch[:,0], gt_torch[:,9], label='vy')
ax[2].plot(gt_torch[:,0], gt_torch[:,10], label='vz')
plt.show()



# gtdata = GroudtruthDataset(csv_file="data/MH_02_easy/mav0/state_groundtruth_estimate0/data.csv")
# gtdata.convert_time_ns_to_s(set_init_time_to_zero=True)

# gtdatanp = gtdata.get_data()
# plot_len = 10000

# fig, ax = plt.subplots(3,1)
# ax[0].plot(gtdatanp[:plot_len,0], gtdatanp[:plot_len,1], label='px')
# ax[1].plot(gtdatanp[:plot_len,0], gtdatanp[:plot_len,2], label='py')
# ax[2].plot(gtdatanp[:plot_len,0], gtdatanp[:plot_len,3], label='pz')

# ax[0].set_title('Position')
# ax[2].set_xlabel('time (s)')
# ax[0].set_ylabel('px')
# ax[0].legend()

# fig.tight_layout()
# plt.show(block=False)



