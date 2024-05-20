import torch
import matplotlib.pyplot as plt

from IMU.dataload import IMUDataset

IMUdata = IMUDataset(csv_file="data/MH_02_easy/mav0/imu0/data.csv", yaml_file="data/MH_02_easy/mav0/imu0/sensor.yaml")
IMUdata.convert_time_ns_to_s(set_init_time_to_zero=True)

plot_len = 10000
fig, ax = plt.subplots(3,1)
ax[0].plot(IMUdata.get_data()[:plot_len,0], IMUdata.get_data()[:plot_len,1], label='w_x')
ax[1].plot(IMUdata.get_data()[:plot_len,0], IMUdata.get_data()[:plot_len,2], label='w_y')
ax[2].plot(IMUdata.get_data()[:plot_len,0], IMUdata.get_data()[:plot_len,3], label='w_z')

ax[0].set_title('Angular Velocity')
ax[0].set_xlabel('time (s)')
ax[0].set_ylabel('w_x')
ax[0].legend()

ax[1].set_xlabel('time (s)')
ax[1].set_ylabel('w_y')
ax[1].legend()

ax[2].set_xlabel('time (s)')
ax[2].set_ylabel('w_z')
ax[2].legend()
fig.tight_layout()
# fig.waitforbuttonpress()
plt.show(block=False)

fig, ax = plt.subplots(3,1)
ax[0].plot(IMUdata.get_data()[:plot_len,0], IMUdata.get_data()[:plot_len,4], label='a_x')
ax[1].plot(IMUdata.get_data()[:plot_len,0], IMUdata.get_data()[:plot_len,5], label='a_y')
ax[2].plot(IMUdata.get_data()[:plot_len,0], IMUdata.get_data()[:plot_len,6], label='a_z')

ax[0].set_title('Acceleration')
ax[0].set_xlabel('time (s)')
ax[0].set_ylabel('a_x')
ax[0].legend()

ax[1].set_xlabel('time (s)')
ax[1].set_ylabel('a_y')
ax[1].legend()

ax[2].set_xlabel('time (s)')
ax[2].set_ylabel('a_z')
ax[2].legend()

fig.tight_layout()
plt.show()