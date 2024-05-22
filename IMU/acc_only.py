import torch
import matplotlib.pyplot as plt
from IMU.dataload import IMUDataset, GroudtruethDataset



IMUdata = IMUDataset(csv_file="data/V2_01_easy/mav0/imu0/data.csv", yaml_file="data/V2_01_easy/mav0/imu0/sensor.yaml")
"""
time, w_x, w_y, w_z, a_x, a_y, a_z
"""

time_all = IMUdata[:,0]
print("time_all.shape:", time_all.shape)

fig, ax = plt.subplots(3,1)
ax[0].plot(IMUdata[:,0], IMUdata[:,1], label='w_x')
ax[1].plot(IMUdata[:,0], IMUdata[:,2], label='w_y')
ax[2].plot(IMUdata[:,0], IMUdata[:,3], label='w_z')

ax[0].set_title('Angular Velocity')
ax[0].set_xlabel('time (s)')
ax[0].set_ylabel('w_x')
ax[1].set_ylabel('w_y')
ax[2].set_ylabel('w_z')
ax[0].legend()
plt.show()

GTdata = GroudtruethDataset(csv_file="data/V2_01_easy/mav0/state_groundtruth_estimate0/data.csv")
"""
time, px, py, pz, qw, qx, qy, qz, vx, vy, vz, bgx, bgy, bgz, bax, bay, baz
"""





# IMUdata.convert_time_ns_to_s(set_init_time_to_zero=True)

# gtdata = GroudtruethDataset(csv_file="data/V2_01_easy/mav0/state_groundtruth_estimate0/data.csv")
# gtdata.convert_time_ns_to_s(set_init_time_to_zero=True)

