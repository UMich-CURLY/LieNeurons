from scipy.spatial.transform import Rotation as scipy_rot
import numpy as np

# Create a quaternion
q  = scipy_rot.from_quat([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)]) # x, y, z, w

R1 = q.as_matrix()

dt = 1/200.0
w = np.array([1, 2, 3])
R2 = scipy_rot.from_rotvec(w*dt).as_matrix()
R3 = R1 @ R2

print("R3: \n", R3)

# q.apply()

temp = np.arange(0, 10, 1)
print(temp)

temp = np.arange(0, 10)
print(temp)

import torch
batch_time_series = 100
batch_size = 10
s = torch.arange(0,batch_time_series * batch_size, batch_time_series)
print(s)
