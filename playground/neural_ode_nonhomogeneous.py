import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import lti
from scipy.signal import lsim2
from scipy import interpolate
from random import randint
import matplotlib.pyplot as plt
from torchdiffeq import odeint                       # modified version

parser = argparse.ArgumentParser('Pendel')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=300)
parser.add_argument('--test_freq', type=int, default=20)
args = parser.parse_args()

# initial conditions / system dynamics 
x0 = [0]   
# definition of the LTI-system
A = np.array([0.05])
B = np.array([0.05])             # x' = 0.05 x + 0.05 u
C = np.array([1])               
D = np.array([0])                # y = x
system = lti(A, B, C, D)   
     
t = torch.linspace(0., 25., args.data_size)
u = torch.Tensor(np.ones_like(t))                       # external input -- validation
tout, y, x = lsim2(system, u, t, x0)
true_y = torch.FloatTensor(y)
ufunc_val = interpolate.interp1d(t, u.reshape(-1,1), kind='linear', axis=0, bounds_error=False, fill_value='extrapolate')
y0 = torch.FloatTensor(x0)

def get_batch():
    u_in = torch.zeros(args.batch_time,args.batch_size)
    true_x = torch.zeros(args.batch_time,args.batch_size)
    batch_t = t[:args.batch_time]  # (T)
    for i in range(args.batch_size):        
        s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), 1, replace=False))
        u = torch.ones(args.batch_time)+randint(0,10)/5     # external input -- training
        tout, y2, x = lsim2(system, u, batch_t, y[s])
        tout = torch.tensor(tout)
        x = torch.tensor(x)
        u_in[:,i] = u.reshape(1, args.batch_time)
        true_x[:,i] = x.reshape(1, args.batch_time)  
    batch_x0 = true_x[0,:]
    return  batch_x0, batch_t, true_x, u_in


def visualize(true_x, pred_x):
    plt.title('Federpendel')
    plt.xlabel('t')
    plt.ylabel('x,v')    
    plt.plot(t.numpy(), true_x.numpy())
    plt.plot(t.numpy(), pred_x.numpy(),'--')
    plt.xlim(t.min(), t.max())
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)

class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()
        self.linear = nn.Linear(2,1)
       
    def forward(self, t, x, args):
        ufun = args[0]
        unew = torch.FloatTensor(ufun(t.detach().numpy()))
        in1 = torch.stack((x, unew), dim=1)
        out = self.linear(in1).squeeze()
        return out

    
if __name__ == '__main__':

    func = ODEFunc()
    optimizer = optim.RMSprop(func.parameters(), lr=0.02)
    lossb = nn.MSELoss()
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_x0, batch_t, batch_x, batch_u = get_batch()
        ufunc = interpolate.interp1d(batch_t, batch_u, kind='linear', axis=0, bounds_error=False, fill_value='extrapolate')
        pred_x = odeint(func, batch_x0, batch_t, args=(ufunc,), method='dopri5')
        loss = lossb(pred_x, batch_x)
        loss.backward()
        optimizer.step()
        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_x = odeint(func, y0, t, args=(ufunc_val,), method='dopri5')
                loss = lossb(pred_x.squeeze(), true_y.squeeze())
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                visualize(true_y.squeeze(), pred_x.squeeze())