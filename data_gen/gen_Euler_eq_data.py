import sys  # nopep8
sys.path.append('.')  # nopep8

import torch
import torch.nn as nn
from torchdiffeq import odeint

from core.lie_alg_util import *

import inspect
import os

if __name__=="__main__":

    num_training = 20 # number of trajectories for training
    num_sample_everyTra = 1000
    T_end = 25.0

    save_pth_dir = "data/Euler_WithInput/"
    save_file_name_train = "Euler_WithInput_train_muliple_include_test.pt"
    save_file_name_test = "Euler_WithInput_test_single.pt" # only the first trajectory for testing
    save_path_train = os.path.join(save_pth_dir, save_file_name_train)
    save_path_test = os.path.join(save_pth_dir, save_file_name_test)
    if not os.path.exists(save_pth_dir):
        os.makedirs(save_pth_dir)

    
    inertia_type = 'iss' # 'iss' or 'model1'

    device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(5566)

    if num_training >1:
        training_y0 = torch.rand((num_training, 3)).to(device)
        training_y0[0,:] = torch.tensor([[2., 1.,3.0]]).to(device)
    else:
        training_y0 = torch.tensor([[2., 1.,3.0]]).to(device)

    
    t = torch.linspace(0., T_end, num_sample_everyTra).to(device)
    Dt = t[1] - t[0]
    ufunc = lambda t: torch.tensor([[torch.sin(t),torch.cos(t),torch.sin(t)]]).to(device)
    
    def sin_or_cos_chosen():
        if torch.rand(1) > 0.5:
            return torch.sin
        else:
            return torch.cos


    class EulerPoincareEquation(nn.Module):
        
        def __init__(self,ufunc) -> None:
            super().__init__()
            '''
            Inertia matrix of the ISS
            https://athena.ecs.csus.edu/~grandajj/ME296M/space.pdf
            page 7-62
            '''
            if inertia_type == 'iss':
                self.I = torch.Tensor([[5410880., -246595., 2967671.],[-246595., 29457838., -47804.],[2967671., -47804., 26744180.]]).unsqueeze(0).to(device)
            elif inertia_type == 'model1':
                self.I = torch.Tensor([[12, -5., 7.],[-5., 20., -2.],[7., -2., 5.]]).unsqueeze(0).to(device)
            self.I_inv = torch.inverse(self.I)
            self.hat_layer = HatLayer(algebra_type='so3').to(device)
            self.ufunc = ufunc

        def forward(self,t,w):
            '''
            w: angular velocity (B,3) or (1,3)
            '''
            w_v = w.unsqueeze(2)

            return -torch.matmul(self.I_inv,torch.matmul(self.hat_layer(w),torch.matmul(self.I,w_v))).squeeze(2)\
                    +torch.matmul(self.I_inv,self.ufunc(t).squeeze(0))

        def func_update(self, ufunc):
            self.ufunc = ufunc

    ODEfunc = EulerPoincareEquation(ufunc).to(device)

    with torch.no_grad():
        training_y = []
        training_u = []
        ufunc_str = [] 
        for i in range(num_training):
            true_y = odeint(ODEfunc, training_y0[i,:].unsqueeze(0), t, method='dopri5')
            training_y.append(true_y)  
            if i == 0:
                ufunc_str.append(inspect.getsource(ufunc).strip()) # save the str of the lambda function
            else:
                func_ux = sin_or_cos_chosen()
                func_uy = sin_or_cos_chosen()
                func_uz = sin_or_cos_chosen()
                coff_ux = torch.rand(1).to(device) * 10.0
                coff_uy = torch.rand(1).to(device) * 10.0
                coff_uz = torch.rand(1).to(device) * 10.0
                ufunc = lambda t: torch.tensor([[coff_ux * func_ux(t), coff_uy * func_uy(t), coff_uz * func_uz(t)]]).to(device)
                str = f"lambda t: torch.tensor([[{coff_ux.item():.4f} * {func_ux.__name__}(t), {coff_uy.item():.4f} * {func_uy.__name__}(t), {coff_uz.item():.4f} * {func_uz.__name__}(t)]]).to(device)"
                ufunc_str.append(str)
            print("ufunc_str in i = ", i, " : ", ufunc_str[-1])
            true_u = torch.stack([ufunc(t[i]) for i in range(t.shape[0])], dim=0).to(device)
            training_u.append(true_u)

    data_train = {}
    data_train['y'] = training_y    # list num = num_traj,  tensor_size =  [num_sample, 1, 3]
    data_train['u'] = training_u    # list num = num_traj,  tensor_size =  [num_sample, 1, 3]
    data_train['t'] = t         # tensor_size =  [num_sample]
    data_train['y0'] = training_y0  # tensor_size =  [num_traj, 3]
    data_train['ufunc_str'] = ufunc_str # list num = num_traj, str of the lambda function


    torch.save(data_train, save_path_train)

    ## Testing data for single trajectory
    testing_y0 = torch.zeros((5,3)).to(device)
    with torch.no_grad():
        y_testing_signle = []
        u_tseting_single = []

        # same with first Trajectory
        true_y = training_y[0]
        y_testing_signle.append(true_y)
        true_u = training_u[0]
        u_tseting_single.append(true_u)
        testing_y0[0,:] = training_y0[0,:]

        # Compare with the first Trajectory: different y0 same ut
        y0 = torch.rand((1, 3)).to(device)
        true_y = odeint(ODEfunc, y0, t, method='dopri5')
        y_testing_signle.append(true_y)
        true_u = training_u[0]
        u_tseting_single.append(true_u)
        testing_y0[1,:] = y0

        # Compare with the first Trajectory: same y0 different ut
        y0 = training_y0[0,:].unsqueeze(0)
        funcu = lambda t: 2. * torch.tensor([[torch.cos(t),torch.sin(t),torch.cos(t)]]).to(device)
        ODEfunc.func_update(funcu)
        true_y = odeint(ODEfunc, y0, t, method='dopri5')
        y_testing_signle.append(true_y)
        true_u = torch.stack([funcu(t[i]) for i in range(t.shape[0])], dim=0).to(device)
        u_tseting_single.append(true_u)
        testing_y0[2,:] = y0

        # Compare with the first Trajectory: different y0 different ut
        y0 = torch.rand((1, 3)).to(device)
        funcu = lambda t: 2. * torch.tensor([[torch.cos(t),torch.sin(t),torch.cos(t)]]).to(device)
        ODEfunc.func_update(funcu)
        true_y = odeint(ODEfunc, y0, t, method='dopri5')
        y_testing_signle.append(true_y)
        true_u = torch.stack([funcu(t[i]) for i in range(t.shape[0])], dim=0).to(device)
        u_tseting_single.append(true_u)
        testing_y0[3,:] = y0

        # Compare with the first Trajectory: R * y0, R * ut
        v_rand = torch.rand((1,3)).to(device)
        v_rand_skew = HatLayer(algebra_type='so3').to(device)(v_rand)
        R = torch.linalg.matrix_exp(v_rand_skew.squeeze(0))
        funcu = lambda t: torch.tensor([[torch.sin(t),torch.cos(t),torch.sin(t)]]).to(device) @ R.T
        ODEfunc.func_update(funcu)
        y0 = training_y0[0,:].unsqueeze(0) @ R.T
        true_y = odeint(ODEfunc, y0, t, method='dopri5')
        y_testing_signle.append(true_y)
        true_u = torch.stack([funcu(t[i]) for i in range(t.shape[0])], dim=0).to(device)
        u_tseting_single.append(true_u)
        testing_y0[4,:] = y0

    
    data_test = {}
    data_test['y'] = y_testing_signle    # list num = num_traj,  tensor_size =  [num_sample, 1, 3]
    data_test['u'] = u_tseting_single    # list num = num_traj,  tensor_size =  [num_sample, 1, 3]
    data_test['t'] = t         # tensor_size =  [num_sample]
    data_test['y0'] = testing_y0  # tensor_size =  [num_traj, 3]

    torch.save(data_test, save_path_test)
    print("Data saved in ", save_pth_dir)


    

