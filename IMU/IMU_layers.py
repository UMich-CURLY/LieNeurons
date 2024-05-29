import torch
import torch.nn as nn

from core.lie_neurons_layers import *
from core.lie_alg_util import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from quat_func import batch_slerp, quaternion_apply_vec

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
    

class Acc_model_linear(nn.Module):
    def __init__(self):
        super(Acc_model_linear, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 3)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)

class Acc_model_mlp_1(nn.Module):
    def __init__(self):
        super(Acc_model_mlp_1, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        return self.net(x)
    
class Acc_model_mlp_2(nn.Module):
    def __init__(self):
        super(Acc_model_mlp_2, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        return self.net(x)
    

class Acc_model_LN_1(nn.Module):
    def __init__(self, R=torch.eye(3)):
        super().__init__()

        self.num_m_feature = 20
        self.R = R.to(device)
        self.m = nn.Parameter(torch.zeros((3,self.num_m_feature))).to(device)
        self.map_m_to_m1 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.map_m_to_m2 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        
        self.ln_bracket = LNLinearAndLieBracketChannelMix(1,20,algebra_type='so3',residual_connect=False)
        
        self.linear = LNLinear(20,1)
        
        nn.init.normal_(self.m, mean=0, std=0.1)
        
    def set_R(self,R):
        self.R = R.to(self.R.device)
        
    def forward(self, x):
        x_reshape = rearrange(x,'t b d -> b 1 d t')
        
        m1 = self.map_m_to_m1(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m2 = self.map_m_to_m2(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        
        m1 = rearrange(m1,'1 f k 1 -> k f')
        m2 = rearrange(m2,'1 f k 1 -> k f')
        
        M1 = torch.matmul(m1,m1.transpose(0,1))
        M2 = torch.matmul(m2,m2.transpose(0,1))
        
    
        out = self.ln_bracket(x_reshape,M1,M2)  # [b f d 1]
        
        out = self.linear(out)
        out = rearrange(out,'b 1 d t -> t b d')
        return out
    
class Acc_model_LN1_plus_Linear(nn.Module):
    def __init__(self, R=torch.eye(3)):
        super().__init__()

        self.num_m_feature = 20
        self.R = R.to(device)
        self.m = nn.Parameter(torch.zeros((3,self.num_m_feature))).to(device)
        self.map_m_to_m1 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.map_m_to_m2 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        
        self.ln_bracket = LNLinearAndLieBracketChannelMix(1,20,algebra_type='so3',residual_connect=False)
        
        self.linear = LNLinear(20,1)
        
        nn.init.normal_(self.m, mean=0, std=0.1)

        self.fc1 = nn.Linear(3, 3)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc1.bias.data.fill_(0.0)
        
    def set_R(self,R):
        self.R = R.to(self.R.device)
        
    def forward(self, x):
        x_reshape = rearrange(x,'t b d -> b 1 d t')
        
        m1 = self.map_m_to_m1(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m2 = self.map_m_to_m2(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        
        m1 = rearrange(m1,'1 f k 1 -> k f')
        m2 = rearrange(m2,'1 f k 1 -> k f')
        
        M1 = torch.matmul(m1,m1.transpose(0,1))
        M2 = torch.matmul(m2,m2.transpose(0,1))
        
    
        out = self.ln_bracket(x_reshape,M1,M2)  # [b f d 1]
        
        out = self.linear(out)
        out = rearrange(out,'b 1 d t -> t b d')
        out = out + self.fc1(x)
        return out
    
def model_choose(model_type):
    match model_type:
        case "Acc_model_linear":
            return Acc_model_linear()
        case "Acc_model_mlp_1":
            return Acc_model_mlp_1()
        case "Acc_model_mlp_2":
            return Acc_model_mlp_2()
        case "Acc_model_LN_1":
            return Acc_model_LN_1()
        case "Acc_model_LN1_plus_Linear":
            return Acc_model_LN1_plus_Linear()
        case _:
            raise ValueError("model_type not found")
    
    

if __name__ == "__main__":
    xx = torch.rand(10,3)
    model_test = Acc_model_LN_1()
    y = model_test(xx)
    print("y.shape:", y.shape)


