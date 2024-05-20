import torch
import torch.nn as nn

# import quaternion

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda:' + str(0))
        print('Using GPU')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    """
    Use sin or cos function randomly to generate ut
    """
    torch.manual_seed(123)
    def sin_or_cos_chosen():
        if torch.rand(1) > 0.5:
            return torch.sin
        else:
            return torch.cos
        
    Dt = 1/200.0
    T_Tra = 10.0

    ## Generate true w and a
    coff_w = torch.rand(3).to(device) * 10.0
    phase_w_offset = torch.rand(3).to(device) * 2 * 3.1415926
    w_freq = torch.rand(3).to(device) * 5.0

    coff_a = torch.rand(3).to(device) * 10.0
    phase_a_offset = torch.rand(3).to(device) * 2 * 3.1415926
    a_freq = torch.rand(3).to(device) * 5.0
    
    func_w = lambda t :torch.tensor([[coff_w[0] * sin_or_cos_chosen()(w_freq[0]*t + phase_w_offset[0]),\
                                      coff_w[1] * sin_or_cos_chosen()(w_freq[1]*t + phase_w_offset[1]),\
                                      coff_w[2] * sin_or_cos_chosen()(w_freq[2]*t + phase_w_offset[2])]]).to(device)
    
    func_RTddotp = lambda t :torch.tensor([[coff_a[0] * sin_or_cos_chosen()(a_freq[0]*t + phase_a_offset[0]),\
                                      coff_a[1] * sin_or_cos_chosen()(a_freq[1]*t + phase_a_offset[1]),\
                                      coff_a[2] * sin_or_cos_chosen()(a_freq[2]*t + phase_a_offset[2])]]).to(device) 
    
    print(f"func_w: lambda t :torch.tensor([[{coff_w[0].item():.4f} * {sin_or_cos_chosen().__name__}({w_freq[0].item():.4f}*t + {phase_w_offset[0].item():.4f}),\
    {coff_w[1].item():.4f} * {sin_or_cos_chosen().__name__}({w_freq[1].item():.4f}*t + {phase_w_offset[1].item():.4f}),\
    {coff_w[2].item():.4f} * {sin_or_cos_chosen().__name__}({w_freq[2].item():.4f}*t + {phase_w_offset[2].item():.4f})]]).to(device)")
    
    print(f"func_a: lambda t :torch.tensor([[{coff_a[0].item():.4f} * {sin_or_cos_chosen().__name__}({a_freq[0].item():.4f}*t + {phase_a_offset[0].item():.4f}),\
    {coff_a[1].item():.4f} * {sin_or_cos_chosen().__name__}({a_freq[1].item():.4f}*t + {phase_a_offset[1].item():.4f}),\
    {coff_a[2].item():.4f} * {sin_or_cos_chosen().__name__}({a_freq[2].item():.4f}*t + {phase_a_offset[2].item():.4f})]]).to(device)")


    ## Generate true R
    
    # quaternion.integrate_angular_velocity(func_w,t0=0.0, t1=T_Tra)

    



    # class EulerPoincareEquation(nn.Module):
        
    #     def __init__(self,ufunc) -> None:
    #         super().__init__()
    #         '''
    #         Inertia matrix of the ISS
    #         https://athena.ecs.csus.edu/~grandajj/ME296M/space.pdf
    #         page 7-62
    #         '''
    #         if inertia_type == 'iss':
    #             self.I = torch.Tensor([[5410880., -246595., 2967671.],[-246595., 29457838., -47804.],[2967671., -47804., 26744180.]]).unsqueeze(0).to(device)
    #         elif inertia_type == 'model1':
    #             self.I = torch.Tensor([[12, -5., 7.],[-5., 20., -2.],[7., -2., 5.]]).unsqueeze(0).to(device)
    #         self.I_inv = torch.inverse(self.I)
    #         self.hat_layer = HatLayer(algebra_type='so3').to(device)
    #         self.ufunc = ufunc

    #     def forward(self,t,w):
    #         '''
    #         w: angular velocity (B,3) or (1,3)
    #         '''
    #         w_v = w.unsqueeze(2)

    #         return -torch.matmul(self.I_inv,torch.matmul(self.hat_layer(w),torch.matmul(self.I,w_v))).squeeze(2)\
    #                 +torch.matmul(self.I_inv,self.ufunc(t).squeeze(0))

    #     def func_update(self, ufunc):
    #         self.ufunc = ufunc


    
    
        