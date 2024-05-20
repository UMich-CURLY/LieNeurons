import torch
import torch.nn as nn

class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y: torch.Tensor, u: torch.Tensor, Dt):
        ii = int(t // Dt)
        if ii >= len(u) - 1:
            ii = len(u) - 1
            u_t_interpolated = u[ii]
        else:
            u_t_interpolated = u[ii] + (t - ii * Dt) * (u[ii + 1] - u[ii]) / Dt
        # print("y.shape: ", y.shape)
        # print("u.shape: ", u.shape)
        # print("u_t_interpolated.shape: ", u_t_interpolated.shape)
        return self.net(torch.cat((y**3, u_t_interpolated), dim=y.dim() - 1))