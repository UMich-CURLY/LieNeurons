import torch


## general parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
niter = 1000


## load data 

## define model
class angular_vel_curve(torch.nn.Module):
    def __init__(self):
        super(angular_vel_curve, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 3)
        )

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0, std=0.1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)

## define loss
def LieGroup_integration(func, t, R0 = torch.eye(3), device = 'cpu'):
    ## t is a tensor of time points to integrate to
    R0 = R0.to(device)
    pass

## get batch

## train loop

optimizer = torch.optim.Adam(angular_vel_curve.parameters(), lr=1e-3)

for i in range(niter):

    ## get batch

    ## forward and loss
    loss = LieGroup_integration()

    ## backward

    ## update

    ## validation

    ## save model

    ## log
    pass

