import torch


from Rigid_body import *
from IMU.lie_algebra import SEn3exp,SEn3leftJaco_inv

if __name__=="__main__":
    torch.manual_seed(0)
    tw1 = screw_axis_from_joint(torch.tensor([1., 0, 0]), torch.rand(3), 'revolute')
    tw2 = screw_axis_from_joint(torch.tensor([0., 1, 0]), torch.rand(3), 'revolute')
    tw3 = screw_axis_from_joint(torch.tensor([0., 0, 1]), torch.rand(3), 'revolute')
    tw4 = screw_axis_from_joint(torch.tensor([1., 0, 0]), torch.rand(3), 'revolute')
    tw5 = screw_axis_from_joint(torch.tensor([0., 1, 0]), torch.rand(3), 'revolute')
    tw6 = screw_axis_from_joint(torch.tensor([0., 0, 1]), torch.rand(3), 'revolute')
    tw_all = torch.cat((tw1.unsqueeze(1), tw2.unsqueeze(1), tw3.unsqueeze(1), tw4.unsqueeze(1), tw5.unsqueeze(1), tw6.unsqueeze(1)), dim=1)

    func_theta = lambda t: 0.1 * torch.tensor([torch.sin(t),torch.sin(t),torch.sin(t),torch.sin(t),torch.sin(t),torch.sin(t)])
    func_theta_dot = lambda t:0.1 * torch.tensor([torch.cos(t),torch.cos(t),torch.cos(t),torch.cos(t),torch.cos(t),torch.cos(t)])

    gst0 = torch.eye(4)
    gst0[:3, :3] = SO3exp_from_unit_vec(torch.tensor([1., 0, 0]), torch.tensor(0.5))
    gst0[:3, 3] = torch.tensor([1., 2, 3])
    
    Vb = lambda t: Body_Jacobian(tw_all, func_theta(t), gst0) @ func_theta_dot(t)
    Vs = lambda t: Adjoint_from_SE3(forward_kinematics(tw_all, func_theta(t), gst0)) @ Vb(t)

    def func_ode(t, y):
        Vb = Body_Jacobian(tw_all, func_theta(t), gst0) @ func_theta_dot(t)
        return SEn3leftJaco_inv(y) @ Vb
    
    y0 = torch.zeros(6)

    t_check = torch.tensor([1.5])
    theta_at_tcheck = func_theta(t_check)
    print("theta_at_tcheck: \n", theta_at_tcheck)
    SE3_true = forward_kinematics(tw_all, theta_at_tcheck, gst0)
    print("SE3_true: \n", SE3_true)

    temp = func_ode(torch.tensor([t_check]), y0)
    print(temp)

    ## Euler integration
    t_check_float = float(t_check)
    h = 1e-2
    gst = gst0
    for t in torch.arange(0, t_check_float, h):
        # print(t)
        gst = gst @ SEn3exp(h * (Vb(t + h/2)) )
    print("numerical integration: Right_mul\n", gst)

    gst = gst0
    for t in torch.arange(0, t_check_float, h):
        # print(t)
        gst = SEn3exp(h * (Vs(t + h/2)) ) @ gst
    print("numerical integration: Left_mul\n", gst)

    # from torchdiffeq import odeint
    # t = torch.tensor([0,0.005])
    # y = odeint(func_ode, y0, t)
    # print(y)


    



    pass