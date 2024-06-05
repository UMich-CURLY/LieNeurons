import torch


if __name__ == "__main__":
    ## generate true data
    """angular velocity and SO3"""
    from Rigid_body import *
    torch.manual_seed(0)
    tw1 = screw_axis_from_joint(torch.tensor([1., 0, 0]), torch.rand(3), 'revolute')
    tw2 = screw_axis_from_joint(torch.tensor([0., 1, 0]), torch.rand(3), 'revolute')
    tw3 = screw_axis_from_joint(torch.tensor([0., 0, 1]), torch.rand(3), 'revolute')
    tw_all = torch.cat((tw1.unsqueeze(1), tw2.unsqueeze(1), tw3.unsqueeze(1)), dim=1)
    theta_all = lambda t: 0.1 * torch.tensor([torch.sin(t), torch.sin(t), torch.sin(t)])
    theta_all_dot = lambda t: 0.1 * torch.tensor([torch.cos(t), torch.cos(t), torch.cos(t)])

    gst0 = torch.eye(4)
    gst0[:3, :3] = SO3exp_from_unit_vec(torch.tensor([1., 0, 0]), torch.tensor(0.5))
    # gst0[:3, 3] = torch.tensor([1., 2, 3])

    wb_true = lambda t: (Body_Jacobian(tw_all, theta_all(t), gst0) @ theta_all_dot(t))[:3]
    R_true = lambda t: forward_kinematics(tw_all, theta_all(t), gst0)[:3, :3]

    # t_test = torch.tensor([0.5])
    # print("wb_true: \n", wb_true(t_test))
    # print("R_true: \n", R_true(t_test))

    """v, p and a"""
    p_true = lambda t: torch.tensor([torch.sin(t), torch.cos(t), torch.sin(t)])
    v_true = lambda t: torch.tensor([torch.cos(t), -torch.sin(t), torch.cos(t)])  
    dot_v_true = lambda t: torch.tensor([-torch.sin(t), -torch.cos(t), -torch.sin(t)])
    # \dot v = R a + g -> a = R^T (\dot v - g)
    g_const = torch.tensor([0, 0, -9.8])
    a_true = lambda t: R_true(t).T @ (dot_v_true(t) - g_const)

    """SE_2(3) X"""
    def X_true(t):
        X = torch.eye(5)
        X[:3, :3] = R_true(t)
        X[:3, 3] = v_true(t)
        X[:3, 4] = p_true(t)
        return X

    ## numerical integration
    
    from IMU.lie_algebra import *
    t0 = torch.tensor([0.])
    X0 = X_true(t0)
    print("X0: \n", X0)

    def func_ode_Omega(t,xi, X0):
        J_l_inv = SEn3leftJaco_inv(xi)
        Xt = X0 @ SEn3exp(xi)
        Rt = Xt[:3, :3]
        vt = Xt[:3, 3]
        A_tangent = torch.zeros(5,5)
        A_tangent[:3, :3] = so3hat(wb_true(t))
        A_tangent[:3, 3] = a_true(t) + Rt.transpose(-1,-2) @ g_const
        A_tangent[:3, 4] = Rt.transpose(-1,-2) @ vt
        return J_l_inv @ sen3vee(A_tangent)

    from RK_utilities import rk4_step_increament
    t_end = 1.5
    h = 1e-2
    X_pred = X0.clone()
    for t in torch.arange(0, t_end, h):
        xi0 = torch.zeros(9)
        func = lambda t, xi: func_ode_Omega(t, xi, X_pred)
        Delta_Omage = rk4_step_increament(func, t, xi0, h)
        X_pred = X_pred @ SEn3exp(Delta_Omage)
        # print("X_pred at t = ", t, ":\n", X_pred)
    print("X_pred: \n", X_pred)

    X_true_end = X_true(torch.tensor([t_end]))
    print("X_true_end: \n", X_true_end)





    pass