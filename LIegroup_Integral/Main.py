import torch


from Rigid_body import *

if __name__=="__main__":
    tw1 = screw_axis_from_joint(torch.tensor([1., 0, 0]), torch.tensor([1., 2, 3]), 'revolute')
    tw2 = screw_axis_from_joint(torch.tensor([0., 1, 0]), torch.tensor([4., 5, 6]), 'revolute')
    tw3 = screw_axis_from_joint(torch.tensor([0., 0, 1]), torch.tensor([7., 8, 9]), 'revolute')
    tw4 = screw_axis_from_joint(torch.tensor([1., 0, 0]), torch.tensor([10., 11, 12]), 'revolute')
    tw5 = screw_axis_from_joint(torch.tensor([0., 1, 0]), torch.tensor([13., 14, 15]), 'revolute')
    tw6 = screw_axis_from_joint(torch.tensor([0., 0, 1]), torch.tensor([16., 17, 18]), 'revolute')
    tw_all = torch.cat((tw1.unsqueeze(1), tw2.unsqueeze(1), tw3.unsqueeze(1), tw4.unsqueeze(1), tw5.unsqueeze(1), tw6.unsqueeze(1)), dim=1)

    func_theta = lambda t:torch.tensor([torch.sin(t),torch.cos(t),torch.sin(t),torch.cos(t),torch.sin(t),torch.cos(t)])
    func_theta_dot = lambda t:torch.tensor([torch.cos(t),-torch.sin(t),torch.cos(t),-torch.sin(t),torch.cos(t),-torch.sin(t)])

    gst0 = torch.eye(4)
    gst0[:3, :3] = SO3exp_from_unit_vec(torch.tensor([1., 1, 1]), torch.tensor(0.5))
    gst0[:3, 3] = torch.tensor([1., 2, 3])
    
    


    pass