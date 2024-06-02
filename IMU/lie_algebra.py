import torch

def SO3exp(w: torch.Tensor, eps = 1e-6)->torch.Tensor:
    """w shape (..., 3)"""
    theta = torch.norm(w, dim = -1, keepdim = True)
    small_theta_mask = theta[...,0] < eps
    Identity = torch.eye(3).expand(w.shape[:-1]+(3,3)).to(w.device)

    unit_w = w[~small_theta_mask] / theta[~small_theta_mask]
    s = torch.sin(theta[~small_theta_mask]).unsqueeze(-1)
    c = torch.cos(theta[~small_theta_mask]).unsqueeze(-1)

    Rotation = torch.zeros_like(Identity)
    Rotation[small_theta_mask] = Identity[small_theta_mask] + so3hat(w[small_theta_mask])
    # outer product is used here (follow Timothy Barfoot formulation, not conventional Rodrigues formula), also used in code from M. Brossard
    Rotation[~small_theta_mask] = c * Identity[~small_theta_mask] + (1-c) * outer_product(unit_w, unit_w) + s * so3hat(unit_w)
    return Rotation

    
def outer_product(v1: torch.Tensor, v2: torch.Tensor)->torch.Tensor:
    """v1, v2 shape (..., 3)"""
    return torch.einsum('...i,...j->...ij', v1, v2)

def so3hat(w: torch.Tensor)->torch.Tensor:
    """w shape (..., 3)"""
    return torch.stack([torch.zeros_like(w[..., 0]), -w[..., 2], w[..., 1],
                        w[..., 2], torch.zeros_like(w[..., 0]), -w[..., 0],
                        -w[..., 1], w[..., 0], torch.zeros_like(w[..., 0])], dim=-1).reshape(w.shape[:-1]+(3,3))
    



def SO3leftJaco(phi: torch.Tensor, eps = 1e-6)->torch.Tensor:
    """left jacobian of SO(3), phi shape (..., 3)"""
    theta = torch.norm(phi, dim = -1, keepdim = True)
    small_theta_mask = theta[...,0] < eps
    Identity = torch.eye(3).expand(phi.shape[:-1]+(3,3)).to(phi.device)

    unit_phi = phi[~small_theta_mask] / theta[~small_theta_mask]
    sss = (torch.sin(theta[~small_theta_mask])/theta[~small_theta_mask]).unsqueeze(-1)
    ccc = ((1.0- torch.cos(theta[~small_theta_mask]))/theta[~small_theta_mask]).unsqueeze(-1)

    Jaco = torch.zeros_like(Identity)
    Jaco[small_theta_mask] = Identity[small_theta_mask] + 0.5 * so3hat(phi[small_theta_mask])
    Jaco[~small_theta_mask] = sss * Identity[~small_theta_mask] + (1.0 - sss) * outer_product(unit_phi, unit_phi) + ccc * so3hat(unit_phi)
    return Jaco

def SO3leftJacoInv(phi: torch.Tensor, eps = 1e-6)->torch.Tensor:
    """inverse of left jacobian of SO(3)"""
    theta = torch.norm(phi, dim = -1, keepdim = True)
    small_theta_mask = theta[...,0] < eps
    ## check singularity
    remaider = theta[~small_theta_mask] % (2 * torch.pi)
    assert torch.all( torch.min(remaider, torch.abs(2 * torch.pi - remaider)) > 1e-3 ), "theta should not be a multiple of 2pi"
    ## end check singularity
    Identity = torch.eye(3).expand(phi.shape[:-1]+(3,3)).to(phi.device)

    unit_phi = phi[~small_theta_mask] / theta[~small_theta_mask]
    sss_cot = (theta[~small_theta_mask]/(2.0 * torch.tan(theta[~small_theta_mask]/2.0))).unsqueeze(-1)

    Jaco = torch.zeros_like(Identity)
    Jaco[small_theta_mask] = Identity[small_theta_mask] - 0.5 * so3hat(phi[small_theta_mask])
    Jaco[~small_theta_mask] = sss_cot * Identity[~small_theta_mask] + (1.0 - sss_cot) * outer_product(unit_phi, unit_phi) - 0.5 * so3hat(phi[~small_theta_mask])
    return Jaco

def SEn3inverse(X: torch.Tensor, numerial_invese = False)->torch.Tensor:
    """inverse of SEn(3) matrix, X, shape (..., 3+n, 3+n) n = 1,2,3..."""
    if numerial_invese:
        X_inv = torch.linalg.inv(X)
    else:
        X_inv = X.clone()
        X_inv[..., :3, :3] = X[..., :3, :3].transpose(-2, -1)
        for i in range(3,X.shape[-1]):
            X_inv[..., :3, i] = - torch.matmul(X_inv[..., :3, :3], X[..., :3, i].unsqueeze(-1)).squeeze(-1)
    return X_inv

def SEn3leftJaco(xi: torch.Tensor, eps = 1e-6)->torch.Tensor:
    """left jacobian of SEn(3), phi shape (..., 3*n), n = 1,2,3..., Order: xi_R, xi_v, xi_p ..."""
    Jaco_left_SO3 = SO3leftJaco(xi[..., :3]) 


    # TODO: check singularity


    N = xi.shape[-1] // 3
    Jaco = torch.zeros(xi.shape[:-1]+(xi.shape[-1],xi.shape[-1])).to(xi.device)
    Jaco[..., :3, :3] = Jaco_left_SO3
    Ql = Ql_forSE3Jaco(xi)
    for i in range(1,N):
        Jaco[..., 3*i:3*(i+1), 3*i:3*(i+1)] = Jaco_left_SO3
        Jaco[..., 3*i:3*(i+1), :3] = torch.zeros_like(Jaco[..., 3*i:3*(i+1), :3])


    return Jaco


def Ql_forSE3Jaco(xi: torch.Tensor)->torch.Tensor:
    """Ql for SEn(3) left jacobian, phi shape (..., 3*n) Order: xi_R, xi_v, xi_p ..."""
    """Assume xi is free of singularity, this should be checked before calling this function"""
    N = round(xi.shape[-1] // 3)
    Ql = torch.zeros(xi.shape[:-1]+(xi.shape[-1]-3,3)).to(xi.device)
    phi = xi[..., :3]
    phi_wedge = so3hat(phi)

    theta = torch.norm(phi[..., :3], dim = -1, keepdim = True)
    theta2 = theta * theta
    theta3 = theta2 * theta
    theta4 = theta3 * theta
    theta5 = theta4 * theta

    s_theta = torch.sin(theta)
    c_theta = torch.cos(theta)

    m1 = 0.5
    m2 = (theta - s_theta) / theta3
    m3 = (theta2 + 2 * c_theta - 2) / (2 * theta4)
    m4 = (2*theta - 3*s_theta + theta * c_theta) / (2 * theta5)

    for i in range(N-1):
        nu_wedge = so3hat(xi[..., 3*(i+1):3*(i+2)])
        v1 =nu_wedge
        v2 = phi_wedge @ nu_wedge + nu_wedge @ phi_wedge + phi_wedge @ nu_wedge @ phi_wedge
        v3 = phi_wedge @ phi_wedge @ nu_wedge + nu_wedge @ phi_wedge @ phi_wedge - 3 * phi_wedge @ nu_wedge @ phi_wedge
        v4 = phi_wedge @ nu_wedge @ phi_wedge @ phi_wedge + phi_wedge @ phi_wedge @ nu_wedge @ phi_wedge
        Ql[..., 3*i:3*(i+1), :] = m1 * v1 + m2 * v2 + m3 * v3 + m4 * v4

    return Ql

def ad_sen3(xi: torch.Tensor)->torch.Tensor:
    """
    Compute the adjoint matrix from a SEn(3) matrix
    :param xi: sen3 lie algebra, shape (...,3*n) Order: xi_R, xi_v, xi_p ...
    :return: the adjoint matrix, shape (...,3*n,3*n)
    """
    adm = torch.zeros(xi.shape[:-1]+(xi.shape[-1],xi.shape[-1])).to(xi.device)
    N = round(xi.shape[-1] // 3)
    phi_wedge = so3hat(xi[..., :3])
    adm[..., :3, :3] = phi_wedge
    for i in range(1,N):
        adm[..., 3*i:3*(i+1), 3*i:3*(i+1)] = phi_wedge
        adm[..., 3*i:3*(i+1), :3] = so3hat(xi[..., 3*i:3*(i+1)])
    pass


if __name__ == '__main__':
    print("Test lie_algebra.py")
    print("---------------------------------")
    torch.set_default_dtype(torch.float64)

    ## test SO3exp
    device = 'cuda'
    w = torch.tensor([[torch.pi/3,0,0],[torch.pi/3,0,0],[0,torch.pi/3,0],[0,0,torch.pi/4]]).to(device).unsqueeze(0)
    w = w.repeat(2,1,1)
    # print("w.shape", w.shape)
    R = SO3exp(w)
    R_t = torch.linalg.matrix_exp(so3hat(w))
    error = torch.norm(R - R_t)
    if error > 1e-6:
        raise ValueError("error: ", error)
    else:
        print("SO3exp test passed, error: ", error)



    # test SEn3inverse
    A = torch.eye(5).to(device).unsqueeze(0).unsqueeze(0)
    A = A.repeat(10,2,1,1)
    temp = SO3exp(torch.randn(10,2,3))
    # print("temp.shape", temp.shape)
    # print(A[..., :3, :3].shape)
    A[..., :3, :3] = temp
    A[..., :3, 3] = torch.randn(10,2,3)
    A[...,:3,4] = torch.randn(10,2,3)
    A_inv = SEn3inverse(A, numerial_invese = False)
    temp = torch.matmul(A, A_inv)
    error = torch.norm(temp - torch.eye(5).to(device))
    if error > 1e-6:
        raise ValueError("error: ", error)
    else:
        print("SEn3inverse test passed, error: ", error)


    ## test SO3leftJaco
    phi = torch.tensor([[torch.pi/3,0,0],[torch.pi*1.8,0,0],[0,0,0],[0,0,torch.pi/4]]).to(device).unsqueeze(0)
    phi = phi.repeat(2,1,1)
    # print("phi.shape", phi.shape)
    temp1 = SO3leftJaco(phi)
    # print(temp1)
    temp2 = SO3leftJacoInv(phi)
    temp3 = torch.matmul(temp1,temp2)
    error = torch.norm(temp3 - torch.eye(3).to(device).unsqueeze(0).unsqueeze(0).repeat(2,4,1,1))
    if error > 1e-6:
        raise ValueError("error: ", error)
    else:
        print("SO3leftJaco and SO3leftJacoInv test passed, error: ", error)

    ## test SEn3leftJaco
    phi = torch.tensor([[torch.pi/3,0,0,1,0,0],[torch.pi*1.8,0,0,1,0,0],[0,0,0,1,0,0],[0,0,torch.pi/4,1,0,0]]).to(device).unsqueeze(0)
    phi = phi.repeat(2,1,1)
    # temp1 = SEn3leftJaco(phi)
    # print(temp1)

    def test():
        return 1,2,3
    
    a = test()
    print(a)
