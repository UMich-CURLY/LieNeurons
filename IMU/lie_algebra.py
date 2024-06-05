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

def SO3log(R: torch.Tensor, eps = 1e-6)->torch.Tensor:
    """R shape (..., 3, 3)"""
    trace = torch.einsum('...ii->...', R)
    trace = torch.clamp(trace, -1, 3)
    theta = torch.acos((trace - 1.0) / 2.0)
    small_theta_mask = theta < eps
    Identity = torch.eye(3).expand(R.shape[:-2]+(3,3)).to(R.device)

    w_so3 = torch.zeros_like(R)
    w_so3[small_theta_mask] = R[small_theta_mask] - Identity[small_theta_mask]
    w_so3[~small_theta_mask] = (0.5 * theta[~small_theta_mask] / torch.sin(theta[~small_theta_mask])).unsqueeze(-1).unsqueeze(-1) * (R[~small_theta_mask] - R[~small_theta_mask].transpose(-1,-2))
    w = so3vee(w_so3)
    return w
    
def outer_product(v1: torch.Tensor, v2: torch.Tensor)->torch.Tensor:
    """v1, v2 shape (..., 3)"""
    return torch.einsum('...i,...j->...ij', v1, v2)

def so3hat(w: torch.Tensor)->torch.Tensor:
    """w shape (..., 3)"""
    return torch.stack([torch.zeros_like(w[..., 0]), -w[..., 2], w[..., 1],
                        w[..., 2], torch.zeros_like(w[..., 0]), -w[..., 0],
                        -w[..., 1], w[..., 0], torch.zeros_like(w[..., 0])], dim=-1).reshape(w.shape[:-1]+(3,3))
    
def so3vee(W: torch.Tensor)->torch.Tensor:
    """W shape (..., 3, 3)"""
    return torch.stack([W[..., 2, 1], W[..., 0, 2], W[..., 1, 0]], dim=-1)

def sen3hat(xi: torch.Tensor)->torch.Tensor:
    """xi shape (..., 3*n), n = 1,2,3..."""
    dim_mat = round(xi.shape[-1] // 3) +2 
    output = torch.zeros(xi.shape[:-1]+(dim_mat,dim_mat)).to(xi.device)
    output[..., :3, :3] = so3hat(xi[..., :3])
    output[..., :3, 3:] = xi[...,3:].reshape(*xi.shape[:-1], -1,3).transpose(-1,-2)
    return output

def sen3vee(X: torch.Tensor)->torch.Tensor:
    """X shape (..., 3+n, 3+n), n = 1,2,3..."""
    reshaped_tensor = X[...,:3,3:].transpose(-1,-2).reshape(*X.shape[:-2], -1)
    return torch.cat((so3vee(X[..., :3, :3]), reshaped_tensor), dim = -1)

def SEn3exp(xi: torch.Tensor, eps = 1e-6)->torch.Tensor:
    """xi shape (..., 3*n), n = 1,2,3..."""
    phi = xi[..., :3]
    R = SO3exp(phi)
    dim_mat = round(xi.shape[-1]//3)+2
    output = torch.eye(dim_mat).expand(xi.shape[:-1]+(dim_mat,dim_mat)).to(xi.device)
    output[..., :3, :3] = R
    Jl =SO3leftJaco(phi)
    temp_rest = Jl @ xi[..., 3:].reshape(*xi.shape[:-1], -1,3).transpose(-1,-2)
    output[..., :3, 3:] = temp_rest
    return output

def SEn3log(X: torch.Tensor, eps = 1e-6)->torch.Tensor:
    """X shape (..., 3+n, 3+n), n = 1,2,3..."""
    phi = SO3log(X[..., :3, :3])
    xi = torch.zeros(X.shape[:-2]+((X.shape[-1]-2)*3,)).to(X.device)
    xi[..., :3] = phi
    temp_rest = SO3leftJacoInv(phi) @ X[..., :3, 3:]
    xi[..., 3:] = temp_rest.transpose(-1,-2).reshape(*temp_rest.shape[:-2], -1)
    return xi

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
    lage_theta_mask = theta[...,0] > 1.8 * torch.pi 
    remaider = theta[lage_theta_mask] % (2 * torch.pi)
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
        temp_rest = - torch.matmul(X_inv[..., :3, :3], X[..., :3, 3:])
        X_inv[..., :3, 3:] = temp_rest
    return X_inv


def SEn3leftJaco(xi: torch.Tensor, eps = 1e-6)->torch.Tensor:
    """left jacobian of SEn(3), phi shape (..., 3*n), n = 1,2,3..., Order: xi_R, xi_v, xi_p ..."""
    """                         phi should be (m1,m2,3*n) or (m1,3*n)"""
    phi = xi[..., :3]

    theta = torch.norm(phi, dim = -1, keepdim = True)
    mask_small_theta = theta[..., 0] < eps
    Identity = torch.eye(xi.shape[-1]).expand(phi.shape[:-1]+(xi.shape[-1],xi.shape[-1])).to(phi.device)

    Jaco = torch.zeros_like(Identity)
    Jaco[mask_small_theta] = Identity[mask_small_theta] + 0.5 * ad_sen3(xi[mask_small_theta])
    Jaco_left_SO3 = SO3leftJaco(phi[~mask_small_theta])
    temp = Jaco[~mask_small_theta]
    for i in range(round(xi.shape[-1] // 3)):
        temp[:, 3*i:3*(i+1), 3*i:3*(i+1)] = Jaco_left_SO3
    temp[:, 3:, :3] = Ql_forSE3Jaco(xi[~mask_small_theta])
    Jaco[~mask_small_theta] = temp
    return Jaco


def SEn3leftJaco_inv(xi: torch.Tensor, eps = 1e-6)->torch.Tensor:
    phi = xi[..., :3]

    theta = torch.norm(phi, dim = -1, keepdim = True)
    mask_small_theta = theta[..., 0] < eps
    Identity = torch.eye(xi.shape[-1]).expand(phi.shape[:-1]+(xi.shape[-1],xi.shape[-1])).to(phi.device)

    Jaco = torch.zeros_like(Identity)
    Jaco[mask_small_theta] = Identity[mask_small_theta] - 0.5 * ad_sen3(xi[mask_small_theta])
    Jaco_left_SO3_inv = SO3leftJacoInv(phi[~mask_small_theta])
    temp = Jaco[~mask_small_theta]
    Ql = Ql_forSE3Jaco(xi[~mask_small_theta])
    temp[:, :3, :3] = Jaco_left_SO3_inv
    for i in range(1, round(xi.shape[-1] // 3)):
        temp[:, 3*i:3*(i+1), 3*i:3*(i+1)] = Jaco_left_SO3_inv
        temp[:, 3*i:3*(i+1), :3] = - Jaco_left_SO3_inv @ Ql[:, 3*(i-1):3*i, :3] @ Jaco_left_SO3_inv
    Jaco[~mask_small_theta] = temp
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
    m2 = ((theta - s_theta) / theta3).unsqueeze(-1)
    m3 = ((theta2 + 2 * c_theta - 2) / (2 * theta4)).unsqueeze(-1)
    m4 = ((2*theta - 3*s_theta + theta * c_theta) / (2 * theta5)).unsqueeze(-1)

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
    return adm


if __name__ == '__main__':
    print("Test lie_algebra.py")
    print("---------------------------------")
    torch.set_default_dtype(torch.float64)

    ## test SO3exp
    device = 'cuda'
    w = torch.tensor([[1.6469, 3.7091, 1.3493],[1e-7,0,0],[0,torch.pi/3,0],[0,0,torch.pi/4]]).to(device).unsqueeze(0)
    w = w.repeat(2,1,1)
    # print("w.shape", w.shape)
    R = SO3exp(w)
    R_t = torch.linalg.matrix_exp(so3hat(w))
    error = torch.norm(R - R_t)
    if error > 1e-6:
        raise ValueError("error: ", error)
    else:
        print("SO3exp test passed, error: ", error)

    ## test SO3log
    w_log = SO3log(R)
    R2 = SO3exp(w_log)
    error = torch.norm(R - R2)
    # w_norm = torch.norm(w, dim = -1, keepdim = True)
    # w_unit = w / w_norm
    # w_clampwith2pi = w_unit * (w_norm % (2. *torch.pi))
    # print("w_clampwith2pi \n", w_clampwith2pi)
    # error = torch.norm(w_unit * (w_norm % (2. *torch.pi)) - w_log)
    # for i in range(2):
    #     for j in range(4):
    #         print("error at ", i, j, ":", torch.norm(w[i,j,...] - w_log[i,j,...]), "w norm: ", torch.norm(w[i,j,...]))
    if error > 1e-6:
        raise ValueError("error: ", error)
    else:
        print("SO3log test passed, error: ", error)



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
    phi = torch.tensor([[torch.pi/3,0,0],[torch.pi*1.8,0,0],[1e-4,0,0],[0,0,0.]]).to(device).unsqueeze(0)
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

    # phi = torch.tensor([torch.pi * 1.8,0,0])
    # temp1 = SO3leftJaco(phi)
    # print("temp1.shape", temp1.shape)
    # print("temp1 \n", temp1)
    # import math
    # Jaco_true = torch.eye(3).to(phi.device)
    # ad_mult = torch.eye(3).to(phi.device)
    # for i in range(1,20):
    #     ad_mult = ad_mult @ so3hat(phi)
    #     Jaco_true = Jaco_true + ad_mult / math.factorial(i+1)
    # print("Jaco_true \n", Jaco_true)

    ## test SEn3leftJaco
    import math
    xi = torch.tensor([[torch.pi/3,0,0,1,0,0],[torch.pi,0,0,1,0,0],[1e-4,0,0,1,0,0],[0,0,torch.pi/4,1,0,0]]).to(device).unsqueeze(0)
    xi = xi.repeat(2,1,1)
    xi = torch.randn(2,4,9).to(device)
    # print("phi.shape", xi.shape)
    Jaco_SE3_true = torch.eye(xi.shape[-1]).expand(xi.shape[:-1]+(xi.shape[-1],xi.shape[-1])).to(device)
    ad_mult = Jaco_SE3_true.clone()
    for i in range(1,20):
        ad_mult = ad_mult @ ad_sen3(xi)
        Jaco_SE3_true += ad_mult / math.factorial(i+1)
    temp1 = SEn3leftJaco(xi)
    error = torch.norm(temp1 - Jaco_SE3_true)
    if error > 1e-6:
        raise ValueError("error: ", error)
    else:
        print("SEn3leftJaco test passed, error: ", error)

    ## test SEn3leftJaco_inv
    xi = torch.randn(2,4,9).to(device)
    temp1 = SEn3leftJaco_inv(xi)
    temp2 = torch.matmul(temp1, SEn3leftJaco(xi))
    error = torch.norm(temp2 - torch.eye(xi.shape[-1]).to(xi.device))
    if error > 1e-6:
        raise ValueError("error: ", error)
    else:
        print("SEn3leftJaco_inv test passed, error: ", error)

    # for i in range(2):
    #     for j in range(4):
    #         print("error at ", i, j, ":", torch.norm(temp1[i,j,...] - Jaco_SE3_true[i,j,...]))
    
    # print("temp1 \n", temp1[0,1,...])
    # print("Jaco_SE3_true \n", Jaco_SE3_true[0,1,...])

    # J_SO3 = SO3leftJaco(xi[..., :3])
    # print("J_SO3 \n", J_SO3[0,0,...])

    # xi = torch.arange(1,10).to(device)
    # temp = sen3hat(xi)
    # print("temp \n", temp)
    # temp = sen3vee(temp)
    # print("temp \n", temp)

    # test SEn3exp
    xi = torch.randn(2,4,6).to(device)
    temp1 = SEn3exp(xi)
    temp2 = torch.linalg.matrix_exp(sen3hat(xi))

    error = torch.norm(temp1 - temp2)
    if error > 1e-6:
        raise ValueError("error: ", error)
    else:
        print("SEn3exp test passed, error: ", error)

    # test SEn3log
    xi_log = SEn3log(temp1)
    temp3 = SEn3exp(xi_log)
    error = torch.norm(temp1 - temp3)
    if error > 1e-6:
        raise ValueError("error: ", error)
    else:
        print("SEn3log test passed, error: ", error)



    print("---------------------------------")
    


    

