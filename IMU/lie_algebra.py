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
    # outer product is used here, not conventional Rodrigues formula, copy from M. Brossard
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
    
    
def SEn3inverse(X: torch.Tensor, numerial_invese = False)->torch.Tensor:
    """inverse of SEn(3) matrix, X, shape (..., 4, 4)"""
    if numerial_invese:
        X_inv = torch.linalg.inv(X)
    else:
        X_inv = X.clone()
        X_inv[..., :3, :3] = X[..., :3, :3].transpose(-2, -1)
        for i in range(3,X.shape[-1]):
            X_inv[..., :3, i] = - torch.matmul(X_inv[..., :3, :3], X[..., :3, i].unsqueeze(-1)).squeeze(-1)
    return X_inv




if __name__ == '__main__':
    ## test SO3exp
    device = 'cuda'
    w = torch.tensor([[torch.pi/3,0,0],[torch.pi/3,0,0],[0,torch.pi/3,0],[0,0,torch.pi/4]]).to(device).unsqueeze(0)
    # w = torch.tensor([[torch.pi/2,0,0]])
    print("w.shape", w.shape)
    R = SO3exp(w)
    print(R)
    R_t = torch.linalg.matrix_exp(so3hat(w))
    print(R_t)
    error = torch.norm(R - R_t)
    print("error: ", error)


    # test SEn3inverse
    A = torch.eye(5).to(device).unsqueeze(0).unsqueeze(0)
    A = A.repeat(10,2,1,1)
    temp = SO3exp(torch.randn(10,2,3))
    print("temp.shape", temp.shape)
    print(A[..., :3, :3].shape)
    A[..., :3, :3] = temp
    A[..., :3, 3] = torch.randn(10,2,3)
    A[...,:3,4] = torch.randn(10,2,3)
    A_inv = SEn3inverse(A, numerial_invese = False)


    test0 = A[1,1,:,:]
    test1 = A_inv[1,1,:,:]
    print("test0\n", test0)
    print("test1\n", test1)
    temp = test0 @ test1
    print("temp\n", temp)
    error = torch.norm(temp - torch.eye(5).to(device))
    print("error: ", error)