import torch

def slerp(q0, q1, t):
    """
    Perform Spherical Linear intERPolation (SLERP) between two quaternions.
    
    Args:
        q0: Tensor of shape (4,) representing the starting quaternion.
        q1: Tensor of shape (4,) representing the ending quaternion.
        t: Float in [0, 1] or a tensor of shape (N,) representing the interpolation parameter.
    
    Returns:
        Interpolated quaternion of shape (4,) or (N, 4).
    """
    q0 = q0 / q0.norm()  # Ensure q0 is a unit quaternion
    q1 = q1 / q1.norm()  # Ensure q1 is a unit quaternion

    dot_product = torch.dot(q0, q1)
    
    # If the dot product is negative, invert one quaternion to take the shorter path
    if dot_product < 0.0:
        q1 = -q1
        dot_product = -dot_product

    dot_product = torch.clamp(dot_product, -1.0, 1.0)  # Clamp for numerical stability
    theta_0 = torch.acos(dot_product)  # Angle between q0 and q1
    sin_theta_0 = torch.sin(theta_0)

    if sin_theta_0 < 1e-6:
        # If the angle is small, use linear interpolation to avoid division by zero
        return (1.0 - t) * q0 + t * q1

    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)

    s0 = torch.sin((1.0 - t) * theta_0) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0

    return s0 * q0 + s1 * q1


def batch_slerp(q0_batch, q1_batch, t_batch):
    """
    Perform SLERP for a batch of quaternions.
    
    Args:
        q0_batch: Tensor of shape (N, 4) representing the starting quaternions.
        q1_batch: Tensor of shape (N, 4) representing the ending quaternions.
        t_batch: Tensor of shape (N,) representing the interpolation parameters.
    
    Returns:
        Interpolated quaternions of shape (N, 4).
    """
    q0_batch = q0_batch / q0_batch.norm(dim=1, keepdim=True)
    q1_batch = q1_batch / q1_batch.norm(dim=1, keepdim=True)

    dot_product = (q0_batch * q1_batch).sum(dim=1)

    mask = dot_product < 0.0
    q1_batch[mask] = -q1_batch[mask]
    dot_product[mask] = -dot_product[mask]

    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    theta_0 = torch.acos(dot_product)
    sin_theta_0 = torch.sin(theta_0)

    mask = sin_theta_0 < 1e-6
    t_mask = t_batch.unsqueeze(1).expand_as(q0_batch)

    s0 = torch.sin((1.0 - t_mask) * theta_0.unsqueeze(1)) / sin_theta_0.unsqueeze(1)
    s1 = torch.sin(t_mask * theta_0.unsqueeze(1)) / sin_theta_0.unsqueeze(1)

    s0[mask] = 1.0 - t_mask[mask]
    s1[mask] = t_mask[mask]

    return s0 * q0_batch + s1 * q1_batch

def quaternion_apply_vec(q: torch.Tensor, vec: torch.Tensor):
    """
    Apply a unit quaternion to a vector.
    q: Tensor of shape (4,) or (N, 4) representing the quaternion(s). Order: w, x, y, z
    vec: Tensor of shape (3,) or (N, 3) representing the vector(s).
    """
    def quat_multiply(q1, q2):
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack([w, x, y, z], dim=-1)
    
    if q.dim() == 1:  # Single quaternion
        q = q.unsqueeze(0)
    if vec.dim() == 1:  # Single vector
        vec = vec.unsqueeze(0)
    
    q_norm = torch.norm(q, dim=-1, keepdim=True)
    q_normalized = q / q_norm
    
    q_inv = torch.cat([q_normalized[..., 0:1], -q_normalized[..., 1:]], dim=-1)
    q_vec = torch.cat([torch.zeros_like(vec[..., :1]), vec], dim=-1)
    
    q_rotated = quat_multiply(quat_multiply(q_normalized, q_vec), q_inv)

    return q_rotated[..., 1:] if q_rotated.shape[0] > 1 or q_rotated.dim() > 2   else q_rotated[0, 1:]


if __name__ == "__main__":
    # Define two unit quaternions
    
    q0 = torch.tensor([0.70711, 0.18898, 0.37796, 0.56695])  # Example quaternion 
    q1 = torch.tensor([0.96593, 0.20752,  0.13834, 0.069172])  # Example quaternion

    # Define interpolation parameter
    t = torch.tensor(0.75)  # Interpolates halfway between q0 and q1

    # Perform SLERP
    q_interp = slerp(q0, q1, t)
    print("q_interp.norm():", q_interp.norm())
    print("Interpolated Quaternion:", q_interp)

    # Example batch of quaternions
    q0_batch = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.70711, 0.18898, 0.37796, 0.56695]])
    q1_batch = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.96593, 0.20752,  0.13834, 0.069172]])
    t_batch = torch.tensor([0.5, 0.75]) 

    # Perform batch SLERP
    q_interp_batch = batch_slerp(q0_batch, q1_batch, t_batch)

    print("Batch Interpolated Quaternions:", q_interp_batch)
    t_batch = torch.tensor([0.75])
    q_interp_batch = batch_slerp(q0_batch, q1_batch, t_batch)
    print("Batch Interpolated Quaternions:", q_interp_batch)


    # Apply quaternion to vector
    q1 = torch.tensor([0.70711, 0.18898, 0.37796, 0.56695])  # Example quaternion
    vec = torch.tensor([1.0, 1.0, 1.0])  # Example vector
    vec_rotated = quaternion_apply_vec(q1, vec)
    print("Rotated Vector:", vec_rotated)

    # Apply batch quaternion to a batch of vectors
    q_batch = torch.tensor([[0.70711, 0.18898, 0.37796, 0.56695], [0.70711, 0.18898, 0.37796, 0.56695]])
    vec_batch = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    vec_rotated_batch = quaternion_apply_vec(q_batch, vec_batch)
    print("Rotated Batch Vectors:", vec_rotated_batch)

    q_batch = torch.tensor([[0.70711, 0.18898, 0.37796, 0.56695], [0.70711, 0.18898, 0.37796, 0.56695]]).to('cuda')
    q_batch.unsqueeze_(0)
    vec_batch = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]).to('cuda')
    vec_batch.unsqueeze_(0)
    vec_rotated_batch = quaternion_apply_vec(q_batch, vec_batch)
    print("Rotated Batch Vectors:", vec_rotated_batch)

    print("vec_rotated_batch.device:", vec_rotated_batch.device)

    



