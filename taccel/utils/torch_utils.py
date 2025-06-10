import torch
from pytorch3d import transforms


@torch.jit.script
def scale(x, lower, upper):
    """Scale the input tensor x from the range [-1, 1] to [lower, upper]."""
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    """Unscale the input tensor x from the range [lower, upper] to [-1, 1]."""
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def tensor_clamp(x, lower, upper):
    return torch.max(torch.min(x, upper), lower)


# @torch.jit.script
def xyzrpy_to_tf(xyzrpy):
    """Convert a tensor of shape (N, 6) to a tensor of shape (N, 4, 4) representing the transformation matrix."""
    N = xyzrpy.shape[0]
    xyz, rpy = xyzrpy[:, :3], xyzrpy[:, 3:]
    rotmat = transforms.euler_angles_to_matrix(rpy, convention="XYZ")
    tf = torch.zeros((N, 4, 4), device=xyzrpy.device, dtype=xyzrpy.dtype)
    tf[:, :3, :3] = rotmat
    tf[:, :3, 3] = xyz
    tf[:, 3, 3] = 1.0
    return tf


@torch.jit.script
def tf_to_xyzrpy(tf):
    """Convert a tensor of shape (N, 4, 4) to a tensor of shape (N, 6) representing the transformation matrix."""
    N = tf.shape[0]
    xyz = tf[:, :3, 3]
    rotmat = tf[:, :3, :3]
    rpy = transforms.matrix_to_euler_angles(rotmat, convention="XYZ")
    return torch.cat((xyz, rpy), dim=1)


@torch.jit.script
def interp_q(q_init: torch.Tensor, q_end: torch.Tensor, n_steps: int):
    if n_steps == 1:
        return q_end.unsqueeze(1)

    ts = torch.linspace(0, 1, n_steps, device=q_end.device).unsqueeze(0)
    q_interp = q_end.unsqueeze(1) * ts.unsqueeze(-1) + q_init.unsqueeze(1) * (1 - ts).unsqueeze(-1)

    return q_interp


@torch.jit.script
def interp_local_se3(pose: torch.Tensor, n_steps: int):
    if n_steps == 1:
        return pose.unsqueeze(1)

    ts = torch.linspace(0, 1, n_steps, device=pose.device).unsqueeze(0)

    pos, rotmat = pose[:, :3, 3], pose[:, :3, :3]

    pos_interp = pos.unsqueeze(1) * ts.unsqueeze(-1)

    rotvec = transforms.matrix_to_axis_angle(rotmat)
    rotvec_rad = torch.norm(rotvec, dim=-1)
    rotvec_axis = rotvec / (rotvec_rad.unsqueeze(-1) + 1e-6)
    rotvec_rad_interp = rotvec_rad.unsqueeze(1) * ts
    rotvec_interp = rotvec_axis.unsqueeze(1) * rotvec_rad_interp.unsqueeze(-1)
    rotmat_interp = transforms.axis_angle_to_matrix(rotvec_interp)

    interp_pose = pose.unsqueeze(1).tile(1, n_steps, 1, 1)
    interp_pose[:, :, :3, 3] = pos_interp
    interp_pose[:, :, :3, :3] = rotmat_interp

    return interp_pose
