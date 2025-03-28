from typing import TYPE_CHECKING
import torch
import warp as wp
import warp.sparse as wps
import warp_ipc.utils.matrix as matrix
from warp_ipc.utils.constants import ENV_STATE_INVALID, ENV_STATE_VALID
from warp_ipc.utils.env_ops import reduce_env_energy_affine_body, reduce_env_energy_soft_vert
from warp_ipc.utils.wp_math import sqr
from warp_ipc.utils.wp_types import vec12d
if TYPE_CHECKING:
    from warp_ipc.sim_model import ASRModel

def val(sim: 'ASRModel', x, y, sm_damping: wps.BsrMatrix, reduce_each: bool, dt: float):
    coef = sim.k_elasticity_damping * (dt * sim.dx_div_dv_scale)
    v_torch = wp.to_torch(sim.sim_cache.damping_sys_v)
    if sim.soft_verts_num > 0:
        x_torch = wp.to_torch(x)
        hat_x_torch = wp.to_torch(sim.hat_x)
        v_torch[sim.n_affine_dofs // 3:, :] = x_torch[sim.affine_verts_num:] - hat_x_torch[sim.affine_verts_num:]
    if sim.affine_body_num > 0:
        y_torch = wp.to_torch(y)
        hat_y_torch = wp.to_torch(sim.hat_y)
        v_torch[:sim.n_affine_dofs // 3, :] = (y_torch - hat_y_torch).view(-1, 3)
    wps.bsr_mv(sm_damping, sim.sim_cache.damping_sys_v, sim.sim_cache.damping_sys_Hv)
    Hv_torch = wp.to_torch(sim.sim_cache.damping_sys_Hv)
    energy = wp.zeros(shape=sim.num_envs, dtype=wp.float64, device=sim.device)
    E_torch = wp.to_torch(sim.sim_cache.damping_sys_energy)
    E_torch[:] = 0.5 * (v_torch * Hv_torch).sum(dim=1) * coef
    if reduce_each:
        reduce_env_energy_affine_body(wp.from_torch(E_torch[:sim.n_affine_dofs // 3].view(-1, 4).sum(axis=1)), energy, sim)
        reduce_env_energy_soft_vert(wp.from_torch(E_torch[sim.n_affine_dofs // 3:]), energy, sim)
        return wp.to_torch(energy)
    else:
        wp.to_torch(sim.sim_cache.affine_body_energy)[:] += E_torch[:sim.n_affine_dofs // 3].view(-1, 4).sum(axis=1)
        wp.to_torch(sim.sim_cache.soft_vert_energy)[:] += E_torch[sim.n_affine_dofs // 3:]
        return None

def grad(sim: 'ASRModel', x, y, sm_damping: wps.BsrMatrix, soft_gradient_x: wp.array, affine_gradient_y: wp.array, dt: float):
    coef = sim.k_elasticity_damping * (dt * sim.dx_div_dv_scale)
    v_torch = wp.to_torch(sim.sim_cache.damping_sys_v)
    if sim.soft_verts_num > 0:
        x_torch = wp.to_torch(x)
        hat_x_torch = wp.to_torch(sim.hat_x)
        v_torch[sim.n_affine_dofs // 3:, :] = x_torch[sim.affine_verts_num:] - hat_x_torch[sim.affine_verts_num:]
    if sim.affine_body_num > 0:
        y_torch = wp.to_torch(y)
        hat_y_torch = wp.to_torch(sim.hat_y)
        v_torch[:sim.n_affine_dofs // 3, :] = (y_torch - hat_y_torch).view(-1, 3)
    wps.bsr_mv(sm_damping, sim.sim_cache.damping_sys_v, sim.sim_cache.damping_sys_Hv)
    grad_torch = wp.to_torch(sim.sim_cache.damping_sys_Hv)
    grad_torch *= coef
    if sim.soft_verts_num > 0:
        wp.to_torch(soft_gradient_x)[:] += grad_torch[sim.n_affine_dofs // 3:, :]
    if sim.affine_body_num > 0:
        wp.to_torch(affine_gradient_y)[:] += grad_torch[:sim.n_affine_dofs // 3, :].view(-1, 12)