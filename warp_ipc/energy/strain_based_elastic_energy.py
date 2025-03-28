from typing import TYPE_CHECKING
import warp as wp
from warp import sparse as wps
import warp_ipc.utils.matrix as matrix
from warp_ipc.utils.constants import ENV_STATE_INVALID, ENV_STATE_NEWTON_SOLVED, ENV_STATE_VALID, _0, _1
from warp_ipc.utils.env_ops import reduce_env_energy_soft_tet
from warp_ipc.utils.make_pd import *
from warp_ipc.utils.make_pd import make_pd_9x9, make_pd_12x12
from warp_ipc.utils.matrix import COOMatrix3x3
from warp_ipc.utils.wp_debug import print_mat_val
from warp_ipc.utils.wp_math import cat_3_vec3d, cat_4_vec3d, col_stack3, column_flatten_3x3, det_3x3, det_derivative_3x3, extract_column_mat33d, is_nan, mat_l2_norm_sqr_3x2, sqr
from warp_ipc.utils.wp_types import mat12x9d, mat66d, mat96d, mat99d, vec6d, vec9d
if TYPE_CHECKING:
    from warp_ipc.sim_model import ASRModel

@wp.kernel
def compute_strain_based_elastic_energy_kernel(energy: wp.array(dtype=wp.float64), x: wp.array(dtype=wp.vec3d), tet_elems: wp.array(dtype=wp.vec4i), IB: wp.array(dtype=wp.mat33d), elem_vol: wp.array(dtype=wp.float64), tet_elems_offset: wp.int32, E: wp.float64, nu: wp.float64, scale: wp.float64):
    pass

@wp.kernel
def compute_strain_based_elastic_grad_kernel(soft_gradient_x: wp.array(dtype=wp.vec3d), x: wp.array(dtype=wp.vec3d), tet_elems: wp.array(dtype=wp.vec4i), IBs: wp.array(dtype=wp.mat33d), elem_vol: wp.array(dtype=wp.float64), tet_elems_offset: wp.int32, E: wp.float64, nu: wp.float64, scale: wp.float64, affine_verts_num: wp.int32, tet_envs: wp.array(dtype=wp.int32), env_states: wp.array(dtype=wp.int32)):
    pass

@wp.kernel
def compute_strain_based_elastic_hess_kernel(hess_soft_diag: COOMatrix3x3, hess_soft_tet_elastic: COOMatrix3x3, x: wp.array(dtype=wp.vec3d), tet_elems: wp.array(dtype=wp.vec4i), IBs: wp.array(dtype=wp.mat33d), elem_vol: wp.array(dtype=wp.float64), tet_elems_offset: wp.int32, E: wp.float64, nu: wp.float64, scale: wp.float64, affine_verts_num: wp.int32, affine_dofs: wp.int32, apply_make_pd: wp.bool, tet_envs: wp.array(dtype=wp.int32), env_states: wp.array(dtype=wp.int32)):
    pass

def val(x: wp.array, sim: 'ASRModel', scale: float, reduce_each: bool):
    if reduce_each:
        sim.sim_cache.soft_tet_energy.zero_()
    for (soft_elem_start, soft_elem_end, E, nu, xi) in zip(sim.soft_tet_elem_offsets[:-1], sim.soft_tet_elem_offsets[1:], sim.E_body[sim.soft_vol_body_offset:], sim.nu_body[sim.soft_vol_body_offset:], sim.xi_body[sim.soft_vol_body_offset:]):
        soft_elem_num = soft_elem_end - soft_elem_start
        wp.launch(compute_strain_based_elastic_energy_kernel, dim=soft_elem_num, inputs=[sim.sim_cache.soft_tet_energy, x, sim.tets, sim.soft_tet_IB, sim.soft_tet_elem_vol, soft_elem_start, E, nu, scale])
    if reduce_each:
        energy = wp.zeros(shape=sim.num_envs, dtype=wp.float64, device=sim.device)
        reduce_env_energy_soft_tet(sim.sim_cache.soft_tet_energy, energy, sim)
        return wp.to_torch(energy)
    else:
        return None

def grad(x: wp.array, sim: 'ASRModel', scale: float, soft_gradient_x: wp.array):
    for (soft_elem_start, soft_elem_end, E, nu, xi) in zip(sim.soft_tet_elem_offsets[:-1], sim.soft_tet_elem_offsets[1:], sim.E_body[sim.soft_vol_body_offset:], sim.nu_body[sim.soft_vol_body_offset:], sim.xi_body[sim.soft_vol_body_offset:]):
        soft_elem_num = soft_elem_end - soft_elem_start
        wp.launch(compute_strain_based_elastic_grad_kernel, dim=soft_elem_num, inputs=[soft_gradient_x, x, sim.tets, sim.soft_tet_IB, sim.soft_tet_elem_vol, soft_elem_start, E, nu, scale, sim.affine_verts_num, sim.tet_envs, sim.env_states])

def hess(x: wp.array, sim: 'ASRModel', scale: float):
    for (soft_elem_start, soft_elem_end, E, nu, xi) in zip(sim.soft_tet_elem_offsets[:-1], sim.soft_tet_elem_offsets[1:], sim.E_body[sim.soft_vol_body_offset:], sim.nu_body[sim.soft_vol_body_offset:], sim.xi_body[sim.soft_vol_body_offset:]):
        soft_elem_num = soft_elem_end - soft_elem_start
        wp.launch(compute_strain_based_elastic_hess_kernel, dim=soft_elem_num, inputs=[sim.hess_soft_diag, sim.hess_soft_vol_elastic, x, sim.tets, sim.soft_tet_IB, sim.soft_tet_elem_vol, soft_elem_start, E, nu, scale, sim.affine_verts_num, sim.n_affine_dofs, True, sim.tet_envs, sim.env_states])