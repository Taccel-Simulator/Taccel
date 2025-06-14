from typing import TYPE_CHECKING
import warp as wp
import warp_ipc.utils.matrix as matrix
from warp_ipc.utils.constants import ENV_STATE_INVALID, ENV_STATE_NEWTON_SOLVED, _0, _1, _2, id_3
from warp_ipc.utils.env_ops import reduce_env_energy_soft_tet
from warp_ipc.utils.make_pd import *
from warp_ipc.utils.make_pd import make_pd_9x9
from warp_ipc.utils.matrix import COOMatrix3x3
from warp_ipc.utils.wp_math import col_stack3, column_flatten_3x3, kronecker_3x3_3x3, mat_l2_norm_sqr, sqr
from warp_ipc.utils.wp_types import mat12x9d, mat99d
if TYPE_CHECKING:
    from warp_ipc.sim_model import ASRModel

@wp.kernel
def compute_stvk_elastic_energy_kernel(energy_buffer: wp.array(dtype=wp.float64), x: wp.array(dtype=wp.vec3d), tet_elems: wp.array(dtype=wp.vec4i), IB: wp.array(dtype=wp.mat33d), elem_vol: wp.array(dtype=wp.float64), soft_tet_E: wp.array(dtype=wp.float64), soft_tet_nu: wp.array(dtype=wp.float64), scale: wp.float64):
    print('[ERROR] Unexpected Recompilation: compute_stvk_elastic_energy_kernel')

def val(x: wp.array, sim: 'ASRModel', scale: float, reduce_each: bool):
    if reduce_each:
        sim.sim_cache.soft_tet_energy.zero_()
    wp.launch(kernel=compute_stvk_elastic_energy_kernel, dim=sim.soft_vol_tets_num, inputs=[sim.sim_cache.soft_tet_energy, x, sim.tets, sim.soft_tet_IB, sim.soft_tet_elem_vol, sim.soft_tet_E, sim.soft_tet_nu, scale])
    if reduce_each:
        energy = wp.zeros(shape=sim.num_envs, dtype=wp.float64, device=sim.device)
        reduce_env_energy_soft_tet(sim.sim_cache.soft_tet_energy, energy, sim)
        return wp.to_torch(energy)
    else:
        return None

@wp.kernel
def compute_stvk_elastic_grad_kernel(soft_gradient_x: wp.array(dtype=wp.vec3d), x: wp.array(dtype=wp.vec3d), tet_elems: wp.array(dtype=wp.vec4i), IBs: wp.array(dtype=wp.mat33d), elem_vol: wp.array(dtype=wp.float64), soft_tet_E: wp.array(dtype=wp.float64), soft_tet_nu: wp.array(dtype=wp.float64), scale: wp.float64, affine_verts_num: wp.int32, tet_envs: wp.array(dtype=wp.int32), env_states: wp.array(dtype=wp.int32)):
    print('[ERROR] Unexpected Recompilation: compute_stvk_elastic_grad_kernel')

def grad(x: wp.array, sim: 'ASRModel', scale: float, soft_gradient_x: wp.array):
    wp.launch(kernel=compute_stvk_elastic_grad_kernel, dim=sim.soft_vol_tets_num, inputs=[soft_gradient_x, x, sim.tets, sim.soft_tet_IB, sim.soft_tet_elem_vol, sim.soft_tet_E, sim.soft_tet_nu, scale, sim.affine_verts_num, sim.tet_envs, sim.env_states])

@wp.kernel
def compute_stvk_elastic_hess_kernel(hess_soft_diag: COOMatrix3x3, hess_soft_tet_elastic: COOMatrix3x3, x: wp.array(dtype=wp.vec3d), tet_elems: wp.array(dtype=wp.vec4i), IBs: wp.array(dtype=wp.mat33d), elem_vol: wp.array(dtype=wp.float64), soft_tet_E: wp.array(dtype=wp.float64), soft_tet_nu: wp.array(dtype=wp.float64), scale: wp.float64, affine_verts_num: wp.int32, affine_dofs: wp.int32, tet_envs: wp.array(dtype=wp.int32), env_states: wp.array(dtype=wp.int32)):
    print('[ERROR] Unexpected Recompilation: compute_stvk_elastic_hess_kernel')

def hess(x: wp.array, sim: 'ASRModel', scale: float):
    wp.launch(kernel=compute_stvk_elastic_hess_kernel, dim=sim.soft_vol_tets_num, inputs=[sim.hess_soft_diag, sim.hess_soft_vol_elastic, x, sim.tets, sim.soft_tet_IB, sim.soft_tet_elem_vol, sim.soft_tet_E, sim.soft_tet_nu, scale, sim.affine_verts_num, sim.n_affine_dofs, sim.tet_envs, sim.env_states])