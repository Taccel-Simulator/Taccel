from typing import TYPE_CHECKING
import warp as wp
import warp_ipc.utils.matrix as matrix
from warp_ipc.utils.constants import ENV_STATE_INVALID, ENV_STATE_NEWTON_SOLVED
from warp_ipc.utils.env_ops import reduce_env_energy_soft_shell
from warp_ipc.utils.make_pd import *
from warp_ipc.utils.matrix import COOMatrix3x3
from warp_ipc.utils.wp_math import col_stack2, mat_l2_norm_sqr_3x2, sqr
from warp_ipc.utils.wp_types import mat66d, mat96d, vec6d
if TYPE_CHECKING:
    from warp_ipc.sim_model import ASRModel

@wp.kernel
def compute_shell_elastic_energy_kernel(energy_buffer: wp.array(dtype=wp.float64), x: wp.array(dtype=wp.vec3d), tri_elems: wp.array(dtype=wp.vec3i), IB: wp.array(dtype=wp.mat22d), elem_vol: wp.array(dtype=wp.float64), affine_tris_offset: wp.int32, soft_shell_E: wp.array(dtype=wp.float64), soft_shell_nu: wp.array(dtype=wp.float64), scale: wp.float64):
    print('[ERROR] Unexpected Recompilation: compute_shell_elastic_energy_kernel')

def val(x: wp.array, sim: 'ASRModel', scale: float, reduce_each: bool):
    if reduce_each:
        sim.sim_cache.soft_shell_energy.zero_()
    wp.launch(kernel=compute_shell_elastic_energy_kernel, dim=sim.soft_shell_tris_num, inputs=[sim.sim_cache.soft_shell_energy, x, sim.face, sim.soft_shell_IB, sim.soft_shell_elem_vol, sim.affine_tris_num, sim.soft_shell_E, sim.soft_shell_nu, scale])
    if reduce_each:
        energy = wp.zeros(shape=sim.num_envs, dtype=wp.float64, device=sim.device)
        reduce_env_energy_soft_shell(sim.sim_cache.soft_shell_energy, energy, sim)
        return wp.to_torch(energy)
    else:
        return None

@wp.kernel
def compute_shell_elastic_grad_kernel(soft_gradient_x: wp.array(dtype=wp.vec3d), x: wp.array(dtype=wp.vec3d), tri_elems: wp.array(dtype=wp.vec3i), IBs: wp.array(dtype=wp.mat22d), elem_vol: wp.array(dtype=wp.float64), affine_tris_offset: wp.int32, soft_shell_E: wp.array(dtype=wp.float64), soft_shell_nu: wp.array(dtype=wp.float64), scale: wp.float64, affine_verts_num: wp.int32, face2env: wp.array(dtype=wp.int32), env_states: wp.array(dtype=wp.int32)):
    print('[ERROR] Unexpected Recompilation: compute_shell_elastic_grad_kernel')

def grad(x: wp.array, sim: 'ASRModel', scale: float, soft_gradient_x: wp.array):
    wp.launch(kernel=compute_shell_elastic_grad_kernel, dim=sim.soft_shell_tris_num, inputs=[soft_gradient_x, x, sim.face, sim.soft_shell_IB, sim.soft_shell_elem_vol, sim.affine_tris_num, sim.soft_shell_E, sim.soft_shell_nu, scale, sim.affine_verts_num, sim.face2env, sim.env_states], outputs=[])

@wp.kernel
def compute_shell_elastic_hess_kernel(hess_soft_diag: COOMatrix3x3, hess_soft_elastic: COOMatrix3x3, x: wp.array(dtype=wp.vec3d), tri_elems: wp.array(dtype=wp.vec3i), IBs: wp.array(dtype=wp.mat22d), elem_vol: wp.array(dtype=wp.float64), affine_tris_offset: wp.int32, soft_shell_E: wp.array(dtype=wp.float64), soft_shell_nu: wp.array(dtype=wp.float64), scale: wp.float64, affine_verts_num: wp.int32, affine_dofs: wp.int32, face2env: wp.array(dtype=wp.int32), env_states: wp.array(dtype=wp.int32)):
    print('[ERROR] Unexpected Recompilation: compute_shell_elastic_hess_kernel')

def hess(x: wp.array, sim: 'ASRModel', scale: float):
    wp.launch(kernel=compute_shell_elastic_hess_kernel, dim=sim.soft_shell_tris_num, inputs=[sim.hess_soft_diag, sim.hess_soft_shell_elastic, x, sim.face, sim.soft_shell_IB, sim.soft_shell_elem_vol, sim.affine_tris_num, sim.soft_shell_E, sim.soft_shell_nu, scale, sim.affine_verts_num, sim.n_affine_dofs, sim.face2env, sim.env_states], outputs=[])