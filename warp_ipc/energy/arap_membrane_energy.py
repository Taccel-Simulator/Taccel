from typing import TYPE_CHECKING
import warp as wp
import warp_ipc.utils.matrix as matrix
from warp_ipc.utils.constants import ENV_STATE_INVALID, ENV_STATE_NEWTON_SOLVED, ENV_STATE_VALID
from warp_ipc.utils.env_ops import reduce_env_energy_soft_shell
from warp_ipc.utils.make_pd import *
from warp_ipc.utils.make_pd import make_pd_6x6
from warp_ipc.utils.matrix import COOMatrix3x3
from warp_ipc.utils.svd import svd3x2
from warp_ipc.utils.wp_math import col_stack2
from warp_ipc.utils.wp_types import mat32d, mat46d, mat66d, mat96d, vec6d
if TYPE_CHECKING:
    from warp_ipc.sim_model import ASRModel

@wp.func
def f_ARAP(s: wp.float64, sHat: wp.float64, stretch_stiff: wp.float64, compress_stiff: wp.float64):
    pass

@wp.func
def g_ARAP(s: wp.float64, sHat: wp.float64, stretch_stiff: wp.float64, compress_stiff: wp.float64):
    pass

@wp.func
def h_ARAP(s: wp.float64, sHat: wp.float64, stretch_stiff: wp.float64, compress_stiff: wp.float64):
    pass

@wp.kernel
def compute_arap_membrane_energy_kernel(energy_buffer: wp.array(dtype=wp.float64), x: wp.array(dtype=wp.vec3d), tri_elems: wp.array(dtype=wp.vec3i), tri_elems_offset: wp.int32, IB: wp.array(dtype=wp.mat22d), elem_vol: wp.array(dtype=wp.float64), stretch_k: wp.array(dtype=wp.float64), compress_k: wp.array(dtype=wp.float64), scale: wp.float64):
    pass

@wp.kernel
def compute_arap_membrane_grad_kernel(soft_gradient_x: wp.array(dtype=wp.vec3d), x: wp.array(dtype=wp.vec3d), tri_elems: wp.array(dtype=wp.vec3i), tri_elems_offset: wp.int32, IBs: wp.array(dtype=wp.mat22d), elem_vol: wp.array(dtype=wp.float64), stretch_k: wp.array(dtype=wp.float64), compress_k: wp.array(dtype=wp.float64), scale: wp.float64, affine_verts_num: wp.int32, face2env: wp.array(dtype=wp.int32), env_states: wp.array(dtype=wp.int32)):
    pass

@wp.func
def Compute_DU_And_DV_Div_DF(U: wp.mat33d, _sigma: wp.vec2d, _V: wp.mat22d):
    pass

@wp.kernel
def compute_arap_membrane_hess_kernel(hess_soft_diag: COOMatrix3x3, hess_soft_elastic: COOMatrix3x3, x: wp.array(dtype=wp.vec3d), tri_elems: wp.array(dtype=wp.vec3i), tri_elems_offset: wp.int32, IBs: wp.array(dtype=wp.mat22d), elem_vol: wp.array(dtype=wp.float64), stretch_k: wp.array(dtype=wp.float64), compress_k: wp.array(dtype=wp.float64), scale: wp.float64, project_pd: wp.bool, affine_verts_num: wp.int32, affine_dofs: wp.int32, face2env: wp.array(dtype=wp.int32), env_states: wp.array(dtype=wp.int32)):
    pass

def val(x: wp.array, sim: 'ASRModel', scale: float, reduce_each: bool):
    if reduce_each:
        sim.sim_cache.soft_shell_energy.zero_()
    wp.launch(compute_arap_membrane_energy_kernel, dim=sim.soft_shell_IB.shape[0], inputs=[sim.sim_cache.soft_shell_energy, x, sim.face, sim.affine_tris_num, sim.soft_shell_IB, sim.soft_shell_elem_vol, sim.arap_stretch_k, sim.arap_compression_k, scale])
    if reduce_each:
        energy = wp.zeros(shape=sim.num_envs, dtype=wp.float64, device=sim.device)
        reduce_env_energy_soft_shell(sim.sim_cache.soft_shell_energy, energy, sim)
        return wp.to_torch(energy)
    else:
        return None

def grad(x: wp.array, sim: 'ASRModel', scale: float, soft_gradient_x: wp.array):
    wp.launch(compute_arap_membrane_grad_kernel, dim=sim.soft_shell_IB.shape[0], inputs=[soft_gradient_x, x, sim.face, sim.affine_tris_num, sim.soft_shell_IB, sim.soft_shell_elem_vol, sim.arap_stretch_k, sim.arap_compression_k, scale, sim.affine_verts_num, sim.face2env, sim.env_states])

def grad_adjoint(x: wp.array, sim: 'ASRModel', scale: float, soft_gradient_x: wp.array):
    wp.launch(compute_arap_membrane_grad_kernel, dim=sim.soft_shell_IB.shape[0], inputs=[soft_gradient_x, x, sim.face, sim.affine_tris_num, sim.soft_shell_IB, sim.soft_shell_elem_vol, sim.arap_stretch_k, sim.arap_compression_k, scale, sim.affine_verts_num], adjoint=True, adj_inputs=[None, None, None, 0, None, None, None, None, 0.0, 0])

def hess(x: wp.array, sim: 'ASRModel', scale: float, project_pd: bool=True):
    wp.launch(compute_arap_membrane_hess_kernel, dim=sim.soft_shell_IB.shape[0], inputs=[sim.hess_soft_diag, sim.hess_soft_shell_elastic, x, sim.face, sim.affine_tris_num, sim.soft_shell_IB, sim.soft_shell_elem_vol, sim.arap_stretch_k, sim.arap_compression_k, scale, project_pd, sim.affine_verts_num, sim.n_affine_dofs, sim.face2env, sim.env_states])