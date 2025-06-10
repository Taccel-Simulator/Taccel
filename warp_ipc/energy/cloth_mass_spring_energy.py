from typing import TYPE_CHECKING
import warp as wp
import warp_ipc.utils.matrix as matrix
from warp_ipc.utils.constants import ENV_STATE_INVALID, ENV_STATE_NEWTON_SOLVED, BodyType, _1, id_3
from warp_ipc.utils.env_ops import reduce_env_energy_edge
from warp_ipc.utils.make_pd import *
from warp_ipc.utils.matrix import COOMatrix3x3
from warp_ipc.utils.wp_types import mat33d, mat66d
if TYPE_CHECKING:
    from warp_ipc.sim_model import ASRModel

@wp.kernel
def compute_edge_stretching_energy_kernel(energy: wp.array(dtype=wp.float64), X: wp.array(dtype=wp.vec3d), x: wp.array(dtype=wp.vec3d), edges: wp.array(dtype=wp.vec2i), edges_start: wp.int32, stiffness: wp.array(dtype=wp.float64)):
    print('[ERROR] Unexpected Recompilation: compute_edge_stretching_energy_kernel')

def val(x: wp.array, sim: 'ASRModel', scale: float, reduce_each: bool):
    sim.sim_cache.edge_energy.zero_()
    for (soft_edges_start, soft_edges_end, soft_type) in zip(sim.soft_stretching_edge_offsets[:-1], sim.soft_stretching_edge_offsets[1:], sim.soft_body_types[:sim.soft_shell_body_num]):
        if soft_type == BodyType.SOFT_VOLUMETRIC:
            raise NotImplementedError()
        soft_edges_num = soft_edges_end - soft_edges_start
        if reduce_each:
            sim.sim_cache.edge_energy.zero_()
        wp.launch(kernel=compute_edge_stretching_energy_kernel, dim=soft_edges_num, inputs=[sim.sim_cache.edge_energy, sim.X, x, sim.edge, soft_edges_start, sim.edge_stretching_stiffness])
        if reduce_each:
            energy = wp.zeros(shape=sim.num_envs, dtype=wp.float64, device=sim.device)
            reduce_env_energy_edge(sim.sim_cache.edge_energy, energy, sim)
            return wp.to_torch(energy) * scale
        else:
            return None

@wp.kernel
def compute_edge_stretching_grad_kernel(soft_gradient_x: wp.array(dtype=wp.vec3d), X: wp.array(dtype=wp.vec3d), x: wp.array(dtype=wp.vec3d), edges: wp.array(dtype=wp.vec2i), edges_start: wp.int32, stiffness: wp.array(dtype=wp.float64), affine_verts_num: wp.int32, scale: wp.float64, edge2env: wp.array(dtype=wp.int32), env_states: wp.array(dtype=wp.int32)):
    print('[ERROR] Unexpected Recompilation: compute_edge_stretching_grad_kernel')

def grad(x: wp.array, sim: 'ASRModel', scale: float, soft_gradient_x: wp.array):
    for (soft_edges_start, soft_edges_end, soft_type) in zip(sim.soft_stretching_edge_offsets[:-1], sim.soft_stretching_edge_offsets[1:], sim.soft_body_types[:sim.soft_shell_body_num]):
        if soft_type == BodyType.SOFT_VOLUMETRIC:
            raise NotImplementedError()
        soft_edges_num = soft_edges_end - soft_edges_start
        wp.launch(kernel=compute_edge_stretching_grad_kernel, dim=soft_edges_num, inputs=[soft_gradient_x, sim.X, x, sim.edge, soft_edges_start, sim.edge_stretching_stiffness, sim.affine_verts_num, scale, sim.edge2env, sim.env_states])

@wp.kernel
def compute_edge_stretching_hess_kernel(hess_soft_diag: COOMatrix3x3, hess_soft_elastic: COOMatrix3x3, X: wp.array(dtype=wp.vec3d), x: wp.array(dtype=wp.vec3d), edges: wp.array(dtype=wp.vec2i), edges_start: wp.int32, stiffness: wp.array(dtype=wp.float64), affine_verts_num: wp.int32, affine_dofs: wp.int32, scale: wp.float64, edge2env: wp.array(dtype=wp.int32), env_states: wp.array(dtype=wp.int32)):
    print('[ERROR] Unexpected Recompilation: compute_edge_stretching_hess_kernel')

def hess(x: wp.array, sim: 'ASRModel', scale):
    for (soft_edges_start, soft_edges_end, soft_type) in zip(sim.soft_stretching_edge_offsets[:-1], sim.soft_stretching_edge_offsets[1:], sim.soft_body_types[:sim.soft_shell_body_num]):
        if soft_type == BodyType.SOFT_VOLUMETRIC:
            raise NotImplementedError()
        soft_edges_num = soft_edges_end - soft_edges_start
        wp.launch(kernel=compute_edge_stretching_hess_kernel, dim=soft_edges_num, inputs=[sim.hess_soft_diag, sim.hess_soft_shell_elastic, sim.X, x, sim.edge, soft_edges_start, sim.edge_stretching_stiffness, sim.affine_verts_num, sim.n_affine_dofs, scale, sim.edge2env, sim.env_states])