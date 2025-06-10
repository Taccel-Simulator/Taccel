from typing import TYPE_CHECKING
import warp as wp
import warp_ipc.utils.matrix as matrix
from warp_ipc.utils.constants import ENV_STATE_INVALID, ENV_STATE_NEWTON_SOLVED
from warp_ipc.utils.env_ops import reduce_env_energy_affine_body, reduce_env_energy_soft_vert
from warp_ipc.utils.wp_math import sqr
from warp_ipc.utils.wp_types import vec12d
if TYPE_CHECKING:
    from warp_ipc.sim_model import ASRModel

@wp.kernel
def init_affine_kinematic_target_kernel(affine_has_constraint: wp.array(dtype=wp.bool), affine_kinematic_target_pose: wp.array(dtype=vec12d), affine_target_dof: wp.array(dtype=vec12d), virtual_object_centers: wp.array(dtype=wp.vec3d), ABD_centers: wp.array(dtype=wp.vec3d)):
    print('[ERROR] Unexpected Recompilation: init_affine_kinematic_target_kernel')

def init(sim: 'ASRModel'):
    helper = sim.kinematic_helper
    wp.launch(kernel=init_affine_kinematic_target_kernel, dim=sim.affine_body_num, inputs=[helper.affine_has_constraint, helper.affine_target_transform, helper.affine_target_dof, sim.virtual_object_centers, sim.ABD_centers])

@wp.kernel
def compute_affine_kinematic_energy(y: wp.array(dtype=vec12d), affine_has_constraint: wp.array(dtype=wp.bool), affine_target_dof: wp.array(dtype=vec12d), weight: wp.float64, affine_energy: wp.array(dtype=wp.float64), mass_body: wp.array(dtype=wp.float64)):
    print('[ERROR] Unexpected Recompilation: compute_affine_kinematic_energy')

@wp.kernel
def compute_soft_kinematic_energy(x: wp.array(dtype=wp.vec3d), soft_has_constraint: wp.array(dtype=wp.bool), soft_target_dof: wp.array(dtype=wp.vec3d), weight: wp.float64, soft_energy: wp.array(dtype=wp.float64), affine_verts_num: wp.int32, soft_verts_mass: wp.array(dtype=wp.float64)):
    print('[ERROR] Unexpected Recompilation: compute_soft_kinematic_energy')

def val(x: wp.array, y: wp.array, sim: 'ASRModel', reduce_each: bool):
    if reduce_each:
        sim.sim_cache.soft_vert_energy.zero_()
        sim.sim_cache.affine_body_energy.zero_()
    helper = sim.kinematic_helper
    wp.launch(kernel=compute_affine_kinematic_energy, dim=sim.affine_body_num, inputs=[y, helper.affine_has_constraint, helper.affine_target_dof, helper._stiffness, sim.sim_cache.affine_body_energy, sim.mass_body])
    wp.launch(kernel=compute_soft_kinematic_energy, dim=sim.soft_verts_num, inputs=[x, helper.soft_has_constraint, helper.soft_target_dof, helper._stiffness, sim.sim_cache.soft_vert_energy, sim.affine_verts_num, sim.soft_verts_mass])
    if reduce_each:
        energy = wp.zeros(shape=sim.num_envs, dtype=wp.float64, device=sim.device)
        reduce_env_energy_affine_body(sim.sim_cache.affine_body_energy, energy, sim)
        reduce_env_energy_soft_vert(sim.sim_cache.soft_vert_energy, energy, sim)
        return wp.to_torch(energy)
    else:
        return None

@wp.kernel
def compute_affine_kinematic_grad(y: wp.array(dtype=vec12d), affine_has_constraint: wp.array(dtype=wp.bool), affine_target_dof: wp.array(dtype=vec12d), weight: wp.float64, affine_grad: wp.array(dtype=vec12d), mass_body: wp.array(dtype=wp.float64), body_env_id: wp.array(dtype=wp.int32), env_states: wp.array(dtype=wp.int32)):
    print('[ERROR] Unexpected Recompilation: compute_affine_kinematic_grad')

@wp.kernel
def compute_soft_kinematic_grad(x: wp.array(dtype=wp.vec3d), soft_has_constraint: wp.array(dtype=wp.bool), soft_target_dof: wp.array(dtype=wp.vec3d), weight: wp.float64, soft_grad: wp.array(dtype=wp.vec3d), affine_verts_num: wp.int32, soft_verts_mass: wp.array(dtype=wp.float64), node2env: wp.array(dtype=wp.int32), env_states: wp.array(dtype=wp.int32)):
    print('[ERROR] Unexpected Recompilation: compute_soft_kinematic_grad')

def grad(x, y, sim: 'ASRModel', soft_gradient_x: wp.array, affine_gradient_y: wp.array):
    helper = sim.kinematic_helper
    wp.launch(kernel=compute_affine_kinematic_grad, dim=sim.affine_body_num, inputs=[y, helper.affine_has_constraint, helper.affine_target_dof, helper._stiffness, affine_gradient_y, sim.mass_body, sim.body_env_id, sim.env_states])
    wp.launch(kernel=compute_soft_kinematic_grad, dim=sim.soft_verts_num, inputs=[x, helper.soft_has_constraint, helper.soft_target_dof, helper._stiffness, soft_gradient_x, sim.affine_verts_num, sim.soft_verts_mass, sim.node2env, sim.env_states])

def grad_adjoint(x, y, sim: 'ASRModel', soft_gradient_x: wp.array, affine_gradient_y: wp.array):
    helper = sim.kinematic_helper
    wp.launch(kernel=compute_affine_kinematic_grad, dim=sim.affine_body_num, inputs=[y, helper.affine_has_constraint, helper.affine_target_dof, helper._stiffness, affine_gradient_y, sim.mass_body], adjoint=True, adj_inputs=[None, None, None, 0.0, None, None])
    wp.launch(kernel=compute_soft_kinematic_grad, dim=sim.soft_verts_num, inputs=[x, helper.soft_has_constraint, helper.soft_target_dof, helper._stiffness, soft_gradient_x, sim.affine_verts_num, sim.soft_verts_mass], adjoint=True, adj_inputs=[None, None, None, 0.0, None, 0, None])

@wp.kernel
def compute_affine_kinematic_hess(affine_has_constraint: wp.array(dtype=wp.bool), weight: wp.float64, hess_affine_diag: matrix.COOMatrix3x3, mass_body: wp.array(dtype=wp.float64), body_env_id: wp.array(dtype=wp.int32), env_states: wp.array(dtype=wp.int32)):
    print('[ERROR] Unexpected Recompilation: compute_affine_kinematic_hess')

@wp.kernel
def compute_soft_kinematic_hess(soft_has_constraint: wp.array(dtype=wp.bool), weight: wp.float64, hess_soft_diag: matrix.COOMatrix3x3, affine_verts_num: wp.int32, soft_verts_mass: wp.array(dtype=wp.float64), node2env: wp.array(dtype=wp.int32), env_states: wp.array(dtype=wp.int32)):
    print('[ERROR] Unexpected Recompilation: compute_soft_kinematic_hess')

def hess(sim: 'ASRModel'):
    helper = sim.kinematic_helper
    wp.launch(kernel=compute_affine_kinematic_hess, dim=sim.affine_body_num, inputs=[helper.affine_has_constraint, helper._stiffness, sim.hess_affine_diag, sim.mass_body, sim.body_env_id, sim.env_states])
    wp.launch(kernel=compute_soft_kinematic_hess, dim=sim.soft_verts_num, inputs=[helper.soft_has_constraint, helper._stiffness, sim.hess_soft_diag, sim.affine_verts_num, sim.soft_verts_mass, sim.node2env, sim.env_states])