from typing import TYPE_CHECKING
import warp as wp
import warp_ipc.utils.matrix as matrix
from warp_ipc.utils.constants import ENV_STATE_INVALID, ENV_STATE_NEWTON_SOLVED, ENV_STATE_VALID
from warp_ipc.utils.env_ops import reduce_env_energy_affine_body, reduce_env_energy_soft_vert
from warp_ipc.utils.matrix import COOMatrix3x3
from ..utils.wp_types import vec12d
if TYPE_CHECKING:
    from warp_ipc.sim_model import ASRModel

@wp.kernel
def compute_inertia_energy_val_affine(energy: wp.array(dtype=wp.float64), y: wp.array(dtype=wp.vec(length=12, dtype=wp.float64)), tilde_y: wp.array(dtype=wp.vec(length=12, dtype=wp.float64)), mass_matrix: wp.array(dtype=wp.mat44d), affine_has_constraint: wp.array(dtype=wp.bool)):
    pass

@wp.kernel
def compute_inertia_energy_val_soft(energy: wp.array(dtype=wp.float64), x: wp.array(dtype=wp.vec3d), soft_tilde_x: wp.array(dtype=wp.vec3d), soft_verts_mass: wp.array(dtype=wp.float64), affine_verts_num: wp.int32, soft_has_constraint: wp.array(dtype=wp.bool)):
    pass

@wp.kernel
def compute_inertia_energy_grad_affine_y(gradient: wp.array(dtype=wp.vec(length=12, dtype=wp.float64)), y: wp.array(dtype=wp.vec(length=12, dtype=wp.float64)), tilde_y: wp.array(dtype=wp.vec(length=12, dtype=wp.float64)), mass_matrix: wp.array(dtype=wp.mat44d), affine_has_constraint: wp.array(dtype=wp.bool), body_env_id: wp.array(dtype=wp.int32), env_states: wp.array(dtype=wp.int32)):
    pass

@wp.kernel
def compute_inertia_energy_grad_soft_x(soft_gradient_x: wp.array(dtype=wp.vec3d), x: wp.array(dtype=wp.vec3d), soft_tilde_x: wp.array(dtype=wp.vec3d), soft_vert_mass: wp.array(dtype=wp.float64), affine_verts_num: wp.int32, soft_has_constraint: wp.array(dtype=wp.bool), node2env: wp.array(dtype=wp.int32), env_states: wp.array(dtype=wp.int32)):
    pass

@wp.kernel
def compute_inertia_energy_hess_affine(hess_affine_diag: COOMatrix3x3, mass_matrix: wp.array(dtype=wp.mat44d), affine_has_constraint: wp.array(dtype=wp.bool), body_env_id: wp.array(dtype=wp.int32), env_states: wp.array(dtype=wp.int32)):
    pass

@wp.kernel
def compute_inertia_energy_hess_soft(soft_verts_mass: wp.array(dtype=wp.float64), hess_soft_diag: COOMatrix3x3, affine_verts_num: wp.int32, soft_has_constraint: wp.array(dtype=wp.bool), node2env: wp.array(dtype=wp.int32), env_states: wp.array(dtype=wp.int32)):
    pass

def val(x: wp.array, y: wp.array, sim: 'ASRModel', reduce_each: bool):
    if reduce_each:
        sim.sim_cache.affine_body_energy.zero_()
        sim.sim_cache.soft_vert_energy.zero_()
    helper = sim.kinematic_helper
    wp.launch(kernel=compute_inertia_energy_val_affine, dim=sim.affine_body_num, inputs=[sim.sim_cache.affine_body_energy, y, sim.tilde_y, sim.affine_mass_matrix, helper.affine_has_constraint])
    wp.launch(kernel=compute_inertia_energy_val_soft, dim=sim.soft_verts_num, inputs=[sim.sim_cache.soft_vert_energy, x, sim.soft_tilde_x, sim.soft_verts_mass, sim.affine_verts_num, helper.soft_has_constraint])
    if reduce_each:
        energy = wp.zeros(shape=sim.num_envs, dtype=wp.float64, device=sim.device)
        reduce_env_energy_affine_body(sim.sim_cache.affine_body_energy, energy, sim)
        reduce_env_energy_soft_vert(sim.sim_cache.soft_vert_energy, energy, sim)
        return wp.to_torch(energy)
    else:
        return None

def grad(x: wp.array, y: wp.array, sim: 'ASRModel', affine_gradient_y: wp.array, soft_gradient_x: wp.array):
    helper = sim.kinematic_helper
    wp.launch(kernel=compute_inertia_energy_grad_affine_y, dim=sim.affine_body_num, inputs=[affine_gradient_y, y, sim.tilde_y, sim.affine_mass_matrix, helper.affine_has_constraint, sim.body_env_id, sim.env_states])
    wp.launch(kernel=compute_inertia_energy_grad_soft_x, dim=sim.soft_verts_num, inputs=[soft_gradient_x, x, sim.soft_tilde_x, sim.soft_verts_mass, sim.affine_verts_num, helper.soft_has_constraint, sim.node2env, sim.env_states])

def grad_adjoint(x: wp.array, y: wp.array, sim: 'ASRModel', affine_gradient_y: wp.array, soft_gradient_x: wp.array):
    helper = sim.kinematic_helper
    wp.launch(kernel=compute_inertia_energy_grad_affine_y, dim=sim.affine_body_num, inputs=[affine_gradient_y, y, sim.tilde_y, sim.affine_mass_matrix, helper.affine_has_constraint], adjoint=True, adj_inputs=[None, None, None, None, None])
    wp.launch(kernel=compute_inertia_energy_grad_soft_x, dim=sim.soft_verts_num, inputs=[soft_gradient_x, x, sim.soft_tilde_x, sim.soft_verts_mass, sim.affine_verts_num, helper.soft_has_constraint], adjoint=True, adj_inputs=[None, None, None, None, 0, None])

def hess(sim: 'ASRModel'):
    helper = sim.kinematic_helper
    wp.launch(kernel=compute_inertia_energy_hess_affine, dim=sim.affine_body_num, inputs=[sim.hess_affine_diag, sim.affine_mass_matrix, helper.affine_has_constraint, sim.body_env_id, sim.env_states])
    wp.launch(kernel=compute_inertia_energy_hess_soft, dim=sim.soft_verts_num, inputs=[sim.soft_verts_mass, sim.hess_soft_diag, sim.affine_verts_num, helper.soft_has_constraint, sim.node2env, sim.env_states])