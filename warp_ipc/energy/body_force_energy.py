from typing import TYPE_CHECKING
import warp as wp
from warp_ipc.utils.constants import ENV_STATE_INVALID, ENV_STATE_NEWTON_SOLVED
from warp_ipc.utils.env_ops import reduce_env_energy_affine_body, reduce_env_energy_soft_vert
from ..utils.wp_math import cat_4_vec3d, col_stack3
from ..utils.wp_types import vec12d
if TYPE_CHECKING:
    from warp_ipc.sim_model import ASRModel

@wp.kernel
def compute_body_force_energy_val_affine(energy: wp.array(dtype=wp.float64), y: wp.array(dtype=wp.vec(length=12, dtype=wp.float64)), hat_y: wp.array(dtype=wp.vec(length=12, dtype=wp.float64)), mass_matrix: wp.array(dtype=wp.mat44d), gravity: wp.vec3d, affine_ext_force: wp.array(dtype=wp.vec3d), affine_ext_y_force: wp.array(dtype=vec12d), scale: wp.float64, affine_has_constraint: wp.array(dtype=wp.bool)):
    print('[ERROR] Unexpected Recompilation: compute_body_force_energy_val_affine')

@wp.kernel
def compute_body_force_energy_val_soft(energy: wp.array(dtype=wp.float64), x: wp.array(dtype=wp.vec3d), soft_verts_mass: wp.array(dtype=wp.float64), gravity: wp.vec3d, scale: wp.float64, affine_verts_num: wp.int32, soft_has_constraint: wp.array(dtype=wp.bool)):
    print('[ERROR] Unexpected Recompilation: compute_body_force_energy_val_soft')

@wp.kernel
def compute_body_force_energy_grad_affine_y(gradient: wp.array(dtype=wp.vec(length=12, dtype=wp.float64)), hat_y: wp.array(dtype=wp.vec(length=12, dtype=wp.float64)), mass_matrix: wp.array(dtype=wp.mat44d), gravity: wp.vec3d, affine_ext_force: wp.array(dtype=wp.vec3d), affine_ext_y_force: wp.array(dtype=vec12d), scale: wp.float64, affine_has_constraint: wp.array(dtype=wp.bool), body_env_id: wp.array(dtype=wp.int32), env_states: wp.array(dtype=wp.int32)):
    print('[ERROR] Unexpected Recompilation: compute_body_force_energy_grad_affine_y')

@wp.kernel
def compute_body_force_energy_grad_soft_x(gradient: wp.array(dtype=wp.vec3d), soft_verts_mass: wp.array(dtype=wp.float64), gravity: wp.vec3d, scale: wp.float64, affine_verts_num: wp.int32, soft_has_constraint: wp.array(dtype=wp.bool), node2env: wp.array(dtype=wp.int32), env_states: wp.array(dtype=wp.int32)):
    print('[ERROR] Unexpected Recompilation: compute_body_force_energy_grad_soft_x')

def val(x, y, sim: 'ASRModel', scale: float, reduce_each: bool):
    helper = sim.kinematic_helper
    if reduce_each:
        sim.sim_cache.affine_body_energy.zero_()
        sim.sim_cache.soft_vert_energy.zero_()
    wp.launch(kernel=compute_body_force_energy_val_affine, dim=sim.affine_body_num, inputs=[sim.sim_cache.affine_body_energy, y, sim.hat_y, sim.affine_mass_matrix, sim.gravity, sim.affine_ext_force_field, sim.affine_ext_y_force, wp.float64(scale), helper.affine_has_constraint])
    wp.launch(kernel=compute_body_force_energy_val_soft, dim=sim.soft_verts_num, inputs=[sim.sim_cache.soft_vert_energy, x, sim.soft_verts_mass, sim.gravity, wp.float64(scale), sim.affine_verts_num, helper.soft_has_constraint])
    if reduce_each:
        energy = wp.zeros(shape=sim.num_envs, dtype=wp.float64, device=sim.device)
        reduce_env_energy_affine_body(sim.sim_cache.affine_body_energy, energy, sim)
        reduce_env_energy_soft_vert(sim.sim_cache.soft_vert_energy, energy, sim)
        return wp.to_torch(energy)
    else:
        return None

def grad(sim: 'ASRModel', scale: wp.array, affine_gradient_y: wp.array, soft_gradient_x: wp.array):
    helper = sim.kinematic_helper
    wp.launch(kernel=compute_body_force_energy_grad_affine_y, dim=sim.affine_body_num, inputs=[affine_gradient_y, sim.hat_y, sim.affine_mass_matrix, sim.gravity, sim.affine_ext_force_field, sim.affine_ext_y_force, wp.float64(scale), helper.affine_has_constraint, sim.body_env_id, sim.env_states])
    wp.launch(kernel=compute_body_force_energy_grad_soft_x, dim=sim.soft_verts_num, inputs=[soft_gradient_x, sim.soft_verts_mass, sim.gravity, wp.float64(scale), sim.affine_verts_num, helper.soft_has_constraint, sim.node2env, sim.env_states])

def grad_adjoint(sim: 'ASRModel', scale: wp.array, affine_gradient_y: wp.array, soft_gradient_x: wp.array):
    helper = sim.kinematic_helper
    wp.launch(kernel=compute_body_force_energy_grad_affine_y, dim=sim.affine_body_num, inputs=[affine_gradient_y, sim.hat_y, sim.affine_mass_matrix, sim.gravity, sim.affine_ext_force_field, sim.affine_ext_y_force, wp.float64(scale), helper.affine_has_constraint], adjoint=True, adj_inputs=[None, None, wp.vec3d(), 0.0, None])
    wp.launch(kernel=compute_body_force_energy_grad_soft_x, dim=sim.soft_verts_num, inputs=[soft_gradient_x, sim.soft_verts_mass, sim.gravity, wp.float64(scale), helper.soft_has_constraint], adjoint=True, adj_inputs=[None, None, wp.vec3d(), 0.0, None])