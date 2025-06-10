from typing import TYPE_CHECKING
import warp as wp
import warp_ipc.utils.matrix as matrix
from warp_ipc.utils.constants import ENV_STATE_INVALID, ENV_STATE_NEWTON_SOLVED
from warp_ipc.utils.env_ops import reduce_env_energy_affine_body
from warp_ipc.utils.make_pd import make_pd_9x9, make_pd_12x12
from warp_ipc.utils.matrix import COOMatrix3x3
from ..utils.wp_types import vec12d
if TYPE_CHECKING:
    from warp_ipc.sim_model import ASRModel

@wp.kernel
def compute_rigidity_energy_val(energy: wp.array(dtype=wp.float64), y: wp.array(dtype=wp.vec(length=12, dtype=wp.float64)), vol_body: wp.array(dtype=wp.float64), E_body: wp.array(dtype=wp.float64), scale: wp.float64, affine_has_constraint: wp.array(dtype=wp.bool)):
    print('[ERROR] Unexpected Recompilation: compute_rigidity_energy_val')

def val(y, sim: 'ASRModel', scale: float, reduce_each: bool):
    helper = sim.kinematic_helper
    if reduce_each:
        sim.sim_cache.affine_body_energy.zero_()
    wp.launch(kernel=compute_rigidity_energy_val, dim=sim.affine_body_num, inputs=[sim.sim_cache.affine_body_energy, y, sim.vol_body, sim.E_body, wp.float64(scale), helper.affine_has_constraint])
    if reduce_each:
        energy = wp.zeros(shape=sim.num_envs, dtype=wp.float64, device=sim.device)
        reduce_env_energy_affine_body(sim.sim_cache.affine_body_energy, energy, sim)
        return wp.to_torch(energy)
    else:
        return None

@wp.kernel
def compute_rigidity_energy_grad(gradient: wp.array(dtype=wp.vec(length=12, dtype=wp.float64)), y: wp.array(dtype=wp.vec(length=12, dtype=wp.float64)), vol_body: wp.array(dtype=wp.float64), E_body: wp.array(dtype=wp.float64), scale: wp.float64, affine_has_constraint: wp.array(dtype=wp.bool), body_env_id: wp.array(dtype=wp.int32), env_states: wp.array(dtype=wp.int32)):
    print('[ERROR] Unexpected Recompilation: compute_rigidity_energy_grad')

def grad(y, sim: 'ASRModel', scale, gradient):
    helper = sim.kinematic_helper
    wp.launch(kernel=compute_rigidity_energy_grad, dim=sim.affine_body_num, inputs=[gradient, y, sim.vol_body, sim.E_body, wp.float64(scale), helper.affine_has_constraint, sim.body_env_id, sim.env_states])

@wp.kernel
def compute_rigidity_energy_hess(hess_affine_diag: COOMatrix3x3, y: wp.array(dtype=wp.vec(length=12, dtype=wp.float64)), vol_body: wp.array(dtype=wp.float64), E_body: wp.array(dtype=wp.float64), scale: wp.float64, project_pd: wp.bool, affine_has_constraint: wp.array(dtype=wp.bool), body_env_id: wp.array(dtype=wp.int32), env_states: wp.array(dtype=wp.int32)):
    print('[ERROR] Unexpected Recompilation: compute_rigidity_energy_hess')

def hess(y, sim: 'ASRModel', scale, project_pd):
    helper = sim.kinematic_helper
    wp.launch(kernel=compute_rigidity_energy_hess, dim=sim.affine_body_num, inputs=[sim.hess_affine_diag, y, sim.vol_body, sim.E_body, wp.float64(scale), wp.bool(project_pd), helper.affine_has_constraint, sim.body_env_id, sim.env_states])