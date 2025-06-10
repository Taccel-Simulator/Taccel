from typing import TYPE_CHECKING
import warp as wp
import warp.sparse as wps
import warp_ipc.collision_detection as collision_detection
import warp_ipc.contact.barrier as barrier
import warp_ipc.utils.matrix as matrix
from warp_ipc.contact.edge_edge_distance import edge_edge_distance
from warp_ipc.contact.edge_edge_tangent import edge_edge_tangent
from warp_ipc.contact.point_triangle_distance import point_triangle_distance
from warp_ipc.contact.point_triangle_tangent import point_triangle_tangent
from warp_ipc.energy.barrier_energy_gradient import *
from warp_ipc.energy.barrier_energy_hessian import *
from warp_ipc.utils.constants import *
from warp_ipc.utils.constants import NUM_COLLISION_OFF_BLOCKS_PER_PAIR
from warp_ipc.utils.env_ops import reduce_env_energy_vert
from warp_ipc.utils.matrix import COOMatrix3x3
from warp_ipc.utils.wp_math import get_combined_coulomb_friction, is_nan
from warp_ipc.utils.wp_types import vec12d
from ..utils.wp_types import vec12d
if TYPE_CHECKING:
    from warp_ipc.sim_model import ASRModel

@wp.kernel
def initialize_friction_hs(hs_lambda: wp.array(dtype=wp.float64), hs_node: wp.array(dtype=wp.int32), hs_ground: wp.array(dtype=wp.int32), x: wp.array(dtype=wp.vec3d), surf_vi: wp.array(dtype=wp.int32), node_xi: wp.array(dtype=wp.float64), node_area: wp.array(dtype=wp.float64), half_space_n: wp.array(dtype=wp.vec3d), half_space_o: wp.array(dtype=wp.vec3d), dhat: wp.float64, kappa: wp.float64):
    print('[ERROR] Unexpected Recompilation: initialize_friction_hs')

@wp.kernel
def initialize_friction_collisions(c_lambda: wp.array(dtype=wp.float64), closest_points: wp.array(dtype=wp.vec2d), normal: wp.array(dtype=wp.vec3d), nodeI: wp.array(dtype=wp.int32), nodeJ: wp.array(dtype=wp.int32), collision_type: wp.array(dtype=wp.int32), x: wp.array(dtype=wp.vec3d), surf_vi: wp.array(dtype=wp.int32), X: wp.array(dtype=wp.vec3d), node_xi: wp.array(dtype=wp.float64), edge_xi: wp.array(dtype=wp.float64), face_xi: wp.array(dtype=wp.float64), node_area: wp.array(dtype=wp.float64), edge_area: wp.array(dtype=wp.float64), edge: wp.array(dtype=wp.vec2i), face: wp.array(dtype=wp.vec3i), dhat: wp.float64, kappa: wp.float64, int_w_PTEE: wp.float64):
    print('[ERROR] Unexpected Recompilation: initialize_friction_collisions')

def initialize_friction(sim: 'ASRModel', cdw: collision_detection.CollisionData, x: wp.array):
    int_w_PTEE = wp.constant(wp.float64(0.25 if sim.handle_EE else 0.5))
    wp.launch(kernel=initialize_friction_hs, dim=int(cdw.num_hs_pair.numpy()[0]), inputs=[cdw.hs_lambda, cdw.hs_node, cdw.hs_ground, x, sim.surf_vi, sim.node_xi, sim.node_area, sim.half_space_n, sim.half_space_o, wp.float64(sim.dhat), wp.float64(sim.kappa)], device=sim.device)
    wp.launch(kernel=initialize_friction_collisions, dim=int(cdw.num_collisions.numpy()[0]), inputs=[cdw.c_lambda, cdw.closest_points, cdw.normal, cdw.nodeI, cdw.nodeJ, cdw.collision_type, x, sim.surf_vi, sim.X, sim.node_xi, sim.edge_xi, sim.face_xi, sim.node_area, sim.edge_area, sim.edge, sim.face, wp.float64(sim.dhat), wp.float64(sim.kappa), wp.float64(int_w_PTEE)], device=sim.device)
    wp.synchronize()

def initialize_friction_adjoint(sim: 'ASRModel', cdw: collision_detection.CollisionData, x: wp.array):
    int_w_PTEE = wp.constant(wp.float64(0.25 if sim.handle_EE else 0.5))
    wp.launch(kernel=initialize_friction_hs, dim=int(cdw.num_hs_pair.numpy()[0]), inputs=[cdw.hs_lambda, cdw.hs_node, cdw.hs_ground, x, sim.surf_vi, sim.node_xi, sim.node_area, sim.half_space_n, sim.half_space_o, wp.float64(sim.dhat), wp.float64(sim.kappa)], adjoint=True, adj_inputs=[None, None, None, None, None, None, None, None, None, 0.0, 0.0], device=sim.device)
    wp.launch(kernel=initialize_friction_collisions, dim=int(cdw.num_collisions.numpy()[0]), inputs=[cdw.c_lambda, cdw.closest_points, cdw.normal, cdw.nodeI, cdw.nodeJ, cdw.collision_type, x, sim.surf_vi, sim.X, sim.node_xi, sim.edge_xi, sim.face_xi, sim.node_area, sim.edge_area, sim.edge, sim.face, wp.float64(sim.dhat), wp.float64(sim.kappa), wp.float64(int_w_PTEE)], adjoint=True, adj_inputs=[None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.0, 0.0, 0.0], device=sim.device)

@wp.kernel
def val_IPC_hs(energy_x: wp.array(dtype=wp.float64), hs_lambda: wp.array(dtype=wp.float64), hs_node: wp.array(dtype=wp.int32), hs_ground: wp.array(dtype=wp.int32), x: wp.array(dtype=wp.vec3d), surf_vi: wp.array(dtype=wp.int32), hat_x: wp.array(dtype=wp.vec3d), node_xi: wp.array(dtype=wp.float64), node_area: wp.array(dtype=wp.float64), half_space_n: wp.array(dtype=wp.vec3d), half_space_o: wp.array(dtype=wp.vec3d), half_space_mu: wp.array(dtype=wp.float64), dhat: wp.float64, hat_h: wp.float64, kappa: wp.float64, scale: wp.float64, epsv: wp.float64, E: wp.int32):
    print('[ERROR] Unexpected Recompilation: val_IPC_hs')

@wp.kernel
def val_IPC_collisions(energy_x: wp.array(dtype=wp.float64), c_lambda: wp.array(dtype=wp.float64), closest_points: wp.array(dtype=wp.vec2d), normal: wp.array(dtype=wp.vec3d), nodeI: wp.array(dtype=wp.int32), nodeJ: wp.array(dtype=wp.int32), bodyI: wp.array(dtype=wp.int32), bodyJ: wp.array(dtype=wp.int32), collision_type: wp.array(dtype=wp.int32), x: wp.array(dtype=wp.vec3d), surf_vi: wp.array(dtype=wp.int32), hat_x: wp.array(dtype=wp.vec3d), X: wp.array(dtype=wp.vec3d), mu_body: wp.array(dtype=wp.float64), node_xi: wp.array(dtype=wp.float64), edge_xi: wp.array(dtype=wp.float64), face_xi: wp.array(dtype=wp.float64), node_area: wp.array(dtype=wp.float64), edge_area: wp.array(dtype=wp.float64), edge: wp.array(dtype=wp.vec2i), face: wp.array(dtype=wp.vec3i), dhat: wp.float64, hat_h: wp.float64, kappa: wp.float64, int_w_PTEE: wp.float64, scale: wp.float64, epsv: wp.float64, E: wp.int32):
    print('[ERROR] Unexpected Recompilation: val_IPC_collisions')

def val_IPC(E, sim: 'ASRModel', x: wp.array, cdw: collision_detection.CollisionData, hat_h: float, scale: float, energy_x: wp.array):
    num_connectivity = int(cdw.num_collisions.numpy()[0])
    num_hs_pair = int(cdw.num_hs_pair.numpy()[0])
    int_w_PTEE = wp.constant(wp.float64(0.25 if sim.handle_EE else 0.5))
    wp.launch(kernel=val_IPC_hs, dim=num_hs_pair, inputs=[energy_x, cdw.hs_lambda, cdw.hs_node, cdw.hs_ground, x, sim.surf_vi, sim.hat_x, sim.node_xi, sim.node_area, sim.half_space_n, sim.half_space_o, sim.half_space_mu, sim.dhat, wp.float64(hat_h), sim.kappa, wp.float64(scale), sim.epsv, E])
    wp.launch(kernel=val_IPC_collisions, dim=num_connectivity, inputs=[energy_x, cdw.c_lambda, cdw.closest_points, cdw.normal, cdw.nodeI, cdw.nodeJ, cdw.bodyI, cdw.bodyJ, cdw.collision_type, x, sim.surf_vi, sim.hat_x, sim.X, sim.mu_body, sim.node_xi, sim.edge_xi, sim.face_xi, sim.node_area, sim.edge_area, sim.edge, sim.face, wp.float64(sim.dhat), wp.float64(hat_h), wp.float64(sim.kappa), int_w_PTEE, wp.float64(scale), wp.float64(sim.epsv), E])

def val(E, sim: 'ASRModel', x: wp.array, cdw: collision_detection.CollisionData, hat_h: float, scale: float, reduce_each: bool):
    if reduce_each:
        sim.sim_cache.vert_energy.zero_()
    val_IPC(E, sim, x, cdw, hat_h, scale, sim.sim_cache.vert_energy)
    if reduce_each:
        energy = wp.zeros(shape=sim.num_envs, dtype=wp.float64, device=sim.device)
        reduce_env_energy_vert(sim.sim_cache.vert_energy, energy, sim)
        return wp.to_torch(energy)
    else:
        return None

def grad_IPC(E: wp.array, sim: 'ASRModel', x: wp.array, cdw: collision_detection.CollisionData, hat_h: float, scale: float, gradient_x: wp.array):
    num_connectivity = int(cdw.num_collisions.numpy()[0])
    num_hs_pair = int(cdw.num_hs_pair.numpy()[0])
    int_w_PTEE = wp.constant(wp.float64(0.25 if sim.handle_EE else 0.5))
    wp.launch(kernel=grad_IPC_hs, dim=num_hs_pair, inputs=[gradient_x, cdw.hs_lambda, cdw.hs_node, cdw.hs_ground, x, sim.surf_vi, sim.hat_x, sim.node_xi, sim.node_area, sim.half_space_n, sim.half_space_o, sim.half_space_mu, sim.dhat, wp.float64(hat_h), sim.kappa, wp.float64(scale), sim.epsv, E, sim.node2env, sim.env_states])
    wp.launch(kernel=grad_IPC_collisions, dim=num_connectivity, inputs=[gradient_x, cdw.c_lambda, cdw.closest_points, cdw.normal, cdw.nodeI, cdw.nodeJ, cdw.bodyI, cdw.bodyJ, cdw.collision_type, x, sim.surf_vi, sim.hat_x, sim.X, sim.mu_body, sim.node_xi, sim.edge_xi, sim.face_xi, sim.node_area, sim.edge_area, sim.edge, sim.face, wp.float64(sim.dhat), wp.float64(hat_h), wp.float64(sim.kappa), int_w_PTEE, wp.float64(scale), wp.float64(sim.epsv), E, sim.node2env, sim.env_states])

def grad_IPC_adjoint(E: wp.array, sim: 'ASRModel', x: wp.array, cdw: collision_detection.CollisionData, hat_h: float, scale: float, gradient_x: wp.array):
    num_connectivity = int(cdw.num_collisions.numpy()[0])
    num_hs_pair = int(cdw.num_hs_pair.numpy()[0])
    int_w_PTEE = wp.constant(wp.float64(0.25 if sim.handle_EE else 0.5))
    wp.launch(kernel=grad_IPC_hs, dim=num_hs_pair, inputs=[gradient_x, cdw.hs_lambda, cdw.hs_node, cdw.hs_ground, x, sim.surf_vi, sim.hat_x, sim.node_xi, sim.node_area, sim.half_space_n, sim.half_space_o, sim.half_space_mu, sim.dhat, wp.float64(hat_h), sim.kappa, wp.float64(scale), sim.epsv, E], adjoint=True, adj_inputs=[None, None, None, None, None, None, None, None, None, None, None, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0])
    wp.launch(kernel=grad_IPC_collisions, dim=num_connectivity, inputs=[gradient_x, cdw.c_lambda, cdw.closest_points, cdw.normal, cdw.nodeI, cdw.nodeJ, cdw.bodyI, cdw.bodyJ, cdw.collision_type, x, sim.surf_vi, sim.hat_x, sim.X, sim.mu_body, sim.node_xi, sim.edge_xi, sim.face_xi, sim.node_area, sim.edge_area, sim.edge, sim.face, wp.float64(sim.dhat), wp.float64(hat_h), wp.float64(sim.kappa), int_w_PTEE, wp.float64(scale), wp.float64(sim.epsv), E], adjoint=True, adj_inputs=[None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0])

@wp.kernel
def add_projected_x_to_y(projected_x: wp.array(dtype=vec12d), x: wp.array(dtype=wp.vec3d), node2body: wp.array(dtype=wp.int32), X: wp.array(dtype=wp.vec3d)):
    print('[ERROR] Unexpected Recompilation: add_projected_x_to_y')

@wp.kernel
def add_x_to_soft_x(x: wp.array(dtype=wp.vec3d), soft_x: wp.array(dtype=wp.vec3d), affine_verts_num: wp.int32):
    print('[ERROR] Unexpected Recompilation: add_x_to_soft_x')

def grad(E, sim: 'ASRModel', x: wp.array, cdw: collision_detection.CollisionData, hat_h: float, scale: float, gradient_x: wp.array, gradient_y: wp.array, soft_gradient_x: wp.array):
    gradient_x.zero_()
    grad_IPC(E, sim, x, cdw, hat_h, scale, gradient_x)
    wp.launch(kernel=add_projected_x_to_y, dim=sim.affine_verts_num, inputs=[gradient_y, gradient_x, sim.node2body, sim.X])
    wp.launch(kernel=add_x_to_soft_x, dim=sim.soft_verts_num, inputs=[gradient_x, soft_gradient_x, sim.affine_verts_num])

def grad_adjoint(E, sim: 'ASRModel', x: wp.array, cdw: collision_detection.CollisionData, hat_h: float, scale: float, gradient_x: wp.array, gradient_y: wp.array, soft_gradient_x: wp.array):
    wp.launch(kernel=add_x_to_soft_x, dim=sim.soft_verts_num, inputs=[gradient_x, soft_gradient_x, sim.affine_verts_num], adjoint=True, adj_inputs=[None, None, 0])
    wp.launch(kernel=add_projected_x_to_y, dim=sim.affine_verts_num, inputs=[gradient_y, gradient_x, sim.node2body, sim.X], adjoint=True, adj_inputs=[None, None, None, None])
    grad_IPC_adjoint(E, sim, x, cdw, hat_h, scale, gradient_x)

@wp.func
def add_hess12_by_mat3(hess_barrier: COOMatrix3x3, index: wp.int32, mat12: wp.mat(shape=(12, 12), dtype=wp.float64), bodyI: wp.int32, bodyJ: wp.int32):
    print('[ERROR] Unexpected Recompilation: add_hess12_by_mat3')

@wp.func
def add_diag_hess12_by_mat3(hess_affine_diag: COOMatrix3x3, index: wp.int32, mat12: wp.mat(shape=(12, 12), dtype=wp.float64)):
    print('[ERROR] Unexpected Recompilation: add_diag_hess12_by_mat3')

@wp.kernel
def assemble_matrix(hess_affine_diag: COOMatrix3x3, hess_barrier: COOMatrix3x3, bodyI: wp.array(dtype=wp.int32), bodyJ: wp.array(dtype=wp.int32), all_hess_b_y: wp.array(dtype=wp.mat(shape=(24, 24), dtype=wp.float64)), affine_body_num: wp.int32, num_connectivity: wp.int32):
    print('[ERROR] Unexpected Recompilation: assemble_matrix')

def hess(E, hess_barrier_coo: COOMatrix3x3, sim: 'ASRModel', x: wp.array, cdw, hat_h, scale, project_pd: bool=True, sm: wps.BsrMatrix | None=None):
    num_connectivity = int(cdw.num_collisions.numpy()[0])
    num_hs_pair = int(cdw.num_hs_pair.numpy()[0])
    matrix.COOMatrix3x3_resize(hess_barrier_coo, NUM_COLLISION_OFF_BLOCKS_PER_PAIR * num_connectivity)
    matrix.COOMatrix3x3_zero_(hess_barrier_coo)
    int_w_PTEE = wp.constant(wp.float64(0.25 if sim.handle_EE else 0.5))
    wp.launch(kernel=hess_hs, dim=num_hs_pair, inputs=[sim.hess_soft_diag, sim.hess_affine_diag, cdw.hs_lambda, cdw.hs_node, cdw.hs_body, cdw.hs_ground, sim.body_is_affine, x, sim.surf_vi, sim.hat_x, sim.X, sim.node_xi, sim.node_area, sim.half_space_n, sim.half_space_o, sim.half_space_mu, wp.float64(sim.dhat), wp.float64(hat_h), wp.float64(sim.kappa), wp.float64(scale), wp.float64(sim.epsv), wp.bool(project_pd), E, sim.affine_verts_num, sim.node2env, sim.env_states])
    wp.launch(kernel=hess_collisions, dim=num_connectivity, inputs=[sim.hess_affine_diag, sim.hess_soft_diag, hess_barrier_coo, cdw.c_lambda, cdw.closest_points, cdw.normal, cdw.nodeI, cdw.nodeJ, cdw.bodyI, cdw.bodyJ, sim.node2body, sim.face2body, cdw.collision_type, x, sim.surf_vi, sim.hat_x, sim.X, sim.mu_body, sim.body_is_affine, sim.node_xi, sim.edge_xi, sim.face_xi, sim.node_area, sim.edge_area, sim.edge, sim.face, wp.float64(sim.dhat), wp.float64(hat_h), wp.float64(sim.kappa), int_w_PTEE, wp.float64(scale), wp.float64(sim.epsv), wp.bool(project_pd), E, sim.affine_body_num, sim.affine_verts_num, sim.node2env, sim.env_states])
    wp.synchronize()
    if sm is not None:
        matrix.bsr_from_coos_inplace([hess_barrier_coo, sim.hess_affine_diag], sm, device=sm.device)