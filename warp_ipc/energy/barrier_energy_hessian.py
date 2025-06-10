import warp as wp
import warp_ipc.contact.barrier as barrier
import warp_ipc.contact.friction as friction
from warp_ipc.contact.distance_type import *
from warp_ipc.contact.edge_edge_distance import *
from warp_ipc.contact.edge_edge_mullifier import *
from warp_ipc.contact.edge_edge_tangent import *
from warp_ipc.contact.point_triangle_distance import *
from warp_ipc.contact.point_triangle_tangent import *
from warp_ipc.utils import matrix as matrix
from warp_ipc.utils.constants import *
from warp_ipc.utils.constants import ENV_STATE_INVALID, ENV_STATE_NEWTON_SOLVED, NUM_COLLISION_OFF_BLOCKS_PER_PAIR
from warp_ipc.utils.make_pd import make_pd_3x3, make_pd_12x12
from warp_ipc.utils.matrix import COOMatrix3x3
from warp_ipc.utils.wp_math import get_combined_coulomb_friction
from warp_ipc.utils.wp_types import vec8i

@wp.func
def H_B_PT(x: wp.array(dtype=wp.vec3d), face: wp.array(dtype=wp.vec3i), node_xi: wp.array(dtype=wp.float64), face_xi: wp.array(dtype=wp.float64), xI: wp.int32, svi: wp.int32, faceJ: wp.int32, coeff: wp.float64, dhat: wp.float64, kappa: wp.float64, project_pd: wp.bool):
    print('[ERROR] Unexpected Recompilation: H_B_PT')

@wp.func
def H_F_PT(x: wp.array(dtype=wp.vec3d), hat_x: wp.array(dtype=wp.vec3d), face: wp.array(dtype=wp.vec3i), xI: wp.int32, faceJ: wp.int32, u: wp.float64, v: wp.float64, normal: wp.vec3d, hat_h: wp.float64, coeff: wp.float64, epsv: wp.float64, project_pd: wp.bool):
    print('[ERROR] Unexpected Recompilation: H_F_PT')

@wp.func
def H_B_EE(x: wp.array(dtype=wp.vec3d), X: wp.array(dtype=wp.vec3d), edge: wp.array(dtype=wp.vec2i), edge_xi: wp.array(dtype=wp.float64), edgeI: wp.int32, edgeJ: wp.int32, coeff: wp.float64, dhat: wp.float64, kappa: wp.float64, project_pd: wp.bool):
    print('[ERROR] Unexpected Recompilation: H_B_EE')

@wp.func
def H_F_EE(x: wp.array(dtype=wp.vec3d), hat_x: wp.array(dtype=wp.vec3d), edge: wp.array(dtype=wp.vec2i), edgeI: wp.int32, edgeJ: wp.int32, u: wp.float64, v: wp.float64, normal: wp.vec3d, hat_h: wp.float64, coeff: wp.float64, epsv: wp.float64, project_pd: wp.bool):
    print('[ERROR] Unexpected Recompilation: H_F_EE')

@wp.kernel
def hess_hs(hess_soft_diag: COOMatrix3x3, hess_affine_diag: COOMatrix3x3, hs_lambda: wp.array(dtype=wp.float64), hs_node: wp.array(dtype=wp.int32), hs_body: wp.array(dtype=wp.int32), hs_ground: wp.array(dtype=wp.int32), body_is_affine: wp.array(dtype=wp.int32), x: wp.array(dtype=wp.vec3d), surf_vi: wp.array(dtype=wp.int32), hat_x: wp.array(dtype=wp.vec3d), X: wp.array(dtype=wp.vec3d), node_xi: wp.array(dtype=wp.float64), node_area: wp.array(dtype=wp.float64), half_space_n: wp.array(dtype=wp.vec3d), half_space_o: wp.array(dtype=wp.vec3d), half_space_mu: wp.array(dtype=wp.float64), dhat: wp.float64, hat_h: wp.float64, kappa: wp.float64, scale: wp.float64, epsv: wp.float64, project_pd: wp.bool, E: wp.int32, affine_verts_num: wp.int32, node2env: wp.array(dtype=wp.int32), env_states: wp.array(dtype=wp.int32)):
    print('[ERROR] Unexpected Recompilation: hess_hs')

@wp.kernel
def hess_collisions(hess_affine_diag: COOMatrix3x3, hess_soft_diag: COOMatrix3x3, hess_barrier_coo: COOMatrix3x3, c_lambda: wp.array(dtype=wp.float64), closest_points: wp.array(dtype=wp.vec2d), normal: wp.array(dtype=wp.vec3d), nodeI: wp.array(dtype=wp.int32), nodeJ: wp.array(dtype=wp.int32), bodyI: wp.array(dtype=wp.int32), bodyJ: wp.array(dtype=wp.int32), node2body: wp.array(dtype=wp.int32), face2body: wp.array(dtype=wp.int32), collision_type: wp.array(dtype=wp.int32), x: wp.array(dtype=wp.vec3d), surf_vi: wp.array(dtype=wp.int32), hat_x: wp.array(dtype=wp.vec3d), X: wp.array(dtype=wp.vec3d), mu_body: wp.array(dtype=wp.float64), body_is_affine: wp.array(dtype=wp.int32), node_xi: wp.array(dtype=wp.float64), edge_xi: wp.array(dtype=wp.float64), face_xi: wp.array(dtype=wp.float64), node_area: wp.array(dtype=wp.float64), edge_area: wp.array(dtype=wp.float64), edge: wp.array(dtype=wp.vec2i), face: wp.array(dtype=wp.vec3i), dhat: wp.float64, hat_h: wp.float64, kappa: wp.float64, int_w_PTEE: wp.float64, scale: wp.float64, epsv: wp.float64, project_pd: wp.bool, E: wp.int32, affine_body_num: wp.int32, affine_verts_num: wp.int32, node2env: wp.array(dtype=wp.int32), env_states: wp.array(dtype=wp.int32)):
    print('[ERROR] Unexpected Recompilation: hess_collisions')