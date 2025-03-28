import warp as wp
import warp_ipc.contact.barrier as barrier
import warp_ipc.contact.friction as friction
from warp_ipc.contact.edge_edge_distance import *
from warp_ipc.contact.edge_edge_mullifier import *
from warp_ipc.contact.edge_edge_tangent import *
from warp_ipc.contact.point_triangle_distance import *
from warp_ipc.contact.point_triangle_tangent import *
from warp_ipc.energy.barrier_energy_hessian import BARRIER, FRICTION
from warp_ipc.utils.constants import *
from warp_ipc.utils.constants import ENV_STATE_INVALID, ENV_STATE_NEWTON_SOLVED, ENV_STATE_VALID
from warp_ipc.utils.wp_math import get_combined_coulomb_friction
from warp_ipc.utils.wp_types import vec12d

@wp.kernel
def grad_IPC_hs(gradient_x: wp.array(dtype=wp.vec3d), hs_lambda: wp.array(dtype=wp.float64), hs_node: wp.array(dtype=wp.int32), hs_ground: wp.array(dtype=wp.int32), x: wp.array(dtype=wp.vec3d), surf_vi: wp.array(dtype=wp.int32), hat_x: wp.array(dtype=wp.vec3d), node_xi: wp.array(dtype=wp.float64), node_area: wp.array(dtype=wp.float64), half_space_n: wp.array(dtype=wp.vec3d), half_space_o: wp.array(dtype=wp.vec3d), half_space_mu: wp.array(dtype=wp.float64), dhat: wp.float64, hat_h: wp.float64, kappa: wp.float64, scale: wp.float64, epsv: wp.float64, E: wp.int32, node2env: wp.array(dtype=wp.int32), env_states: wp.array(dtype=wp.int32)):
    pass

@wp.kernel
def grad_IPC_collisions(gradient_x: wp.array(dtype=wp.vec3d), c_lambda: wp.array(dtype=wp.float64), closest_points: wp.array(dtype=wp.vec2d), normal: wp.array(dtype=wp.vec3d), nodeI: wp.array(dtype=wp.int32), nodeJ: wp.array(dtype=wp.int32), bodyI: wp.array(dtype=wp.int32), bodyJ: wp.array(dtype=wp.int32), collision_type: wp.array(dtype=wp.int32), x: wp.array(dtype=wp.vec3d), surf_vi: wp.array(dtype=wp.int32), hat_x: wp.array(dtype=wp.vec3d), X: wp.array(dtype=wp.vec3d), mu_body: wp.array(dtype=wp.float64), node_xi: wp.array(dtype=wp.float64), edge_xi: wp.array(dtype=wp.float64), face_xi: wp.array(dtype=wp.float64), node_area: wp.array(dtype=wp.float64), edge_area: wp.array(dtype=wp.float64), edge: wp.array(dtype=wp.vec2i), face: wp.array(dtype=wp.vec3i), dhat: wp.float64, hat_h: wp.float64, kappa: wp.float64, int_w_PTEE: wp.float64, scale: wp.float64, epsv: wp.float64, E: wp.int32, node2env: wp.array(dtype=wp.int32), env_states: wp.array(dtype=wp.int32)):
    pass