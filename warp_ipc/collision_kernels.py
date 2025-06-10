from typing import TYPE_CHECKING
import numpy as np
import warp as wp
from warp_ipc.contact.edge_edge_ccd import edge_edge_ccd
from warp_ipc.contact.edge_edge_distance import edge_edge_distance
from warp_ipc.contact.point_triangle_ccd import point_triangle_ccd
from warp_ipc.contact.point_triangle_distance import point_triangle_distance
from warp_ipc.utils.constants import *
from warp_ipc.utils.constants import ENV_STATE_INVALID, ENV_STATE_NEWTON_SOLVED, _0, _1
from warp_ipc.utils.wp_math import get_smallest_positive_real_cubic_root
if TYPE_CHECKING:
    pass

@wp.kernel
def prune_hs(new_num_hs_pair: wp.array(dtype=wp.int32), new_hs_node: wp.array(dtype=wp.int32), new_hs_body: wp.array(dtype=wp.int32), new_hs_ground: wp.array(dtype=wp.int32), hs_node: wp.array(dtype=wp.int32), hs_body: wp.array(dtype=wp.int32), hs_ground: wp.array(dtype=wp.int32), x: wp.array(dtype=wp.vec3d), surf_vi: wp.array(dtype=wp.int32), node_xi: wp.array(dtype=wp.float64), half_space_n: wp.array(dtype=wp.vec3d), half_space_o: wp.array(dtype=wp.vec3d), half_space_mu: wp.array(dtype=wp.float64), dhat: wp.float64, exclude_zero_friction: wp.bool):
    print('[ERROR] Unexpected Recompilation: prune_hs')

@wp.kernel
def prune_collisions(new_num_collisions: wp.array(dtype=wp.int32), new_nodeI: wp.array(dtype=wp.int32), new_nodeJ: wp.array(dtype=wp.int32), new_bodyI: wp.array(dtype=wp.int32), new_bodyJ: wp.array(dtype=wp.int32), new_collision_type: wp.array(dtype=wp.int32), nodeI: wp.array(dtype=wp.int32), nodeJ: wp.array(dtype=wp.int32), bodyI: wp.array(dtype=wp.int32), bodyJ: wp.array(dtype=wp.int32), collision_type: wp.array(dtype=wp.int32), x: wp.array(dtype=wp.vec3d), surf_vi: wp.array(dtype=wp.int32), node_xi: wp.array(dtype=wp.float64), edge_xi: wp.array(dtype=wp.float64), face_xi: wp.array(dtype=wp.float64), edge: wp.array(dtype=wp.vec2i), face: wp.array(dtype=wp.vec3i), mu_body: wp.array(dtype=wp.float64), dhat: wp.float64, exclude_zero_friction: wp.bool):
    print('[ERROR] Unexpected Recompilation: prune_collisions')

@wp.kernel
def init_step_size_hs(alpha_hs: wp.array(dtype=wp.float64), env_hs: wp.array(dtype=wp.int32), node2body: wp.array(dtype=wp.int32), body_env_id: wp.array(dtype=wp.int32), env_states: wp.array(dtype=wp.int32), x: wp.array(dtype=wp.vec3d), surf_vi: wp.array(dtype=wp.int32), p_x: wp.array(dtype=wp.vec3d), node_xi: wp.array(dtype=wp.float64), hs_node: wp.array(dtype=wp.int32), hs_ground: wp.array(dtype=wp.int32), half_space_n: wp.array(dtype=wp.vec3d), half_space_o: wp.array(dtype=wp.vec3d), eta: wp.float64):
    print('[ERROR] Unexpected Recompilation: init_step_size_hs')

@wp.kernel
def init_step_size_collisions(alpha_PTEE: wp.array(dtype=wp.float64), env_PTEE: wp.array(dtype=wp.int32), node2body: wp.array(dtype=wp.int32), body_env_id: wp.array(dtype=wp.int32), env_states: wp.array(dtype=wp.int32), nodeI: wp.array(dtype=wp.int32), nodeJ: wp.array(dtype=wp.int32), collision_type: wp.array(dtype=wp.int32), x: wp.array(dtype=wp.vec3d), surf_vi: wp.array(dtype=wp.int32), p_x: wp.array(dtype=wp.vec3d), node_xi: wp.array(dtype=wp.float64), edge_xi: wp.array(dtype=wp.float64), face_xi: wp.array(dtype=wp.float64), edge: wp.array(dtype=wp.vec2i), face: wp.array(dtype=wp.vec3i), eta: wp.float64):
    print('[ERROR] Unexpected Recompilation: init_step_size_collisions')

@wp.kernel
def init_step_size_inversion_free_kernel(x: wp.array(dtype=wp.vec3d), p_x: wp.array(dtype=wp.vec3d), tet_elems: wp.array(dtype=wp.vec4i), alphas: wp.array(dtype=wp.float64), node2body: wp.array(dtype=wp.int32), body_env_id: wp.array(dtype=wp.int32), env_states: wp.array(dtype=wp.int32), tet_envs: wp.array(dtype=wp.int32), img_tol: wp.float64, cubic_coef_tol: wp.float64):
    print('[ERROR] Unexpected Recompilation: init_step_size_inversion_free_kernel')

@wp.kernel
def compute_triangle_BB(triangleBB_lower: wp.array(dtype=wp.vec3f), triangleBB_upper: wp.array(dtype=wp.vec3f), x: wp.array(dtype=wp.vec3d), dx: wp.array(dtype=wp.vec3d), triangle: wp.array(dtype=wp.vec3i), face_xi: wp.array(dtype=wp.float64), dist: wp.float64):
    print('[ERROR] Unexpected Recompilation: compute_triangle_BB')

@wp.kernel
def compute_edge_BB(edgeBB_lower: wp.array(dtype=wp.vec3f), edgeBB_upper: wp.array(dtype=wp.vec3f), x: wp.array(dtype=wp.vec3d), dx: wp.array(dtype=wp.vec3d), edge: wp.array(dtype=wp.vec2i), edge_xi: wp.array(dtype=wp.float64), dist: wp.float64):
    print('[ERROR] Unexpected Recompilation: compute_edge_BB')

@wp.kernel
def compute_node_BB(nodeBB_lower: wp.array(dtype=wp.vec3f), nodeBB_upper: wp.array(dtype=wp.vec3f), x: wp.array(dtype=wp.vec3d), dx: wp.array(dtype=wp.vec3d), node_xi: wp.array(dtype=wp.float64), dist: wp.float64, surf_vi: wp.array(dtype=wp.int32)):
    print('[ERROR] Unexpected Recompilation: compute_node_BB')

@wp.kernel
def compute_hs_collision(hs_node: wp.array(dtype=wp.int32), hs_body: wp.array(dtype=wp.int32), hs_ground: wp.array(dtype=wp.int32), num_hs_pair: wp.array(dtype=wp.int32), num_HS: wp.int32, x: wp.array(dtype=wp.vec3d), surf_vi: wp.array(dtype=wp.int32), dx: wp.array(dtype=wp.vec3d), node_xi: wp.array(dtype=wp.float64), node2body: wp.array(dtype=wp.int32), half_space_n: wp.array(dtype=wp.vec3d), half_space_o: wp.array(dtype=wp.vec3d), dhat: wp.float64):
    print('[ERROR] Unexpected Recompilation: compute_hs_collision')

@wp.func
def subset3(s: wp.vec3i, i: wp.int32):
    print('[ERROR] Unexpected Recompilation: subset3')

@wp.func
def subset2(s: wp.vec2i, i: wp.int32):
    print('[ERROR] Unexpected Recompilation: subset2')

@wp.kernel
def bvh_query_aabb_PT(num_collisions: wp.array(dtype=wp.int32), PT_count: wp.array(dtype=wp.int32), nodeI: wp.array(dtype=wp.int32), nodeJ: wp.array(dtype=wp.int32), bodyI: wp.array(dtype=wp.int32), bodyJ: wp.array(dtype=wp.int32), collision_type: wp.array(dtype=wp.int32), PT_bvh_id: wp.uint64, nodeBB_lower: wp.array(dtype=wp.vec3), nodeBB_upper: wp.array(dtype=wp.vec3), face: wp.array(dtype=wp.vec3i), face2body: wp.array(dtype=wp.int32), node2body: wp.array(dtype=wp.int32), surf_vi: wp.array(dtype=wp.int32), body_enable_self_collision: wp.array(dtype=wp.int32), body_env_id: wp.array(dtype=wp.int32), env_states: wp.array(dtype=wp.int32), body_collision_layer: wp.array(dtype=wp.int64), collision_layer_filter: wp.array(dtype=wp.int64), stitch_map: wp.array2d(dtype=wp.int32), num_stitch_per_x: wp.array(dtype=wp.int32), max_collision: wp.int32, explode: wp.array(dtype=wp.int32)):
    print('[ERROR] Unexpected Recompilation: bvh_query_aabb_PT')

@wp.kernel
def bvh_query_aabb_EE(num_collisions: wp.array(dtype=wp.int32), EE_count: wp.array(dtype=wp.int32), nodeI: wp.array(dtype=wp.int32), nodeJ: wp.array(dtype=wp.int32), bodyI: wp.array(dtype=wp.int32), bodyJ: wp.array(dtype=wp.int32), collision_type: wp.array(dtype=wp.int32), EE_bvh_id: wp.uint64, edgeBB_lower: wp.array(dtype=wp.vec3), edgeBB_upper: wp.array(dtype=wp.vec3), edge: wp.array(dtype=wp.vec2i), edge2body: wp.array(dtype=wp.int32), body_enable_self_collision: wp.array(dtype=wp.int32), body_env_id: wp.array(dtype=wp.int32), env_states: wp.array(dtype=wp.int32), body_collision_layer: wp.array(dtype=wp.int64), collision_layer_filter: wp.array(dtype=wp.int64), stitch_map: wp.array2d(dtype=wp.int32), num_stitch_per_x: wp.array(dtype=wp.int32), max_collision: wp.int32, explode: wp.array(dtype=wp.int32)):
    print('[ERROR] Unexpected Recompilation: bvh_query_aabb_EE')