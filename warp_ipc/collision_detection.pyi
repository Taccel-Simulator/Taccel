from .utils.constants import *
import warp as wp
from .collision_kernels import bvh_query_aabb_EE as bvh_query_aabb_EE, bvh_query_aabb_PT as bvh_query_aabb_PT, compute_edge_BB as compute_edge_BB, compute_hs_collision as compute_hs_collision, compute_node_BB as compute_node_BB, compute_triangle_BB as compute_triangle_BB, init_step_size_collisions as init_step_size_collisions, init_step_size_hs as init_step_size_hs, init_step_size_inversion_free_kernel as init_step_size_inversion_free_kernel, prune_collisions as prune_collisions, prune_hs as prune_hs
from .sim_model import ASRModel as ASRModel
from .utils import log as log
from .utils.constants import ENV_STATE_INVALID as ENV_STATE_INVALID, ENV_STATE_NEWTON_SOLVED as ENV_STATE_NEWTON_SOLVED
from _typeshed import Incomplete

class CollisionData:
    triangleBB_lower: Incomplete
    triangleBB_upper: Incomplete
    edgeBB_lower: Incomplete
    edgeBB_upper: Incomplete
    nodeBB_lower: Incomplete
    nodeBB_upper: Incomplete
    num_hs_pair: Incomplete
    hs_node: Incomplete
    hs_body: Incomplete
    hs_ground: Incomplete
    max_collisions: Incomplete
    num_collisions: Incomplete
    PT_count: Incomplete
    EE_count: Incomplete
    nodeI: Incomplete
    nodeJ: Incomplete
    bodyI: Incomplete
    bodyJ: Incomplete
    collision_type: Incomplete
    hs_lambda: Incomplete
    c_lambda: Incomplete
    closest_points: Incomplete
    normal: Incomplete
    def __init__(self, sim: ASRModel) -> None: ...
    def requires_grad(self, flag) -> None: ...
    def expand(self, max_collisions) -> None: ...
    def copy_from(self, cdw) -> None: ...
    def counter_set_zero(self) -> None: ...

def compute_hs(cdw: CollisionData, sim: ASRModel, x: wp.array, dx: wp.array): ...
def compute(cdw: CollisionData, sim: ASRModel, x: wp.array, dx: wp.array): ...
def prune(cdw, sim: ASRModel, exclude_zero_friction: bool, x: wp.array): ...
def init_step_size(sim: ASRModel, x: wp.array, cdw: CollisionData, p_x: wp.array, eta: float = 0.2): ...
def init_step_size_inversion_free(sim: ASRModel, x: wp.array, p_x: wp.array, inversion_free_im_tol: float, inversion_free_cubic_coef_tol: float): ...
