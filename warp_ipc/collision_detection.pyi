import _cython_3_0_12
import warp_ipc.utils.log as log
from warp_ipc.collision_kernels import bvh_query_aabb_EE as bvh_query_aabb_EE, bvh_query_aabb_PT as bvh_query_aabb_PT, compute_edge_BB as compute_edge_BB, compute_hs_collision as compute_hs_collision, compute_node_BB as compute_node_BB, compute_triangle_BB as compute_triangle_BB, init_step_size_collisions as init_step_size_collisions, init_step_size_hs as init_step_size_hs, init_step_size_inversion_free_kernel as init_step_size_inversion_free_kernel, prune_collisions as prune_collisions, prune_hs as prune_hs
from warp_ipc.utils.constants import AffineMaterialType as AffineMaterialType, BodyType as BodyType, MembraneType as MembraneType, VolMaterialType as VolMaterialType

BARRIER: int
FLOAT64_EPSILON: float
FRICTION: int
NUM_COLLISION_OFF_BLOCKS_PER_PAIR: int
TYPE_CHECKING: bool
__test__: dict
compute: _cython_3_0_12.cython_function_or_method
compute_hs: _cython_3_0_12.cython_function_or_method
init_step_size: _cython_3_0_12.cython_function_or_method
init_step_size_inversion_free: _cython_3_0_12.cython_function_or_method
prune: _cython_3_0_12.cython_function_or_method

class CollisionData:
    def __init__(self, *args, **kwargs) -> None: ...
    def copy_from(self, *args, **kwargs): ...
    def counter_set_zero(self, *args, **kwargs): ...
    def expand(self, *args, **kwargs): ...
    def requires_grad(self, *args, **kwargs): ...
