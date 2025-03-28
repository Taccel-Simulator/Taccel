import warp as wp
from .utils.constants import ENV_STATE_INVALID, ENV_STATE_NEWTON_SOLVED, ENV_STATE_VALID
from .utils.matrix import COOMatrix3x3
from .utils.wp_types import vec12d, vec12i

@wp.kernel
def update_dof_satisfied_kernel(x_has_constraint: wp.array(dtype=wp.bool), x: wp.array(dtype=wp.vec3d), x_target: wp.array(dtype=wp.vec3d), x_target_reached: wp.array(dtype=wp.bool), y_has_constraint: wp.array(dtype=wp.bool), y: wp.array(dtype=vec12d), y_target: wp.array(dtype=vec12d), y_target_reached: wp.array(dtype=wp.bool), dt: wp.float64, tol: wp.float64):
    pass

@wp.kernel
def project_system_kernel(offsets: wp.array(dtype=wp.int32), columns: wp.array(dtype=wp.int32), values: wp.array(dtype=wp.mat33d), gradient: wp.array(dtype=wp.vec3d), y_target_reached: wp.array(dtype=wp.bool), x_target_reached: wp.array(dtype=wp.bool)):
    pass

@wp.kernel
def check_is_satisfied_kernel(y_target_reached: wp.array(dtype=wp.bool), y_has_constraint: wp.array(dtype=wp.bool), x_target_reached: wp.array(dtype=wp.bool), x_has_constraint: wp.array(dtype=wp.bool), satisfied_val: wp.array(dtype=wp.bool), affine_verts_num: wp.int32, body_env_id: wp.array(dtype=wp.int32), node2env: wp.array(dtype=wp.int32), env_states: wp.array(dtype=wp.int32)):
    pass