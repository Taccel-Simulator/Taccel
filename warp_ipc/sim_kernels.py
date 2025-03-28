import warp as wp
import warp_ipc.utils.matrix as matrix
from warp_ipc.utils.constants import ENV_STATE_INVALID, ENV_STATE_NEWTON_SOLVED
from warp_ipc.utils.matrix import COOMatrix3x3
from warp_ipc.utils.wp_types import vec12d, vec12i

@wp.kernel
def update_x_kernel(sim_x: wp.array(dtype=wp.vec3d), new_x: wp.array(dtype=wp.vec3d)):
    pass

@wp.kernel
def initialize_tilde_y(tilde_y: wp.array(dtype=wp.vec(length=12, dtype=wp.float64)), y: wp.array(dtype=wp.vec(length=12, dtype=wp.float64)), v_y: wp.array(dtype=wp.vec(length=12, dtype=wp.float64)), dt: wp.float64, time_int_rule: wp.int32):
    pass

@wp.kernel
def initialize_soft_tilde_x(soft_tilde_x: wp.array(dtype=wp.vec3d), x: wp.array(dtype=wp.vec3d), v_x: wp.array(dtype=wp.vec3d), dt: wp.float64, affine_verts_num: wp.int32, time_int_rule: wp.int32):
    pass

@wp.kernel
def negate_arr_vec3d(x: wp.array(dtype=wp.vec3d)):
    pass

@wp.kernel
def negate_arr_vec12d(x: wp.array(dtype=vec12d)):
    pass

@wp.kernel
def multiply_arr_vec3d_mul_scalar(x: wp.array(dtype=wp.vec3d)):
    pass

@wp.kernel
def multiply_arr_vec12d_mul_scalar(x: wp.array(dtype=vec12d)):
    pass

@wp.kernel
def absolutize_arr_vec3d(x: wp.array(dtype=wp.vec3d)):
    pass

@wp.kernel
def step_affine_y(y0: wp.array(dtype=wp.vec(12, dtype=wp.float64)), sys_direction: wp.array(dtype=wp.vec3d), stepsizes: wp.array(dtype=wp.float64), y: wp.array(dtype=wp.vec(12, dtype=wp.float64)), body_env_id: wp.array(dtype=wp.int32)):
    pass

@wp.kernel
def step_x(x0: wp.array(dtype=wp.vec3d), direction_x: wp.array(dtype=wp.vec3d), stepsizes: wp.array(dtype=wp.float64), x: wp.array(dtype=wp.vec3d), node2body: wp.array(dtype=wp.int32), body_env_id: wp.array(dtype=wp.int32)):
    pass

@wp.kernel
def clamp_search_direction(alpha_vol: wp.array(dtype=wp.float64), edge: wp.array(dtype=wp.vec2i), edge2body: wp.array(dtype=wp.int32), y: wp.array(dtype=wp.vec(length=12, dtype=wp.float64)), p_y: wp.array(dtype=wp.vec(length=12, dtype=wp.float64)), X: wp.array(dtype=wp.vec3d)):
    pass

@wp.kernel
def y_to_x(x: wp.array(dtype=wp.vec3d), X: wp.array(dtype=wp.vec3d), node2body: wp.array(dtype=wp.int32), y: wp.array(dtype=wp.vec(length=12, dtype=wp.float64))):
    pass

@wp.kernel
def sys_to_x_affine(sys_dir: wp.array(dtype=wp.vec3d), x_dir: wp.array(dtype=wp.vec3d), X: wp.array(dtype=wp.vec3d), node2body: wp.array(dtype=wp.int32)):
    pass

@wp.kernel
def sys_to_x_soft(sys_dir: wp.array(dtype=wp.vec3d), x_dir: wp.array(dtype=wp.vec3d), affine_verts_num: wp.int32, affine_body_num: wp.int32):
    pass

@wp.kernel
def safeguard_direction_x_kernel(x_dir: wp.array(dtype=wp.vec3d), node2env: wp.array(dtype=wp.int32), env_states: wp.array(dtype=wp.int32)):
    pass

@wp.kernel
def advection_y(yn: wp.array(dtype=wp.vec(length=12, dtype=wp.float64)), vn: wp.array(dtype=wp.vec(length=12, dtype=wp.float64)), y: wp.array(dtype=wp.vec(length=12, dtype=wp.float64)), a_y: wp.array(dtype=wp.vec(length=12, dtype=wp.float64)), dt: wp.float64, time_int_rule: wp.int32):
    pass

@wp.kernel
def advection_x(xn: wp.array(dtype=wp.vec3d), vn: wp.array(dtype=wp.vec3d), x: wp.array(dtype=wp.vec3d), a_x: wp.array(dtype=wp.vec3d), dt: wp.float64, time_int_rule: wp.int32):
    pass

@wp.kernel
def affine_to_sys_grad(affine_grad: wp.array(dtype=wp.vec(12, dtype=wp.float64)), sys_grad: wp.array(dtype=wp.vec3d)):
    pass

@wp.kernel
def soft_to_sys_grad(soft_grad: wp.array(dtype=wp.vec3d), sys_grad: wp.array(dtype=wp.vec3d), affine_dofs_div_3: wp.int32):
    pass

@wp.kernel
def init_soft_diag_hess_inds_kernel(hess_soft_diag: COOMatrix3x3, affine_body_num: wp.int32):
    pass

@wp.kernel
def init_affine_diag_hess_inds_kernel(hess_affine_diag: COOMatrix3x3):
    pass

@wp.kernel
def init_affine_mass_matrix_kernel(num_face: wp.int32, x: wp.array(dtype=wp.vec3d), face: wp.array(dtype=wp.vec3i), face2body: wp.array(dtype=wp.int32), body_is_affine: wp.array(dtype=wp.int32), affine_is_closed: wp.array(dtype=wp.int32), affine_flipped: wp.array(dtype=wp.int32), affine_mass_xi: wp.array(dtype=wp.float64), affine_density: wp.array(dtype=wp.float64), affine_center: wp.array(dtype=wp.vec3d), affine_mass_matrix: wp.array(dtype=wp.mat44d)):
    pass