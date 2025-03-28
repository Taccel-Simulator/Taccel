import warp as wp
from .utils.constants import _0, _1
from .utils.wp_types import vec12d

@wp.kernel
def set_sim_affine_state(affine_update_mask: wp.array(dtype=wp.int32), affine_rotations: wp.array(dtype=wp.mat33d), affine_translations: wp.array(dtype=wp.vec3d), y: wp.array(dtype=vec12d), y_hat: wp.array(dtype=vec12d), virtual_object_centers: wp.array(dtype=wp.vec3d), ABD_centers: wp.array(dtype=wp.vec3d)):
    pass

@wp.kernel
def set_sim_affine_vel(affine_update_mask: wp.array(dtype=wp.int32), affine_linear_vel: wp.array(dtype=wp.vec3d), affine_angular_vel: wp.array(dtype=wp.vec3d), y: wp.array(dtype=vec12d), v_y: wp.array(dtype=vec12d), virtual_object_centers: wp.array(dtype=wp.vec3d), ABD_centers: wp.array(dtype=wp.vec3d)):
    pass

@wp.kernel
def set_sim_soft_state(soft_update_mask: wp.array(dtype=wp.int32), soft_verts_positions: wp.array(dtype=wp.vec3d), x: wp.array(dtype=wp.vec3d), x_hat: wp.array(dtype=wp.vec3d), affine_verts_num: wp.int32):
    pass

@wp.kernel
def set_sim_soft_vel(soft_update_mask: wp.array(dtype=wp.int32), soft_verts_vel: wp.array(dtype=wp.vec3d), v_x: wp.array(dtype=wp.vec3d), affine_verts_num: wp.int32):
    pass