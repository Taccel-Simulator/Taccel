import warp as wp
from warp_ipc.contact.distance_type import *
from warp_ipc.contact.point_line_distance import *
from warp_ipc.contact.point_plane_distance import *
from warp_ipc.contact.point_point_distance import *

@wp.func
def point_triangle_distance(p: wp.vec3d, t0: wp.vec3d, t1: wp.vec3d, t2: wp.vec3d) -> wp.float64:
    pass

@wp.func
def point_triangle_distance_gradient(p: wp.vec3d, t0: wp.vec3d, t1: wp.vec3d, t2: wp.vec3d):
    pass

@wp.func
def point_triangle_distance_hessian(p: wp.vec3d, t0: wp.vec3d, t1: wp.vec3d, t2: wp.vec3d):
    pass