import warp as wp
from warp_ipc.contact.point_triangle_distance import *

@wp.func
def point_triangle_ccd(p: wp.vec3d, t0: wp.vec3d, t1: wp.vec3d, t2: wp.vec3d, dp: wp.vec3d, dt0: wp.vec3d, dt1: wp.vec3d, dt2: wp.vec3d, eta: wp.float64, thickness: wp.float64) -> wp.float64:
    pass