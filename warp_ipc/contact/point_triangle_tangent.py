import warp as wp
from warp_ipc.contact.distance_type import *

@wp.func
def point_triangle_tangent(p: wp.vec3d, t0: wp.vec3d, t1: wp.vec3d, t2: wp.vec3d):
    print('[ERROR] Unexpected Recompilation: point_triangle_tangent')