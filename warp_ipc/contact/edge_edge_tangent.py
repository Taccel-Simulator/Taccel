import warp as wp
from warp_ipc.contact.distance_type import *

@wp.func
def edge_edge_tangent(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d):
    print('[ERROR] Unexpected Recompilation: edge_edge_tangent')