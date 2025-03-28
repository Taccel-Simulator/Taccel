import warp as wp
from warp_ipc.contact.edge_edge_distance import *

@wp.func
def edge_edge_ccd(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d, dea0: wp.vec3d, dea1: wp.vec3d, deb0: wp.vec3d, deb1: wp.vec3d, eta: wp.float64, thickness: wp.float64) -> wp.float64:
    pass