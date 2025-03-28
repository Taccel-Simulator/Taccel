import warp as wp
from warp_ipc.contact.distance_type import *
from warp_ipc.contact.line_line_distance import *
from warp_ipc.contact.point_line_distance import *
from warp_ipc.contact.point_point_distance import *

@wp.func
def edge_edge_distance(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d):
    pass

@wp.func
def edge_edge_distance_gradient(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d):
    pass

@wp.func
def edge_edge_distance_hessian(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d):
    pass