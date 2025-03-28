import warp as wp

@wp.func
def solveLdlt2D(A00: wp.float64, A01: wp.float64, A11: wp.float64, b0: wp.float64, b1: wp.float64):
    pass

@wp.func
def point_triangle_distance_type(p: wp.vec3d, t0: wp.vec3d, t1: wp.vec3d, t2: wp.vec3d) -> int:
    pass

@wp.func
def edge_edge_distance_type(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d) -> int:
    pass