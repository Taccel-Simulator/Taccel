import warp as wp

@wp.func
def g_PE3D(v01: wp.float64, v02: wp.float64, v03: wp.float64, v11: wp.float64, v12: wp.float64, v13: wp.float64, v21: wp.float64, v22: wp.float64, v23: wp.float64):
    pass

@wp.func
def H_PE3D(v01: wp.float64, v02: wp.float64, v03: wp.float64, v11: wp.float64, v12: wp.float64, v13: wp.float64, v21: wp.float64, v22: wp.float64, v23: wp.float64):
    pass

@wp.func
def point_line_distance(p: wp.vec3d, e0: wp.vec3d, e1: wp.vec3d):
    pass

@wp.func
def point_line_distance_gradient(p: wp.vec3d, e0: wp.vec3d, e1: wp.vec3d):
    pass

@wp.func
def point_line_distance_hessian(p: wp.vec3d, e0: wp.vec3d, e1: wp.vec3d):
    pass