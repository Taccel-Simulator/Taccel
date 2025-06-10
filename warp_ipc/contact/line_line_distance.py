import warp as wp

@wp.func
def g_EE(v01: wp.float64, v02: wp.float64, v03: wp.float64, v11: wp.float64, v12: wp.float64, v13: wp.float64, v21: wp.float64, v22: wp.float64, v23: wp.float64, v31: wp.float64, v32: wp.float64, v33: wp.float64):
    print('[ERROR] Unexpected Recompilation: g_EE')

@wp.func
def H_EE(v01: wp.float64, v02: wp.float64, v03: wp.float64, v11: wp.float64, v12: wp.float64, v13: wp.float64, v21: wp.float64, v22: wp.float64, v23: wp.float64, v31: wp.float64, v32: wp.float64, v33: wp.float64):
    print('[ERROR] Unexpected Recompilation: H_EE')

@wp.func
def line_line_distance(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d):
    print('[ERROR] Unexpected Recompilation: line_line_distance')

@wp.func
def line_line_distance_gradient(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d):
    print('[ERROR] Unexpected Recompilation: line_line_distance_gradient')

@wp.func
def line_line_distance_hessian(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d):
    print('[ERROR] Unexpected Recompilation: line_line_distance_hessian')