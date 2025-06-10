import warp as wp

@wp.func
def g_PT(v01: wp.float64, v02: wp.float64, v03: wp.float64, v11: wp.float64, v12: wp.float64, v13: wp.float64, v21: wp.float64, v22: wp.float64, v23: wp.float64, v31: wp.float64, v32: wp.float64, v33: wp.float64):
    print('[ERROR] Unexpected Recompilation: g_PT')

@wp.func
def H_PT(v01: wp.float64, v02: wp.float64, v03: wp.float64, v11: wp.float64, v12: wp.float64, v13: wp.float64, v21: wp.float64, v22: wp.float64, v23: wp.float64, v31: wp.float64, v32: wp.float64, v33: wp.float64):
    print('[ERROR] Unexpected Recompilation: H_PT')

@wp.func
def point_plane_distance(p: wp.vec3d, t0: wp.vec3d, t1: wp.vec3d, t2: wp.vec3d):
    print('[ERROR] Unexpected Recompilation: point_plane_distance')

@wp.func
def point_plane_distance_gradient(p: wp.vec3d, t0: wp.vec3d, t1: wp.vec3d, t2: wp.vec3d):
    print('[ERROR] Unexpected Recompilation: point_plane_distance_gradient')

@wp.func
def point_plane_distance_hessian(p: wp.vec3d, t0: wp.vec3d, t1: wp.vec3d, t2: wp.vec3d):
    print('[ERROR] Unexpected Recompilation: point_plane_distance_hessian')