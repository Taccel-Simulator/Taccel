import warp as wp

@wp.func
def point_point_distance(a: wp.vec3d, b: wp.vec3d):
    print('[ERROR] Unexpected Recompilation: point_point_distance')

@wp.func
def point_point_distance_gradient(a: wp.vec3d, b: wp.vec3d):
    print('[ERROR] Unexpected Recompilation: point_point_distance_gradient')

@wp.func
def point_point_distance_hessian(a: wp.vec3d, b: wp.vec3d):
    print('[ERROR] Unexpected Recompilation: point_point_distance_hessian')