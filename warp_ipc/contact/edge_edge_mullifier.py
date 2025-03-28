import warp as wp

@wp.func
def g_EECN2(v01: wp.float64, v02: wp.float64, v03: wp.float64, v11: wp.float64, v12: wp.float64, v13: wp.float64, v21: wp.float64, v22: wp.float64, v23: wp.float64, v31: wp.float64, v32: wp.float64, v33: wp.float64):
    pass

@wp.func
def H_EECN2(v01: wp.float64, v02: wp.float64, v03: wp.float64, v11: wp.float64, v12: wp.float64, v13: wp.float64, v21: wp.float64, v22: wp.float64, v23: wp.float64, v31: wp.float64, v32: wp.float64, v33: wp.float64):
    pass

@wp.func
def EEM(input: wp.float64, eps_x: wp.float64):
    pass

@wp.func
def g_EEM(input: wp.float64, eps_x: wp.float64):
    pass

@wp.func
def H_EEM(input: wp.float64, eps_x: wp.float64):
    pass

@wp.func
def edge_edge_cross_norm2(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d):
    pass

@wp.func
def edge_edge_cross_norm2_gradient(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d):
    pass

@wp.func
def edge_edge_cross_norm2_hessian(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d):
    pass

@wp.func
def edge_edge_mollifier_threshold(ea0_rest: wp.vec3d, ea1_rest: wp.vec3d, eb0_rest: wp.vec3d, eb1_rest: wp.vec3d):
    pass

@wp.func
def edge_edge_mollifier(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d, eps_x: wp.float64):
    pass

@wp.func
def edge_edge_mollifier_gradient(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d, eps_x: wp.float64):
    pass

@wp.func
def edge_edge_mollifier_hessian(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d, eps_x: wp.float64):
    pass