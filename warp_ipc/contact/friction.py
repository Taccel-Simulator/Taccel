import warp as wp

@wp.func
def f0(vbarnorm: wp.float64, epsv: wp.float64, hhat: wp.float64) -> wp.float64:
    pass

@wp.func
def f1_div_vbarnorm(vbarnorm: wp.float64, epsv: wp.float64) -> wp.float64:
    pass

@wp.func
def f_hess_term(vbarnorm: wp.float64, epsv: wp.float64) -> wp.float64:
    pass

@wp.func
def friction_energy(n: wp.vec3d, x: wp.vec3d, hat_x: wp.vec3d, hat_h: wp.float64, epsv: wp.float64, coeff: wp.float64) -> wp.float64:
    pass

@wp.func
def friction_gradient(n: wp.vec3d, x: wp.vec3d, hat_x: wp.vec3d, hat_h: wp.float64, epsv: wp.float64, coeff: wp.float64) -> wp.vec3d:
    pass

@wp.func
def friction_hessian(n: wp.vec3d, x: wp.vec3d, hat_x: wp.vec3d, hat_h: wp.float64, epsv: wp.float64, coeff: wp.float64) -> wp.mat33d:
    pass