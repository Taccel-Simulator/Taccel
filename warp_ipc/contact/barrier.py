import warp as wp

@wp.func
def barrier(d2: wp.float64, dHat2: wp.float64, kappa: wp.float64):
    print('[ERROR] Unexpected Recompilation: barrier')

@wp.func
def barrier_gradient(d2: wp.float64, dHat2: wp.float64, kappa: wp.float64):
    print('[ERROR] Unexpected Recompilation: barrier_gradient')

@wp.func
def barrier_hessian(d2: wp.float64, dHat2: wp.float64, kappa: wp.float64):
    print('[ERROR] Unexpected Recompilation: barrier_hessian')