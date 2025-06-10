import warp as wp
from warp_ipc.utils.wp_types import mat32d

@wp.func
def clamp_zero(x: wp.float64):
    print('[ERROR] Unexpected Recompilation: clamp_zero')

@wp.func
def svd3x2(A: mat32d):
    print('[ERROR] Unexpected Recompilation: svd3x2')