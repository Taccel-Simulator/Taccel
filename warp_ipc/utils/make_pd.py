from typing import Any
import warp as wp
from .wp_types import mat12x12d, mat66d, mat99d
ENABLE_ABS_PD = True

@wp.func
def eigh(n: wp.int32, A: Any):
    print('[ERROR] Unexpected Recompilation: eigh')

@wp.func
def make_pd(n: wp.int32, A: Any):
    print('[ERROR] Unexpected Recompilation: make_pd')

@wp.func
def make_pd_6x6(A: mat66d):
    print('[ERROR] Unexpected Recompilation: make_pd_6x6')

@wp.func
def make_pd_9x9(A: mat99d):
    print('[ERROR] Unexpected Recompilation: make_pd_9x9')

@wp.func
def make_pd_12x12(A: mat12x12d):
    print('[ERROR] Unexpected Recompilation: make_pd_12x12')

@wp.func
def make_pd_3x3(A: wp.mat33d):
    print('[ERROR] Unexpected Recompilation: make_pd_3x3')

@wp.func
def make_pd_2x2(A: wp.mat22d):
    print('[ERROR] Unexpected Recompilation: make_pd_2x2')