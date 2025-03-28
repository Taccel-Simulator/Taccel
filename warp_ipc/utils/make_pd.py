import warp as wp
from typing import Any
from .wp_types import mat66d, mat99d, mat12x12d
ENABLE_ABS_PD = True

@wp.func
def eigh(n: wp.int32, A: Any):
    pass

@wp.func
def make_pd(n: wp.int32, A: Any):
    pass

@wp.func
def make_pd_6x6(A: mat66d):
    pass

@wp.func
def make_pd_9x9(A: mat99d):
    pass

@wp.func
def make_pd_12x12(A: mat12x12d):
    pass

@wp.func
def make_pd_3x3(A: wp.mat33d):
    pass

@wp.func
def make_pd_2x2(A: wp.mat22d):
    pass