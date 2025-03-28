from typing import Any
import warp as wp
from .wp_math import is_inf, is_nan

@wp.func
def print_mat_val(info: str, mat: Any, n_rows: wp.int32, n_cols: wp.int32):
    pass

@wp.func
def check_mat_val(info: str, mat: Any, n_rows: wp.int32, n_cols: wp.int32):
    pass

@wp.func
def print_vec_val(info: str, vec: Any, n: wp.int32):
    pass

@wp.func
def check_vec_val(info: str, vec: Any, n: wp.int32):
    pass