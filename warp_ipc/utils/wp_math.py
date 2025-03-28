import numpy as np
import warp as wp
import warp.sparse as wps
from scipy.sparse import bsr_matrix
from warp.sparse import BsrMatrix
from .constants import _0, _1, _2, __1
from .wp_types import mat32d, mat33d, mat99d, vec9d, vec12d

@wp.struct
class Complex64:
    real: wp.float64
    imag: wp.float64

    def __init__(self, real: wp.float64=0.0, imag: wp.float64=0.0):
        self.real = real
        self.imag = imag

@wp.func
def add(self: Complex64, other: Complex64) -> Complex64:
    pass

@wp.func
def sub(self: Complex64, other: Complex64) -> Complex64:
    pass

@wp.func
def mul(self: Complex64, other: Complex64) -> Complex64:
    pass

@wp.func
def truediv(self: Complex64, other: Complex64) -> Complex64:
    pass

@wp.func
def magnitude(self: Complex64) -> wp.float64:
    pass

@wp.func
def phase(self: Complex64) -> wp.float64:
    pass

@wp.func
def conjugate(self: Complex64) -> Complex64:
    pass

@wp.func
def pow_c(self: Complex64, exponent: wp.float64) -> Complex64:
    pass

@wp.func
def sqrt_c(self: wp.float64) -> Complex64:
    pass

@wp.func
def inverse_c(self: Complex64) -> Complex64:
    pass

@wp.func
def cat_3_vec3d(a: wp.vec3d, b: wp.vec3d, c: wp.vec3d):
    pass

@wp.func
def cat_4_vec3d(a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, d: wp.vec3d):
    pass

@wp.func
def extract_column_mat33d(mat: wp.mat33d, column: wp.int32):
    pass

@wp.func
def is_nan(x: wp.float64) -> wp.bool:
    pass

@wp.func
def is_pos_inf(x: wp.float64) -> wp.bool:
    pass

@wp.func
def is_neg_inf(x: wp.float64) -> wp.bool:
    pass

@wp.func
def is_inf(x: wp.float64) -> wp.bool:
    pass

@wp.func
def mat_l2_norm_sqr(mat: wp.mat33d):
    pass

@wp.func
def mat_l2_norm_sqr_3x2(mat: mat32d):
    pass

@wp.func
def sqr(x: wp.float64):
    pass

@wp.func
def safe_divide(num: wp.float64, denom: wp.float64):
    pass

@wp.func
def get_combined_coulomb_friction(mu1: wp.float64, mu2: wp.float64):
    pass

@wp.func
def kronecker_3x3_3x3(A: mat33d, B: mat33d) -> mat99d:
    pass

@wp.func
def col_stack2(vec0: wp.vec3d, vec1: wp.vec3d):
    pass

@wp.func
def col_stack3(vec0: wp.vec3d, vec1: wp.vec3d, vec2: wp.vec3d):
    pass

@wp.func
def col_stack3(vec0: wp.vec3d, vec1: wp.vec3d, vec2: wp.vec3d):
    pass

@wp.kernel
def dot_vec3d_arr_kernel(a: wp.array(dtype=wp.vec3d), b: wp.array(dtype=wp.vec3d), products: wp.array(dtype=wp.float64)):
    pass

def dot_vec3d_arr(a: wp.array, b: wp.array):
    assert len(a) == len(b)
    energy = wp.zeros((len(a),), dtype=wp.float64)
    wp.launch(kernel=dot_vec3d_arr_kernel, dim=len(a), inputs=[a, b, energy])
    return wp.utils.array_sum(energy)

@wp.func
def det_derivative_3x3(mat: wp.mat33d):
    pass

@wp.func
def column_flatten_3x3(mat: wp.mat33d):
    pass

@wp.func
def det_3x3(mat: wp.mat33d):
    pass

@wp.func
def from_y_to_x(y: vec12d, w: wp.vec3d):
    pass

@wp.func
def get_smallest_positive_real_quad_root(a: wp.float64, b: wp.float64, c: wp.float64, tol: wp.float64):
    pass

@wp.func
def get_smallest_positive_real_cubic_root(a: wp.float64, b: wp.float64, c: wp.float64, d: wp.float64, img_tol: wp.float64, cubic_coef_tol: wp.float64):
    pass

def scipy_bsr_to_triplets(bsr: bsr_matrix):
    data = bsr.data
    indices = bsr.indices
    indptr = bsr.indptr
    block_size = bsr.blocksize
    (block_rows, block_cols) = block_size
    block_row_indices = []
    block_col_indices = []
    blocks_data = []
    for block_row in range(len(indptr) - 1):
        start = indptr[block_row]
        end = indptr[block_row + 1]
        for idx in range(start, end):
            block_col = indices[idx]
            block_row_indices.append(block_row)
            block_col_indices.append(block_col)
            blocks_data.append(data[idx])
    return (np.stack(block_row_indices, dtype=np.int32, axis=0), np.stack(block_col_indices, dtype=np.int32, axis=0), np.stack(blocks_data, axis=0))

def bsr_scipy_to_warp(A: bsr_matrix, device: str='cuda:0') -> BsrMatrix:
    block_size = A.blocksize
    block_type = wp.mat(shape=block_size, dtype=wp.float64)
    (np_rows, np_cols, np_vals) = scipy_bsr_to_triplets(A)
    rows = wp.from_numpy(np_rows, dtype=wp.int32, device=device)
    cols = wp.from_numpy(np_cols, dtype=wp.int32, device=device)
    vals = wp.from_numpy(np_vals, dtype=block_type, shape=len(np_vals), device=device)
    sm = wps.bsr_zeros(A.shape[0] // block_size[0], A.shape[1] // block_size[1], block_type=block_type, device=device)
    wps.bsr_set_from_triplets(sm, rows, cols, vals)
    return sm

def bsr_warp_to_scipy(A: BsrMatrix) -> bsr_matrix:
    block_size = A.block_shape
    nnz = A.nnz
    nrow = A.nrow
    offsets = A.offsets.numpy()[:nrow + 1]
    columns = A.columns.numpy()[:nnz]
    values = A.values.numpy()[:nnz]
    sparse_A = bsr_matrix((values, columns, offsets), blocksize=block_size, shape=(A.shape[0], A.shape[1]))
    return sparse_A